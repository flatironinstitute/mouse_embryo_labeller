
import numpy as np
import json
from scipy.ndimage import gaussian_filter

white = np.array([255,255,255])
red = np.array([255,0,0])
black = np.array([0,0,0])

class Timestamp:

    """
    Information about a timestamp.
    """

    def __init__(self, identifier):
        self.identifier = identifier
        self.unique_labels = None
        self.label_to_nucleus = None
        self.manifest = None
        self.array_path = None
        self.special_labels = []
        self.reset_all_arrays()

    def clean_label_to_nucleus(self):
        result = {}
        l2n = self.label_to_nucleus
        if l2n:
            for (k, v) in l2n.items():
                if v is not None:
                    result[k] = v
        return result

    def nuclei_mask_slicing(self, nuclei_names=None):
        from mouse_embryo_labeller import geometry
        self.load_truncated_arrays()
        l3d = self.l3d_truncated
        # release memory
        self.reset_all_arrays()
        mask = np.zeros(l3d.shape, dtype=np.int)
        #l2n = self.label_to_nucleus
        l2n = self.clean_label_to_nucleus()
        if nuclei_names is None:
            labels = l2n.keys()
        else:
            labels = [label for (label, n) in l2n.items() if n.identifier in nuclei_names]
        for label in labels:
            mask += (l3d == label)
        result = geometry.positive_slicing(mask)
        #(self.identifier, "masking", list(labels), "labels", result)
        return result

    def assign_index(self, index):
        for nucleus in self.clean_label_to_nucleus().values():
            nucleus.add_timestamp_index(index)

    def nucleus_names(self):
        return set(n.identifier for n in self.clean_label_to_nucleus().values())

    def label_colors(self, nucleus_names=None):
        label_to_color = {}
        for (label, nucleus) in self.clean_label_to_nucleus().items():
            if (nucleus_names is None) or (nucleus.identifier in nucleus_names):
                label_to_color[label] = list(nucleus.color)
        return label_to_color

    def reset_all_arrays(self):
        self.raster3d = None
        self.labels3d = None
        self.r3d_truncated = None
        self.r3d_max_intensity = None
        self.l3d_truncated = None
        self.l3d_extruded = None

    def __repr__(self):
        return "Timestamp(%s)" % repr(self.identifier)

    def save_array_path(self, path):
        #print(self.identifier, "array path", path)
        self.array_path = path

    def save_truncated_arrays(self, to_path, discard=True):
        l = self.l3d_truncated
        r = self.r3d_truncated
        e = self.l3d_extruded
        m = self.r3d_max_intensity
        u = np.array(list(self.unique_labels), dtype=np.int)
        for (i, a) in enumerate([l, r, e, u, m]):
            assert a is not None, "Timestamp not fully processed for storage " + repr((i, self.identifier))
        f = open(to_path, "wb")
        np.savez_compressed(f, l3d_truncated=l, r3d_truncated=r, l3d_extruded=e, r3d_max_intensity=m, unique_labels=u)
        f.close()
        if discard:
            # discard the arrays to prevent leaking memory problem
            self.reset_all_arrays()

    def load_truncated_arrays(self, from_path=None):
        if from_path is None:
            from_path = self.array_path
        #print(self.identifier, "loading arrays", repr(from_path))
        f = open(from_path, "rb")
        L = np.load(f, allow_pickle=True)
        self.l3d_extruded = L["l3d_extruded"]
        self.l3d_truncated = L["l3d_truncated"]
        self.r3d_truncated = L["r3d_truncated"]
        self.r3d_max_intensity = L["r3d_max_intensity"]
        self.unique_labels = set(L["unique_labels"].tolist())
        f.close()

    def nlayers(self):
        return len(self.l3d_truncated)

    def extrude_labels(self):
        "extrude nonzero labels along the z axis and compute max intensity extrusion"
        l = self.l3d_truncated
        r = self.r3d_truncated
        assert l is not None, "labels must be loaded and truncated before extrusion: " + repr(self.identifier)
        extruded = np.zeros(l.shape, dtype=l.dtype)
        max_intensity = np.zeros(r.shape, dtype=r.dtype)
        mask = extruded[0]
        max_mask = r[0].copy()
        for i in range(len(extruded)):
            layer = l[i]
            nz = (layer != 0)
            mask = np.choose(nz, [mask, layer])
            max_mask = np.maximum(r[i], max_mask)
            extruded[i] = mask
            max_intensity[i] = max_mask
        assert extruded.shape == max_intensity.shape[:3], "shapes should match: " + repr((extruded.shape, max_intensity.shape))
        self.r3d_max_intensity = max_intensity
        self.l3d_extruded = extruded

    def load_mapping(self, from_path=None, nucleus_collection=None):
        if from_path is None:
            from_path = self.save_path
        f = open(from_path)
        json_info = json.load(f)
        self.special_labels = json_info.get("special_labels", [])
        assert json_info["timestamp"] == self.identifier, "wrong timestamp in json file: " + repr((json_info["timestamp"],self.identifier))
        f.close()
        label_to_nucleus_id = json_info["label_to_nucleus_id"]
        label_to_nucleus = {}
        for (label, identifier) in label_to_nucleus_id.items():
            label = int(label)
            if identifier is not None:
                assert nucleus_collection is not None, "no collection -- cannot map nucleus id. " + repr((label, identifier))
                n = nucleus_collection.get_nucleus(identifier, check=False)
                # silently ignore bogus nucleus?
                if n is not None:
                    label_to_nucleus[label] = n
        self.save_path = from_path
        self.label_to_nucleus = label_to_nucleus

    def assign_nucleus(self, label, nucleus):
        l2n = self.label_to_nucleus
        if (nucleus is None) and (label in l2n):
            del l2n[label]
        else:
            l2n[label] = nucleus
        self.save_mapping()

    def forget_nucleus(self, nucleus):
        l2n = self.label_to_nucleus
        found = (nucleus in list(l2n.values()))
        if found:
            self.label_to_nucleus = {l: n for (l, n) in l2n.items() if n is not nucleus}
            self.save_mapping()
        return found

    def relabel(self, old_nucleus, replacement_nucleus):
        l2n = self.clean_label_to_nucleus()
        found = (old_nucleus in list(l2n.values()))
        if found:
            new_l2n = {}
            for (label, nucleus) in l2n.items():
                if nucleus is old_nucleus:
                    nucleus = replacement_nucleus
                if nucleus is not None:
                    new_l2n[label] = nucleus
            self.label_to_nucleus = new_l2n
            self.save_mapping()
        return found

    def get_nucleus(self, label):
        return self.label_to_nucleus.get(label)

    def colorization_mapping(self, id_to_nucleus=None, zero_map=(0,0,0), unassigned=(100,100,100)):
        u = self.unique_labels
        ln = max(u) + 1
        result = np.zeros((ln, 3), dtype=np.int)
        n = self.label_to_nucleus
        for i in range(ln):
            nucleus = n.get(i)
            color_choice = unassigned
            if nucleus is not None:
                if (id_to_nucleus is None) or (nucleus.identifier in id_to_nucleus):
                    color_choice = nucleus.color
            result[i] = color_choice
        result[0] = zero_map
        return result

    def raster_slice(self, slice_i, extruded=False):
        if extruded:
            r = self.r3d_max_intensity
        else:
            r = self.r3d_truncated
        return r[slice_i]

    def raster_slice_with_boundary(self, slice_i, extruded=False, colorize=True, blur=True, normalized=True):
        # xxx could refactor pasted code...
        r_slice = self.raster_slice(slice_i, extruded)
        if blur:
            sigma = 1
            blurred_image = gaussian_filter(r_slice, sigma=sigma)
            r_slice = blurred_image
        if colorize:
            if len(r_slice.shape) == 2:
                r_slice = false_colors(r_slice)
        elif normalized:
            r_slice = normalize(r_slice)
        a = self.l3d_truncated
        if extruded:
            a = self.l3d_extruded
        assert a is not None, "Data is not loaded and processed: " + repr(self.identifier)
        l_slice = a[slice_i]
        #bound = boundary(l_slice)
        #if colorize or (len(r_slice.shape) > len(bound.shape)):
        #    # make broadcast compatible
        #    bound = bound.reshape(bound.shape + (1,))
        #r_slice = np.choose(bound, [r_slice, 255])
        special_labels = self.special_labels
        r_slice = special_boundary(l_slice, special_labels, background_color=r_slice)
        return r_slice

    def get_label(self, layer, i, j, extruded=False):
        a = self.l3d_truncated
        if extruded:
            a = self.l3d_extruded
        return a[layer, i, j]

    def get_intensity(self, layer, i, j, extruded):
        #a = self.r3d_truncated
        r = self.raster_slice(layer, extruded)
        return r[i, j]

    def colorized_label_slice(self, color_mapping_array, slice_i, extruded=False, outline=True):
        a = self.l3d_truncated
        if extruded:
            a = self.l3d_extruded
        assert a is not None, "Data is not loaded and processed: " + repr(self.identifier)
        slice = a[slice_i]
        #if outline:
        ##    bound = boundary(slice)
        #    slice = np.choose(bound, [slice, 0])
        s = slice.shape
        colors = color_mapping_array[slice.flatten()]
        if outline:
            #bound = boundary(slice)
            #white = np.array([255,255,255], dtype=np.int).reshape((1, 3))
            #fbound = bound.flatten()
            #cbound = np.zeros(colors.shape, dtype=np.int)
            #cbound[:] = fbound.reshape(fbound.shape + (1,))
            #colors = np.choose(cbound, [colors, white])
            special_labels = self.special_labels
            colors = colors.reshape(slice.shape + (3,))
            colors = special_boundary(slice, special_labels, background_color=colors)
        sout = s + (3,)
        return colors.reshape(sout)

    def save_mapping(self, to_path=None):
        if to_path is None:
            to_path = self.save_path
        self.save_path = to_path
        f = open(to_path, "w")
        json_info = self.json_mapping()
        json.dump(json_info, f, indent=2)
        f.close()

    def json_mapping(self):
        label_to_nucleus_id = {}
        n = self.label_to_nucleus
        assert n is not None, "Mapping not loaded: " + repr(self.identifier)
        for label in n:
            nucleus = n[label]
            identity = (None if nucleus is None else nucleus.identifier)
            label = str(label)
            label_to_nucleus_id[label] = identity
        return {
            "timestamp": self.identifier,
            "label_to_nucleus_id": label_to_nucleus_id,
            "special_labels": [int(x) for x in self.special_labels],
        }

    def get_truncated_arrays(self, test_array=None):
        # test_array not used for now
        r = self.raster3d
        l = self.labels3d
        self.l3d_truncated = l
        self.r3d_truncated = r
        self.extrude_labels()
        return

    def add_source_arrays(self, raster3d, labels3d):
        self.raster3d = raster3d
        self.labels3d = labels3d
        self.check()

    def check(self):
        r = self.raster3d
        l = self.labels3d
        u = self.unique_labels
        n = self.label_to_nucleus
        if l is not None:
            if r is not None:
                assert l.shape == r.shape[:3], (
                    "labels array and raster array must have same shape. " + repr([l.shape, r.shape])
                )
            if u is None:
                self.unique_labels = set(np.unique(l))
            if n is None:
                self.label_to_nucleus = {label: None for label in range(max(self.unique_labels))}
            else:
                pass # could check validity of mapping

def false_colors(array):
    byte_array = normalize(array)
    shape = array.shape
    flat_bytes = byte_array.flatten()
    flat_colorized = COLORMAP[flat_bytes]
    return flat_colorized.reshape(shape + (3,))

def normalize(array):
    "adjust contrast for array to range 0..255"
    farray = array.astype(np.float)
    m = farray.min()
    M = farray.max()
    M = max(M, m + 0.001)
    narray = (farray - m) / (M - m)
    byte_array = (narray * 255).astype(np.int)
    return byte_array

def make_color_map():
    colormap = np.zeros((256, 3), dtype=np.int)
    for i in range(128):
        i2 = 2 * i
        im = 256 - i2
        # first half from blue to green
        colormap[i] = (0, i2, im)
        # second half from green to red
        colormap[i + 128] = (im, 0, i2)
    return colormap

COLORMAP = make_color_map()

def boundary(array):
    """
    Generate an outline array demarcating uniform labelled regions from the input array.
    """
    result = np.zeros(array.shape, dtype=np.bool)
    result[:-1, 1:] = (array[1:, 1:] != array[:-1, 1:])
    result[1:, :-1] |= (array[1:, 1:] != array[1:, :-1])
    return result

def select_labels(labels_array, selected_labels):
    """
    Restrict the labels array to the selected labels.
    Return array of zeros except where labels_array has one of the selected labels.
    """
    selected_labels = set(selected_labels)
    flat_array = labels_array.ravel()
    flat_result = np.zeros(flat_array.shape, dtype=flat_array.dtype)
    labels = np.unique(flat_array)
    limit = labels.max() + 1
    chooser = list(range(limit))
    for i in range(limit):
        if i not in selected_labels:
            chooser[i] = 0
    #chooser[0] = flat_result
    #flat_result = np.choose(flat_array, chooser)
    flat_result = big_choose(flat_array, chooser)
    result = flat_result.reshape(labels_array.shape)
    return result


def special_boundary(labels_array, special_labels, normal_color=white, special_color=red, background_color=black):
    """
    Return labelled areas in labels_array outlined with boundaries.
    Use the special_color if the boundary marks a region from special_labels,
    otherwise use the normal_color.
    Fill non-outline pixels using the background_color (which can be an array or a single color).
    """
    normal_color = np.array(normal_color).reshape((1,3))
    special_color = np.array(special_color).reshape((1,3))
    background_color = np.array(background_color)
    background_dim = len(background_color.shape)
    if background_dim == 1:
        # solid background color
        background_color = background_color.reshape((1,3))
    elif background_dim == 2:
        # monotone raster
        background_color = background_color.reshape((labels_array.size, 1))
    else:
        # full color background
        assert background_dim == 3, "only backgrounds up to 3 dimensions supported " + repr(background_color.shape)
        background_color = background_color.reshape((labels_array.size, 3))
    special_array = select_labels(labels_array, special_labels)
    special_boundary = boundary(special_array).astype(np.int)
    normal_boundary = boundary(labels_array).astype(np.int)
    boundary_selector = np.maximum(special_boundary * 2, normal_boundary).ravel().reshape((labels_array.size, 1))
    colorized_ravel = np.choose(boundary_selector, [background_color, normal_color, special_color])
    colorized_boundaries = colorized_ravel.reshape(labels_array.shape + (3,))
    return colorized_boundaries


def big_choose(indices, choices):
    "Alternate to np.choose that supports more than 30 choices."
    indices = np.array(indices)
    if (indices.max() <= 30) or (len(choices) <= 31):
        # optimized fallback
        choices = choices[:31]
        return np.choose(indices, choices)
    result = 0
    while (len(choices) > 0) and not np.all(indices == -1):
        these_choices = choices[:30]
        remaining_choices = choices[30:]
        shifted_indices = indices + 1
        too_large_indices = (shifted_indices > 30).astype(np.int)
        clamped_indices = np.choose(too_large_indices, [shifted_indices, 0])
        choices_with_default = [result] + list(these_choices)
        result = np.choose(clamped_indices, choices_with_default)
        choices = remaining_choices
        if len(choices) > 0:
            indices = indices - 30
            too_small = (indices < -1).astype(np.int)
            indices = np.choose(too_small, [indices, -1])
    return result

def test():
    x = np.arange(15).reshape((5,3))
    print(x)
    print (false_colors(x))

if __name__ == "__main__":
    test()
