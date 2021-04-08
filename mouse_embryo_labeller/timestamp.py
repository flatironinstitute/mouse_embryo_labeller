
import numpy as np
import json

class Timestamp:

    """
    Information about a timestamp.
    """

    def __init__(self, identifier):
        self.identifier = identifier
        self.raster3d = None
        self.labels3d = None
        self.unique_labels = None
        self.label_to_nucleus = None
        self.r3d_truncated = None
        self.l3d_truncated = None
        self.l3d_extruded = None

    def save_truncated_arrays(self, to_path):
        l = self.l3d_truncated
        r = self.r3d_truncated
        e = self.l3d_extruded
        u = np.array(list(self.unique_labels), dtype=np.int)
        for a in (l, r, e, u):
            assert a is not None, "Timestamp not fully processed for storage " + repr(self.identifier)
        f = open(to_path, "wb")
        np.savez_compressed(f, l3d_truncated=l, r3d_truncated=r, l3d_extruded=e, unique_labels=u)
        f.close()

    def load_truncated_arrays(self, from_path):
        f = open(from_path, "rb")
        L = np.load(f, allow_pickle=True)
        self.l3d_extruded = L["l3d_extruded"]
        self.l3d_truncated = L["l3d_truncated"]
        self.r3d_truncated = L["r3d_truncated"]
        self.unique_labels = set(L["unique_labels"].tolist())
        f.close()

    def nlayers(self):
        return len(self.l3d_truncated)

    def extrude_labels(self):
        "extrude nonzero labels along the z axis"
        l = self.l3d_truncated
        assert l is not None, "labels must be loaded and truncated before extrusion: " + repr(self.identifier)
        extruded = np.zeros(l.shape, dtype=l.dtype)
        mask = extruded[0]
        for i in range(len(extruded)):
            layer = l[i]
            nz = (layer != 0)
            mask = np.choose(nz, [mask, layer])
            extruded[i] = mask
        self.l3d_extruded = extruded

    def load_mapping(self, from_path, nucleus_collection=None):
        f = open(from_path)
        json_info = json.load(f)
        assert json_info["timestamp"] == self.identifier, "wrong timestamp in json file: " + repr((json_info["timestamp"],self.identifier))
        f.close()
        label_to_nucleus_id = json_info["label_to_nucleus_id"]
        label_to_nucleus = {}
        for (label, identifier) in label_to_nucleus_id.items():
            label = int(label)
            if identifier is not None:
                assert nucleus_collection is not None, "no collection -- cannot map nucleus id. " + repr((label, identifier))
                n = nucleus_collection.get_nucleus(identifier)
                label_to_nucleus[label] = n
        self.label_to_nucleus = label_to_nucleus

    def assign_nucleus(self, label, nucleus):
        self.label_to_nucleus[label] = nucleus

    def forget_nucleus(self, nucleus):
        l2n = self.label_to_nucleus
        found = (nucleus in list(l2n.values()))
        if found:
            self.label_to_nucleus = {l: n for (l, n) in l2n.items() if n is not nucleus}
        return found

    def get_nucleus(self, label):
        return self.label_to_nucleus.get(label)

    def colorization_mapping(self, zero_map=(0,0,0), unassigned=(100,100,100)):
        u = self.unique_labels
        ln = max(u) + 1
        result = np.zeros((ln, 3), dtype=np.int)
        n = self.label_to_nucleus
        for i in range(ln):
            nucleus = n.get(i)
            if nucleus is None:
                result[i] = unassigned
            else:
                result[i] = nucleus.color
        result[0] = zero_map
        return result

    def raster_slice(self, slice_i):
        return self.r3d_truncated[slice_i]

    def raster_slice_with_boundary(self, slice_i, extruded=False):
        # xxx could refactor pasted code...
        r_slice = self.raster_slice(slice_i)
        a = self.l3d_truncated
        if extruded:
            a = self.l3d_extruded
        assert a is not None, "Data is not loaded and processed: " + repr(self.identifier)
        l_slice = a[slice_i]
        bound = boundary(l_slice)
        r_slice = np.choose(bound, [r_slice, 0])
        return r_slice

    def get_label(self, layer, i, j, extruded=False):
        a = self.l3d_truncated
        if extruded:
            a = self.l3d_extruded
        return a[layer, i, j]

    def get_intensity(self, layer, i, j):
        a = self.r3d_truncated
        return a[layer, i, j]

    def colorized_label_slice(self, color_mapping_array, slice_i, extruded=False, outline=True):
        a = self.l3d_truncated
        if extruded:
            a = self.l3d_extruded
        assert a is not None, "Data is not loaded and processed: " + repr(self.identifier)
        slice = a[slice_i]
        if outline:
            bound = boundary(slice)
            slice = np.choose(bound, [slice, 0])
        s = slice.shape
        colors = color_mapping_array[slice.flatten()]
        sout = s + (3,)
        return colors.reshape(sout)

    def save_mapping(self, to_path):
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
        }

    def get_truncated_arrays(self, test_array=None):
        r = self.raster3d
        l = self.labels3d
        if test_array is None:
            test_array = l
        (nzI, nzJ, nzK) = np.nonzero(test_array)
        mI, MI = nzI.min(), nzI.max()
        mJ, MJ = nzJ.min(), nzJ.max()
        mK, MK = nzK.min(), nzK.max()
        self.l3d_truncated = l[mI:MI, mJ:MJ, mK:MK]
        self.r3d_truncated = r[mI:MI, mJ:MJ, mK:MK]
        self.extrude_labels()

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
                assert l.shape == r.shape, "labels array and raster array must have same shape."
            if u is None:
                self.unique_labels = set(np.unique(l))
            if n is None:
                self.label_to_nucleus = {label: None for label in range(max(self.unique_labels))}
            else:
                pass # could check validity of mapping


def boundary(array):
    result = np.zeros(array.shape, dtype=np.bool)
    result[:-1, 1:] = (array[1:, 1:] != array[:-1, 1:])
    result[1:, :-1] |= (array[1:, 1:] != array[1:, :-1])
    return result
