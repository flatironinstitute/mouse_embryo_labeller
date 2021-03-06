
from mouse_embryo_labeller.timestamp import COLORMAP
from mouse_embryo_labeller.nucleus import nucleus_from_json
import numpy as np
from pprint import pprint
from mouse_embryo_labeller import ellipsoid_fit

def positive_slicing(M):
    """
    for 3d matrix M determine I,J,K (start, end) slicing of minimal volume containing all positive M[i,j,k]
    """
    slices = np.zeros((3, 2), dtype=np.int)
    Itest = M.max(axis=2).max(axis=1)
    (inz,) = np.nonzero(Itest > 0)
    slices[0] = (inz.min(), inz.max()+1)
    Jtest = M.max(axis=2).max(axis=0)
    (jnz,) = np.nonzero(Jtest > 0)
    slices[1] = (jnz.min(), jnz.max()+1)
    Ktest = M.max(axis=1).max(axis=0)
    (knz,) = np.nonzero(Ktest > 0)
    slices[2] = (knz.min(), knz.max()+1)
    return slices

def slice3(M, s):
    "Slice M by array generated by positive_slicing."
    # xxx not neeeded?  For completeness
    im, iM = s[0]
    jm, jM = s[1]
    km, kM = s[2]
    return M[im:iM, jm:jM, km:kM]

def unify_slicing(s1, s2):
    "combine slicings s1 and s2 into smallest slicing containing both s1 and s2."
    slices = np.zeros((3, 2), dtype=np.int)
    slices[:, 0] = np.minimum(s1[:, 0], s2[:, 0])
    slices[:, 1] = np.maximum(s1[:, 1], s2[:, 1])
    return slices

def positive_extent_info(array):
    "Compute slicing statistics dietionary for array"
    slicing = positive_slicing(array)
    return slicing_info(slicing)

def apply_slicing(slicing, to_array):
    [[i0, i1], [j0, j1], [k0, k1]] = slicing
    sliced = to_array[i0: i1, j0: j1, k0: k1]
    return sliced

# field names
SLICING = "slicing"
DIMENSIONS = 'dimensions'
CENTER = "center"
TIMESTAMPS = 'timestamps'
LABELS = 'labels'
RADIUS = "radius"
COLOR = "color"
NUCLEUS_ID = "nucleus_id"

def slicing_info(slicing):
    "Compute JSON compatible slicing statistics dictionary for slicing."
    m = slicing[:, 0]
    M = slicing[:, 1]
    dimensions = 0.5 * (M - m)
    radius = max(dimensions)
    center = 0.5 * (M + m)
    return dict(
        slicing=slicing.tolist(), 
        dimensions=dimensions.tolist(), 
        center=center.tolist(),
        radius=radius,
        )

def unify_slicing_info(info1, info2):
    "slicing info after unifying the slicings"
    s1 = np.array(info1[SLICING])
    s2 = np.array(info2[SLICING])
    u = unify_slicing(s1, s2)
    return slicing_info(u)

def center_point(slicing_info):
    "For interpolation: 0 valume slicing with same center."
    result = slicing_info.copy()
    result[RADIUS] = 0
    result[DIMENSIONS] = [0, 0, 0]
    return result

def fa(x):
    return np.array(x, dtype=np.float)

def tl(x):
    return x.tolist()

def interpolate_slicing_infos(slicing0, slicing1, lambda1):
    """
    Interpolate between slicing0 and slicing1 (lambda1=0 gives slicing1, lambda1=1 gives slicing1).
    """
    lambda0 = 1.0 - lambda1
    result = slicing0.copy()
    result.update(slicing1)
    result[RADIUS] = lambda0 * slicing0[RADIUS] + lambda1 * slicing1[RADIUS]
    result[DIMENSIONS] = tl(lambda0 * fa(slicing0[DIMENSIONS]) + lambda1 * fa(slicing1[DIMENSIONS]))
    result[CENTER] = tl(lambda0 * fa(slicing0[CENTER]) + lambda1 * fa(slicing1[CENTER]))
    return result

def timestamp_geometry(ts):
    # not optimized -- scans the entire array many more times than technically needed
    ts.load_truncated_arrays()
    labels = ts.unique_labels
    label_array = ts.l3d_truncated
    l2n = ts.label_to_nucleus
    label_geometries = {}
    slicing = None
    for label in labels:
        # skip 0
        if label == 0:
            continue
        test_array = (label_array == label)
        string_label = str(label)  # json requires string keys
        g = label_geometries[string_label] = positive_extent_info(test_array)
        nucleus = l2n.get(label)
        color = None
        nucleus_id = None
        if nucleus is not None:
            color = nucleus.color.tolist()
            nucleus_id = nucleus.identifier
        g[COLOR] = color
        g[NUCLEUS_ID] = nucleus_id
        label_slicing = np.array(g["slicing"])
        if slicing is None:
            slicing = label_slicing
        else:
            slicing = unify_slicing(slicing, label_slicing)
    geometry = slicing_info(slicing)
    geometry["labels"] = label_geometries
    ts.geometry = geometry
    ts.reset_all_arrays()  # dont' hog memory
    return geometry

def preprocess_geometry_to_json(tsc, json_path, di=2.0, dj=0.208, dk=0.208):
    import json
    f = open(json_path, "w")
    geometry = timestamp_collection_geometry(tsc)
    geometry = add_ellipsoids(tsc, geometry, di=di, dj=dj, dk=dk)
    json.dump(geometry, f, indent=4, sort_keys=True)
    f.close()
    print("Dumped json geometry to", repr(json_path))

class GeometryViewer:

    def __init__(self, tsc_geometry):
        dummy_points = np.zeros((3,3))  # just to get access to fitter methods
        self.fitter = ellipsoid_fit.EllipsoidFitter(dummy_points)
        g = self.geometry = tsc_geometry
        self.dimensions = np.array(g["dimensions"])
        self.distortion = np.array(g["distortion"])
        self.offset = self.dimensions * self.distortion
        self.radius = self.offset.max()
        self.timestamp = None
        self._widget = None
        self.animation_arrays = []

    def ts_geometry(self, tsid):
        from mouse_embryo_labeller import ellipsoid_fit
        C = ellipsoid_fit.Combined3DEllipseGeometries()
        (tsinfo, tsg) = self.get_ts_info(tsid)
        # add outer ellipse
        matrix = tsinfo.Minv
        C.add(matrix, [25,25,55])
        labels = tsg["labels"]
        for info in labels.values():
            color = info.get("color")
            if color is None:
                color = [100, 100, 100]
            lellipse = info["ellipse"]
            linfo = ellipsoid_fit.EllipsoidInfo(lellipse)
            matrix = linfo.Minv
            C.add(matrix, color)
        return C

    def geometry_json(self):
        g = self.geometry
        tss = g["timestamps"]
        tsids = sorted([int(x) for x in tss.keys()])
        geometry_jsons = []
        for tsid in tsids:
            C = self.ts_geometry(tsid)
            g_json = C.flat_json_dump(tsid)
            geometry_jsons.append(g_json)
        inner = ",\n".join(geometry_jsons)
        return "[\n%s\n]" % inner

    def save_json(self, to_path):
        f = open(to_path, 'w')
        json = self.geometry_json()
        f.write(json)
        f.close()

    def widget(self, side=800):
        from jp_doodle import dual_canvas, nd_frame
        import ipywidgets as widgets
        from jp_doodle.data_tables import widen_notebook
        widen_notebook()
        g = self.geometry
        tss = g["timestamps"]
        tsids = sorted([int(x) for x in tss.keys()])
        self.tsids = tsids
        minid = min(tsids)
        maxid = max(tsids)
        ellipsoid_canvas = self.ellipsoid_canvas = dual_canvas.DualCanvasWidget(width=side, height=side)
        (maxx, maxy, maxz) = self.offset * 2
        self.ellipsoid2d = ellipsoid_canvas.frame_region(
            0, 0, side, side,
            0, 0, maxx, maxy,
        )
        self.ellipsoid3d = nd_frame.ND_Frame(self.ellipsoid_canvas, self.ellipsoid2d)
        r = self.radius
        self.ellipsoid3d.orbit(center3d=self.offset, radius=3*r, shift2d=(r/2, r/3))
        self.fitter.draw_box(self.ellipsoid3d, self.offset, self.offset * 1.2, "pink")
        ellipsoid_canvas.fit()
        self.info = widgets.HTML(value="Info here.")
        self.draw_ts(minid) # temp
        self.slider = widgets.IntSlider(description="timestamp", min=minid, max=maxid, value=minid)
        self.slider.observe(self.change_timestamp, "value")
        widget = widgets.VBox(children=[self.slider, ellipsoid_canvas, self.info])
        #widget = ellipsoid_canvas
        self._widget = widget
        return widget

    def get_animation_arrays(self):
        import time
        assert self.widget is not None, "Please create a widget before making an animation."
        tsids = self.tsids
        arrays = self.animation_arrays = []
        for tsid in tsids:
            self.draw_ts(tsid)
            array = self.ellipsoid_canvas.pixels_array()
            # make the array non-transparent
            alpha = array[:,:,3].copy()
            alpha = alpha.reshape(alpha.shape + (1,))
            white = np.array([255,255,255,255]).reshape((1,1,4))
            array1 = np.where(alpha > 0, array, white).astype(np.uint8)
            array1[:,:,3] = 255
            arrays.append(array1)
        return arrays

    def save_animated_gif(self, to_path="geometry.gif", duration=0.2):
        import imageio
        arrays = self.animation_arrays
        if not arrays:
            arrays = self.get_animation_arrays()
        # add ten frames fo the last array to the end of the animation
        arrays = list(arrays) + ([arrays[-1]] * 10)
        imageio.mimsave(to_path, arrays, format="GIF", duration=duration)
        print("Saved", to_path, "as animated GIF.")

    def change_timestamp(self, ignored):
        timestamp = self.slider.value
        self.draw_ts(timestamp)

    def get_ts_info(self, tsid):
        g = self.geometry
        tss = g["timestamps"]
        tsg = tss.get(str(tsid))
        tsellipse = tsg["ellipse"]
        tsinfo = ellipsoid_fit.EllipsoidInfo(tsellipse)
        return (tsinfo, tsg)

    def draw_ts(self, tsid):
        #print ("drawing ts", tsid)
        self.info.value = "tsid: " + repr(tsid)
        self.timestamp = tsid
        g = self.geometry
        tss = g["timestamps"]
        tsg = tss.get(str(tsid))
        tsellipse = tsg["ellipse"]
        tsinfo = ellipsoid_fit.EllipsoidInfo(tsellipse)
        with self.ellipsoid_canvas.delay_redraw():
            self.ellipsoid3d.reset()
            color = "rgb(130,200,255)"
            self.fitter.draw_circles(self.ellipsoid3d, tsinfo.Minv, color=color)
            labels = tsg["labels"]
            for info in labels.values():
                color = info.get("color")
                if color is None:
                    color = [150,150,150]
                scolor = "rgb%s" % (tuple(color),)
                #print(scolor)
                lellipse = info["ellipse"]
                linfo = ellipsoid_fit.EllipsoidInfo(lellipse)
                self.fitter.draw_circles(self.ellipsoid3d, linfo.Minv, color=scolor)
                #break
        r = self.radius
        off = self.offset
        self.ellipsoid3d.orbit_all(2 * r, off)


def timestamp_collection_geometry(tsc):
    """
    Compute extents for labels with a timestamp as slicing_info dictionaries.  
     Example structure:
     
     geometry = {
        'slicing': [[11, 44], [44, 169], [77, 183]],
        'dimensions': [16.5, 62.5, 53.0],
        'center': [27.5, 106.5, 130.0],
        'radius': 62.5,
        'timestamps': {
            '0': {  # ts 0 geometry
                'slicing': [[12, 39], [72, 157], [92, 183]],
                'dimensions': [13.5, 42.5, 45.5],
                'center': [25.5, 114.5, 137.5],
                'labels': { # labels for ts 0
                    '1': {
                        'slicing': [[12, 19], [78, 98], [128, 145]],
                        'dimensions': [3.5, 10.0, 8.5],
                        'center': [15.5, 88.0, 136.5],
                        'color': None,
                        'nucleus_id': None
                        },
                    '2': {
                        'slicing': [[14, 21], [112, 132], [122, 141]],
                        'dimensions': [3.5, 10.0, 9.5],
                        'center': [17.5, 122.0, 131.5],
                        'color': None,
                        'nucleus_id': None
                        },
                    '3': {
                        'slicing': [[16, 24], [104, 123], [164, 183]],
                        'dimensions': [4.0, 9.5, 9.5],
                        'center': [20.0, 113.5, 173.5],
                        'color': None,
                        'nucleus_id': None
                        },
                    '4': {
                        'slicing': [[25, 32], [72, 90], [102, 121]],
                        'dimensions': [3.5, 9.0, 9.5],
                        'center': [28.5, 81.0, 111.5],
                        'color': [255, 255, 38],
                        'nucleus_id': 'A'
                        },  
                    }, # end of labels for ts 0
                },  # end of ts 0 geometry
            "1": {
                # ts 1 description...
                },  # ...
        } # end of ts geometries
    }
    """
    id2ts = tsc.id_to_timestamp
    slicing = None
    ts_geometries = {}
    for (id, ts) in id2ts.items():
        print("slicing ts", ts)
        str_id = str(id)  # json requires string keys
        g = ts_geometries[str_id] = timestamp_geometry(ts)
        ts_slicing = np.array(g["slicing"])
        if slicing is None:
            slicing = ts_slicing
        else:
            slicing = unify_slicing(slicing, ts_slicing)
    geometry = slicing_info(slicing)
    geometry["timestamps"] = ts_geometries
    tsc.geometry = geometry
    return geometry

def add_ellipsoids(tsc, tsc_geometry, di=2.0, dj=0.208, dk=0.208):
    "Add embryo and nucleus ellipsoids to timestamp collection geometry."
    slicing = tsc_geometry["slicing"]
    tsc_geometry["distortion"] = [di, dj, dk]
    id2ts = tsc.id_to_timestamp
    ts_geometries = tsc_geometry["timestamps"]
    for (id, ts) in id2ts.items():
        print("surrounding ts", id)
        str_id = str(id)  # json requires string keys
        tsg = ts_geometries[str_id]
        ts.load_truncated_arrays()
        labels = ts.unique_labels
        label_array = ts.l3d_truncated
        #l2n = ts.label_to_nucleus
        sliced_labels = apply_slicing(slicing, label_array)
        fitter = ellipsoid_fit.ArrayFitter(sliced_labels, di, dj, dk)
        info = fitter.fit_ellipse_to_range(lower_limit=1)
        tsg["ellipse"] = info.json_object()
        label_geometries = tsg["labels"]
        for label in labels:
            if label == 0:
                continue
            print("    surrounding label", label)
            string_label = str(label)  # json requires string keys
            info = fitter.fit_ellipse_to_range(lower_limit=label, upper_limit=label)
            lg = label_geometries[string_label]
            lg["ellipse"] = info.json_object()
        ts.reset_all_arrays()  # dont' hog memory
    return tsc_geometry

def draw_octohedron(on_frame_3d, center, dimensions, color, opacity=0.3, shrink=0.7, distortion=[3,1,1], outline_color="rgb(100,100,100)"):
    # XXXX NOT USED
    center = np.array(center, dtype=np.float).reshape((1,3))
    dimensions = np.array(dimensions, dtype=np.float).reshape((1,3))
    distortion = np.array(distortion, dtype=np.float).reshape((1,3))
    [r,g,b] = color
    color_str = "rgba(%s,%s,%s,%s)" % (r, g, b, opacity)
    #print ("color", color_str)
    triangle_template = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float)
    for I in (-1, 1):
        for J in (-1, 1):
            for K in (-1, 1):
                IJK = np.array([I, J, K], dtype=np.float)
                triangle = (center + triangle_template * (IJK * dimensions)) * distortion
                #print ("triangle", triangle)
                tcenter = triangle.mean(axis=0).reshape((1,3))
                toffset = triangle - tcenter
                shrink_triangle = tcenter + shrink * toffset
                #print ("shrink_triangle", shrink_triangle)
                if on_frame_3d is not None:
                    on_frame_3d.polygon(shrink_triangle, color=color_str)
                    on_frame_3d.polygon(shrink_triangle, color=outline_color, fill=False)
    return triangle

def draw_box(on_frame_3d, dimensions, center, distortion, color_str="red"):
    corners = [(i, j, k) for i in (-1,1) for j in (-1,1) for k in (-1,1)]
    #scale = np.array(dimensions) * np.array(distortion)
    corners = np.array(corners, dtype=np.int)
    for c1 in corners:
        for c2 in corners:
            if np.all(c1 >= c2) and (c1 - c2).sum() == 2:
                p1 = distortion * (center + c1 * dimensions)
                p2 = distortion * (center + c2 * dimensions)
                on_frame_3d.line(p1, p2, color=color_str)

def ts_swatch(ts, pixels=700, distortion=[3,1,1]):
    from jp_doodle import nd_frame
    geometry = timestamp_geometry(ts)
    distortion = np.array(distortion, dtype=np.float)
    center = np.array(geometry[CENTER])
    center_d = distortion * center
    dimensions = geometry[DIMENSIONS]
    #radius = geometry[RADIUS]
    radius = max(np.array(dimensions) * distortion)
    swatch = nd_frame.swatch3d(pixels=pixels, model_height=radius * 2)
    swatch.orbit(center3d=center_d, radius=radius, shift2d=(-0.25 * radius, -0.2 * radius))
    swatch.orbit_all(center3d=center_d, radius=radius)
    draw_box(swatch, dimensions, center, distortion)
    draw_ts_geometry(geometry, swatch, distortion=distortion)
    swatch.fit(0.6)
    return swatch

def draw_ts_geometry(geometry, frame3d, default_color=[0,0,0], distortion=[3,1,1]):
    #with frame3d.in_canvas.delay_redraw():
    for label_geometry in geometry[LABELS].values():
        center = label_geometry[CENTER]
        dimensions = label_geometry[DIMENSIONS]
        color = label_geometry[COLOR]
        if color is None:
            color = default_color
        draw_octohedron(frame3d, center, dimensions, color, distortion=distortion)

def interpolate_timestamps(ts0, ts1, include_orphans=True):
    return TimeStampInterpolator(timestamp_geometry(ts0), timestamp_geometry(ts1), include_orphans=include_orphans)

class TimeStampInterpolator:
    
    def __init__(self, ts0_geometry, ts1_geometry, pixels=1000, distortion=[3,1,1], include_orphans=True):
        self.pixels = pixels
        self.distortion = distortion
        self.ts0_geometry = ts0_geometry
        self.ts1_geometry = ts1_geometry
        self.unified = unify_slicing_info(ts0_geometry, ts1_geometry)
        self.make_pairings(include_orphans=include_orphans)

    def make_pairings(self, include_orphans=True):
        paired0 = set()
        paired1 = set()
        pairs = []
        nucleus2label0 = {}
        nucleus2label1 = {}
        labels0 = self.ts0_geometry[LABELS]
        labels1 = self.ts1_geometry[LABELS]
        for (label, descr) in labels0.items():
            nid = descr[NUCLEUS_ID]
            if nid is not None:
                nucleus2label0[nid] = label
        for (label, descr) in labels1.items():
            nid = descr[NUCLEUS_ID]
            if nid is not None:
                nucleus2label1[nid] = label
        for nid in nucleus2label0:
            if nid in nucleus2label1:
                label1 = nucleus2label1[nid]
                label0 = nucleus2label0[nid]
                descr1 = labels1[label1]
                descr0 = labels0[label0]
                pair = (descr0, descr1)
                pairs.append(pair)
                paired0.add(label0)
                paired1.add(label1)
        for (label, descr) in labels0.items():
            if label not in paired0:
                if include_orphans or (descr[NUCLEUS_ID] is not None):
                    point = center_point(descr)
                    pair = (descr, point)
                    pairs.append(pair)
        for (label, descr) in labels1.items():
            if label not in paired1:
                if include_orphans or (descr[NUCLEUS_ID] is not None):
                    point = center_point(descr)
                    pair = (point, descr)
                    pairs.append(pair)
        self.pairs = pairs

    def interpolate_pairings(self, lambda1):
        result = self.ts1_geometry.copy()
        labels = {}
        for (index, (l0, l1)) in enumerate(self.pairs):
            labels[str(index)] = interpolate_slicing_infos(l0, l1, lambda1)
        result[LABELS] = labels
        return result

    def make_assembly(self, pixels=None):
        import ipywidgets as widgets
        if pixels is not None:
            self.pixels = pixels
        self.get_swatch()
        self.slider = widgets.FloatSlider(
            value=0,
            min=0.0,
            max=1.0,
            step=0.01,
            description='Interpolation:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=dict(width=str(self.pixels) + "px")
        )
        self.slider.observe(self.slide_interpolation, "value")
        self.assembly = widgets.VBox([
            self.canvas,
            self.slider,
        ])
        self.slide_interpolation(None)
        return self.assembly

    last_interpolation = None

    def slide_interpolation(self, change):
        value = self.slider.value
        #print("slide", value)
        interpolation = self.interpolate_pairings(value)
        #pprint(interpolation)
        self.last_interpolation = interpolation
        with self.swatch.in_canvas.delay_redraw():
            self.swatch.reset()
            draw_box(self.swatch, self.dimensions, self.center, self.distortion)
            draw_ts_geometry(interpolation, self.swatch)
            self.swatch.orbit_all(center3d=self.center_d, radius=self.radius)
        
    def get_swatch(self):
        pixels = self.pixels
        #distortion = self.distortion
        from jp_doodle import nd_frame
        geometry = self.unified
        self.distortion = np.array(self.distortion, dtype=np.float)
        self.center = np.array(geometry[CENTER])
        self.dimensions = np.array(geometry[DIMENSIONS])
        self.center_d = self.center * self.distortion
        #radius = geometry[RADIUS]
        self.radius = max(self.dimensions * self.distortion)
        swatch = nd_frame.swatch3d(pixels=pixels, model_height=self.radius * 2, auto_show=False)
        swatch.orbit(center3d=self.center_d, radius=self.radius, shift2d=(-0.25 * self.radius, -0.2 * self.radius))
        swatch.orbit_all(center3d=self.center_d, radius=self.radius)
        draw_box(swatch, self.dimensions, self.center, self.distortion)
        #draw_ts_geometry(geometry, swatch, distortion=distortion)
        swatch.fit(0.8)
        self.swatch = swatch
        self.canvas = swatch.in_canvas
        return swatch

def interpolate_timestamp_collection(tsc, include_orphans=False):
    return TimeStampCollectionInterpolator(timestamp_collection_geometry(tsc), include_orphans=include_orphans)

class GeometryFileNotFound(FileNotFoundError):
    "Geometry file not found exception"

def store_timestamp_collection_geometry(tsc, to_folder, filename="timestamp_collection_geometry.json", verbose=True):
    import os, json
    to_path = os.path.join(to_folder, filename)
    geometry = tsc.geometry
    if geometry is None:
        if verbose:
            print ("calculating timestamp collection geometry -- this may take a while...")
        geometry = timestamp_collection_geometry(tsc)
    f = open(to_path, "w")
    json.dump(geometry, f)
    if verbose:
        print("saved time stamp geometry to", to_path)

def load_timestamp_collection_geometry(from_folder, filename="timestamp_collection_geometry.json"):
    import os, json
    to_path = os.path.join(from_folder, filename)
    if not os.path.isfile(to_path):
        raise GeometryFileNotFound("no such file: " + to_path)
    f = open(to_path)
    return json.load(f)

def load_or_create_geometry(folder, filename="timestamp_collection_geometry.json", verbose=True, include_orphans=False):
    from mouse_embryo_labeller import tools
    try:
        g = load_timestamp_collection_geometry(folder, filename=filename)
    except GeometryFileNotFound as e:
        if verbose:
            print ("Failed to load geometry from file: " + repr(e))
    else:
        if verbose:
            print ("loaded geometry from folder", repr(folder), "file", repr(filename))
        TI = TimeStampCollectionInterpolator(g, include_orphans=include_orphans)
        return TI
    if verbose:
        print("Fallback: Loading collections.")
    nc = tools.get_example_nucleus_collection(folder)
    tsc = tools.get_example_timestamp_collection(folder, nc)
    if verbose:
        print("Calculating geometry.  This may taka a while...")
    TI = interpolate_timestamp_collection(tsc, include_orphans=include_orphans)
    store_timestamp_collection_geometry(tsc, folder, verbose=verbose)
    return TI

class TimeStampCollectionInterpolator:

    def __init__(self, geometry, pixels=700, distortion=[3,1,1], include_orphans=False):
        self.pixels = pixels
        self.distortion = distortion
        self.geometry = geometry
        self.include_orphans = include_orphans
        self.make_timestamp_interpolators()

    def make_timestamp_interpolators(self):
        last_ts_geometry = None
        interpolators = []
        # get integer indexed timestamps
        ts_index = {}
        for (ts_id, ts_geometry) in self.geometry[TIMESTAMPS].items():
            ts_index[int(ts_id)] = ts_geometry
        for i in sorted(ts_index.keys()):
            ts_geometry = ts_index[i]
            if last_ts_geometry is not None:
                interpolator = TimeStampInterpolator(last_ts_geometry, ts_geometry, include_orphans=self.include_orphans)
                interpolators.append(interpolator)
            last_ts_geometry = ts_geometry
        assert len(interpolators) > 0, "nothing to interpolate?"
        self.interpolators = interpolators

    def make_assembly(self, pixels=None):
        # xxx cut/paste -- could unify
        import ipywidgets as widgets
        from jp_doodle.data_tables import widen_notebook
        widen_notebook()
        if pixels is not None:
            self.pixels = pixels
        self.info = widgets.HTML(value="Simple 3d view.")
        self.get_swatch()
        self.slider = widgets.FloatSlider(
            value=0,
            min=0.0,
            max=1.0 * len(self.interpolators),
            step=0.05,
            description='Interpolation:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=dict(width=str(self.pixels) + "px")
        )
        self.slider.observe(self.slide_interpolation, "value")
        self.assembly = widgets.VBox([
            self.info,
            self.canvas.debugging_display(),
            self.slider,
        ])
        self.slide_interpolation(None)
        return self.assembly

    def slide_interpolation(self, change):
        interpolators = self.interpolators
        nint = len(interpolators)
        value = self.slider.value
        index = min(nint - 1, int(value))
        lambda1 = value - index
        interpolator = interpolators[index]
        #print("slide", value)
        interpolation = interpolator.interpolate_pairings(lambda1)
        #pprint(interpolation)
        self.last_interpolation = interpolation
        with self.swatch.in_canvas.delay_redraw():
            self.swatch.reset()
            draw_box(self.swatch, self.dimensions, self.center, self.distortion)
            draw_ts_geometry(interpolation, self.swatch)
            self.swatch.orbit_all(center3d=self.center_d, radius=self.radius)
  
    def get_swatch(self):
        # xxx cut/paste -- could unify
        pixels = self.pixels
        #distortion = self.distortion
        from jp_doodle import nd_frame
        geometry = self.geometry
        self.distortion = np.array(self.distortion, dtype=np.float)
        self.center = np.array(geometry[CENTER])
        self.dimensions = np.array(geometry[DIMENSIONS])
        self.center_d = self.center * self.distortion
        #radius = geometry[RADIUS]
        self.radius = max(self.dimensions * self.distortion)
        swatch = nd_frame.swatch3d(pixels=pixels, model_height=self.radius * 2, auto_show=False)
        swatch.orbit(center3d=self.center_d, radius=self.radius, shift2d=(-0.25 * self.radius, -0.2 * self.radius))
        swatch.orbit_all(center3d=self.center_d, radius=self.radius)
        draw_box(swatch, self.dimensions, self.center, self.distortion)
        #draw_ts_geometry(geometry, swatch, distortion=distortion)
        swatch.fit(0.8)
        self.swatch = swatch
        self.canvas = swatch.in_canvas
        return swatch
