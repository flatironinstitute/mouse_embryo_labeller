
"""
Tools for manipulating Fiji ROI files.
"Region of interest."

Based on:
http://wsr.imagej.net/macros/js/DecodeRoiFile.js
https://imagej.nih.gov/ij/developer/api/ij/ij/io/RoiDecoder.html
https://imagej.nih.gov/ij/developer/api/ij/ij/io/RoiDecoder.html
"""

import numpy as np

# File offset constants:
VERSION_OFFSET = 4
TYPE = 6
TOP = 8
LEFT = 10
BOTTOM = 12
RIGHT = 14
N_COORDINATES = 16
X1=18
Y1=22
X2=26
Y2=30
COORDINATES = 64
stroke_color_start = 40

# Type indicator constants
polygon=0
rect=1
oval=2
line=3
freeline=4
polyline=5
noRoi=6
freehand=7
traced=8
angle=9
point=10

poly_types = set([freehand, traced, polyline, polygon])
open_types = set([freehand, traced, polyline, polygon])
closed_types = set([freehand, traced, polyline, polygon])

big_endian_signed_short = np.dtype(">i2")

class ROIdata:

    def __init__(self):
        self.reset()

    def reset(self):
        self.version = None
        self.type = None
        self.stroke_color = None
        self.xbase = 0 # default
        self.ybase = 0 # default
        self.points = []

    def dump_to_bytes(self):
        print ("dumping")
        assert self.type in poly_types, "Dump for this type not yet supported: " + repr(type)
        points = self.points
        header = np.zeros((COORDINATES,), dtype=np.ubyte)
        header[:4] = [int(x) for x in  b"Iout"]
        #header[VERSION_OFFSET] = self.version
        self.int_to_index(header, VERSION_OFFSET, self.version)
        header[TYPE] = self.type
        self.int_to_index(header, LEFT, self.xbase)
        self.int_to_index(header, TOP, self.ybase)
        self.int_to_index(header, N_COORDINATES, len(points))
        header[stroke_color_start: stroke_color_start+4] = self.stroke_color
        print("header stroke color is", header[stroke_color_start: stroke_color_start+4])
        L = [bytes(header)]
        for (x, y) in points:
            L.append(self.int_to_byte_pair(x - self.xbase))
        for (x, y) in points:
            L.append(self.int_to_byte_pair(y - self.ybase))
        return b''.join(L)

    def dump_to_path(self, path):
        f = open(path, "wb")
        f.write(self.dump_to_bytes())
        f.close()

    def load_from_bytes(self, byte_seq):
        assert bytes(byte_seq[:4]) == b"Iout"
        self.reset()
        self.type = int(byte_seq[TYPE])
        #self.version = int(byte_seq[VERSION_OFFSET])
        self.version = self.int_from_index(byte_seq, VERSION_OFFSET)
        self.xbase = self.int_from_index(byte_seq, LEFT)
        self.ybase = self.int_from_index(byte_seq, TOP)
        print("version, type, xbase, ybase", self.version, self.type, self.xbase, self.ybase)
        self.stroke_color = [int(x) for x in byte_seq[stroke_color_start: stroke_color_start+4]]
        print("stroke color", self.stroke_color)
        if self.type in poly_types:
            n_coords = self.int_from_index(byte_seq, N_COORDINATES)
            print("n_coords", n_coords)
            y_start = COORDINATES + 2 * n_coords
            points = self.points
            for i in range(n_coords):
                shift = i * 2
                x_offset = COORDINATES + shift
                y_offset = y_start + shift
                x = self.int_from_index(byte_seq, x_offset) + self.xbase
                y = self.int_from_index(byte_seq, y_offset) + self.ybase
                points.append((x, y))
            #print("points", points)
        else:
            print ("Parsing non-polygon types is not yet supported:", self.type)

    def load_from_path(self, path):
        byte_seq = open(path, "rb").read()
        return self.load_from_bytes(byte_seq)

    def widget(self, array, color="red"):
        from jp_doodle import dual_canvas
        (iheight, iwidth) = array.shape[:2]
        c = dual_canvas.DualCanvasWidget(width=iwidth, height=iheight)
        f = c.frame_region(
            minx=0, miny=0, maxx=iwidth, maxy=iheight,
            frame_minx=0, frame_miny=iheight, frame_maxx=iwidth, frame_maxy=0
        )
        name = "array image"
        c.name_image_array(name, array)
        f.named_image(name, 0, iheight, iwidth, iheight, name=True)
        f.polygon(self.points, color=color, close=False, fill=False)
        c.fit()
        return c

    def int_to_byte_pair(self, integer):
        a = np.array(integer, dtype=big_endian_signed_short)
        return a.tobytes()

    def int_to_byte_list(self, integer):
        pair = self.int_to_byte_pair(integer)
        return list(pair)

    def int_to_index(self, byte_list, index, integer):
        byte_list[index: index+2] = self.int_to_byte_list(integer)

    def int_from_index(self, byte_list, index):
        return self.int_from_byte_list(byte_list[index: index+2])

    def int_from_byte_list(self, byte_list):
        assert len(byte_list) == 2, "must have 2 bytes " + repr(byte_list)
        s = bytes(byte_list)
        return np.frombuffer(s, dtype=big_endian_signed_short)[0]

class RegionTracer:

    def __init__(self, array, lower, upper=None):
        if upper is None:
            upper = lower
        from .timestamp import boundary
        self.array = array
        self.lower = lower
        self.upper = upper
        self.region = np.logical_and( (array >= lower), (array <= upper))
        self.boundary = boundary(self.region)

    def boundary_tuples(self):
        b = self.boundary
        (i_indices, j_indices) = np.nonzero(b)
        return list(zip(i_indices, j_indices))

    offset_order = [(0,1), (1,1), (1,0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1,1), (0,1)]

    def boundary_parent_map(self):
        btups = sorted(self.boundary_tuples())
        bset = set(btups)
        chosen = set()
        result = {}
        r = self.region
        (maxi, maxj) = r.shape
        for t in btups:
            (ii, jj) = t
            choice = None
            crossed_in = crossed_out = False
            for (di, dj) in self.offset_order:
                i = ii + di
                j = jj + dj
                if (0 <= i < maxi) and (0 <= j < maxj):
                    if r[i,j]:
                        crossed_in = True
                    else:
                        crossed_out = True
                if (choice is not None) and crossed_in and crossed_out:
                    break
                t2 = (i, j)
                if t2 != t and t2 in bset and t2 not in chosen:
                    choice = t2
            # don't add trivial looping pairs
            if (choice is not None) and result.get(choice) != t:
                result[t] = choice
                chosen.add(choice)
        return result

    def widget(self, size=800):
        from jp_doodle import dual_canvas
        c = dual_canvas.DualCanvasWidget(width=size, height=size)
        f = c.rframe(10, 10)
        r = self.region
        b = self.boundary
        (rows, cols) = r.shape
        for i in range(rows):
            for j in range(cols):
                color = "yellow"
                fill = True
                if b[i,j]:
                    color = "green"
                    fill = True
                elif r[i,j]:
                    color = "cyan"
                    fill = True
                f.circle(x=i, y=j, r=3, color=color, fill=fill)
        bmap = self.boundary_parent_map()
        for ((i,j), (i2,j2)) in bmap.items():
            f.arrow(i, j, i2, j2, 0.5, lineWidth=2, color="magenta")
        for path in self.combined_paths():
            f.polyline(path, color="orange")
        c.fit()
        return c

    def combined_paths(self):
        parent_child = self.boundary_parent_map()
        child_parent = {p: c for (c, p) in parent_child.items()}
        parents = set(parent_child.keys())
        children = set(child_parent.keys())
        roots = parents - children
        leaves = children - parents
        #print ("roots", list(roots))
        #print ("leaves", list(leaves))
        root_paths = {root: self.path(root, parent_child) for root in roots}
        leaf_paths = {leaf: self.path(leaf, child_parent) for leaf in leaves}
        result = []
        visited = set()
        while root_paths:
            print ("roots now", list(root_paths.keys()))
            root = None
            path = None
            for r in root_paths:
                p = root_paths[r]
                if path is None or len(path) < len(p):
                    root = r
                    path = p
            del root_paths[root]
            for v in path:
                visited.add(v)
            path_done = False
            while not path_done:
                back_path = None
                # trace backwards from path end
                last = path[-1]
                if last in leaf_paths:
                    del leaf_paths[last]
                for leaf in leaf_paths:
                    if self.adjacent(last, leaf):
                        test_path = leaf_paths[leaf]
                        if (back_path is None) or (len(test_path) > len(back_path)):
                            back_path = test_path
                if back_path is None:
                    path_done = True
                else:
                    path = path + back_path
                    for v in back_path:
                        visited.add(v)
                    del leaf_paths[back_path[0]]
                    last = path[-1]
                    if last in root_paths:
                        del root_paths[last]
                    forward_path = None
                    for r in root_paths:
                        if self.adjacent(last, r):
                            test_path = root_paths[r]
                            if (forward_path is None) or (len(test_path) < len(forward_path)):
                                forward_path = test_path
                    if forward_path is None:
                        path_done = True
                    else:
                        for v in forward_path:
                            visited.add(v)
                        path = path + forward_path
            result.append(path)
            #for v in path:
            #    visited.add(v)
        # any unvisited parent should lie on a loop -- add the loops
        looping = parents - visited
        while looping:
            loop_start = looping.pop()
            print("loop start", loop_start)
            loop = self.path(loop_start, parent_child)
            for v in loop:
                if v in looping:
                    looping.remove(v)
                result.append(loop)
        return result

    def path(self, from_source, in_mapping):
        result = [from_source]
        visited = set([from_source])
        current = from_source
        while current in in_mapping:
            current = in_mapping[current]
            if current in visited:
                # path loops
                break
            visited.add(current)
            result.append(current)
        return result

    def adjacent(self, i2a, i2b):
        a = np.array(i2a, dtype=np.int)
        b = np.array(i2b, dtype=np.int)
        absd = np.abs(a - b)
        return absd.max() == 1

SAMPLE = (
    b'Iout\x00\xe4\x02\x00\x02H\x03\xca\x02t\x03\xf8\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x13\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x03\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00x\x00y\x00z')

def test():
    R = ROIdata()
    for L in ([0,1], [1,0]):
        i = R.int_from_byte_list(L)
        b = R.int_to_byte_list(i)
        print ("bytes", L, i, b)
        assert b == L, "bad byte conversion: " + repr((L, i, b))
    SAMPLE = open("/Users/awatters/misc/Abraham_Kohrman/roi/banana/0344-0476.roi", "rb").read()
    R.load_from_bytes(SAMPLE)

if __name__ == "__main__":
    test()