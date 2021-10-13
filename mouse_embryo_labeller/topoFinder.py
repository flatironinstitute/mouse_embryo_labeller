import numpy as np
import heapq
import time

def increase_mapping(key, value, mapping):
    vset = mapping.get(key, set())
    vset.add(value)
    mapping[key] = vset
    return mapping

def decrease_mapping(key, value, mapping):
    if key not in mapping:
        return mapping
    vset = mapping[key]
    if value in vset:
        vset.remove(value)
    if not vset:
        del mapping[key]
    return mapping

def check_mapping(key, mapping):
    mapping[key] = mapping.get(key, set())

class TopologyFinder:
    
    def __init__(self, triangles):
        self.reset_internals()
        self.edge_widths = {}
        self.vertex_size = {}
        for t in triangles:
            self.add_float_triangle(t)

    def reset_internals(self):
        self.vmin = np.zeros((3,))
        self.vrange = 1.0  # coordinate normalization scalar
        self.scale_factor = 1000.0
        #self.vertex_tuples = set()
        #self.segments = set()
        #self.triangle_sets = set()
        self.triangle_to_edges = {}
        self.edge_to_length = {}
        self.edge_to_triangles = {}
        self.vertex_to_edges = {}
        self.vertex_to_triangles = {}
        self.selected_edge = None
        self.length_edge_heap = None  # harden

    def prepare_length_edge_heap(self):
        h = [(ln, edge) for (edge, ln) in self.edge_to_length.items()]
        heapq.heapify(h)
        self.length_edge_heap = h
        return h

    def optimized_collapse_edge(self, redraw=True, sleep=1):
        t2e = self.triangle_to_edges
        ntriangles_before = len(t2e)
        h = self.length_edge_heap
        if h is None:
            h = self.prepare_length_edge_heap()
        chosen_edge = None
        e2t = self.edge_to_triangles
        while chosen_edge is None:
            (width, chosen_edge) = heapq.heappop(h)
            chosen_triangles = e2t.get(chosen_edge, set())
            if not chosen_triangles:
                chosen_edge = None
        vertices = list(chosen_edge)
        if redraw:
            self.draw_poly(points=vertices, color="red", fill=False, lineWidth=3)
            time.sleep(sleep)
        midpoint = tuple(np.array(vertices).mean(axis=0).astype(np.int))
        self.vertex_size[midpoint] = width
        new_vertex = frozenset([midpoint])
        (v1, v2) = vertices
        v2e = self.vertex_to_edges
        replace_edges = v2e.get(v1, set()) | v2e.get(v2, set())
        v2t = self.vertex_to_triangles
        replace_triangles = v2t.get(v1, set()) | v2t.get(v2, set())
        self.delete_vertex(v1)
        self.delete_vertex(v2)
        assert chosen_edge not in e2t
        for old_edge in replace_edges:
            new_edge = (old_edge - chosen_edge) | new_vertex
            if len(new_edge) == 2:
                self.add_edge_set(new_edge)
        for old_triangle in replace_triangles:
            new_triangle = (old_triangle - chosen_edge) | new_vertex
            ln = len(new_triangle)
            if ln < 2:
                pass # ignore degenerate???
            elif ln == 2:
                self.add_edge_set(new_triangle)
            else:
                assert ln == 3, "too big new triangle " + repr(new_triangle)
                self.add_triangle_set(new_triangle)
        if redraw:
            with self.canvas.delay_redraw():
                self.draw_frame(fit=False)
        ntriangles_after = len(t2e)
        assert ntriangles_after < ntriangles_before, "triangles not decreasing"

    def kiss_collapse_edge(self, redraw=True, sleep=1):
        "unoptimized collapse smallest edge."
        sorter = [(self.tedge_distance(e), e) for e in self.edge_to_triangles if self.edge_to_triangles[e]]
        sorter = sorted(sorter)
        (width, chosen_edge) = sorter[0]
        vertices = list(chosen_edge)
        if redraw:
            self.draw_poly(points=list(chosen_edge), color="red", fill=False, lineWidth=3)
            time.sleep(sleep)
        # collapse edge
        #(v1, v2) = vertices
        midpoint = tuple(np.array(vertices).mean(axis=0).astype(np.int))
        self.vertex_size[midpoint] = width
        new_vertex = frozenset([midpoint])
        #print("midpoint", midpoint)
        remove_triangles = self.edge_to_triangles[chosen_edge]
        keep_triangles = set()
        keep_edges = set()
        for edge in self.edge_to_triangles:
            if edge != chosen_edge:
                if edge & chosen_edge:
                    edge = (edge - chosen_edge) | new_vertex
                    #self.edge_widths[edge] = max(1, width/4.0)
                keep_edges.add(edge)
        for triangle in self.triangle_to_edges:
            #print("checking triangle", triangle)
            if triangle not in remove_triangles:
                if chosen_edge & triangle:
                    triangle = (triangle - chosen_edge) | new_vertex
                    #print ("intersects", triangle)
                    assert len(triangle) == 3
                else:
                    #print ("no intersection")
                    pass
                keep_triangles.add(triangle)
            else:
                #print("deleting: includes edge")
                pass
        self.reset_internals()
        for edge in keep_edges:
            self.add_edge_set(edge)
        for triangle in keep_triangles:
            self.add_triangle_set(triangle)
        if redraw:
            with self.canvas.delay_redraw():
                self.draw_frame(fit=False)

    def add_vertex_tuple(self, vertex_tuple):
        check_mapping(vertex_tuple, self.vertex_to_edges)
        check_mapping(vertex_tuple, self.vertex_to_triangles)

    def add_edge_set(self, edge_set):
        ln = len(edge_set)
        if ln < 2:
            return  # ignore degenerate
        assert ln == 2, "too large edge set " + repr(edge_set)
        v2e = self.vertex_to_edges
        for v in edge_set:
            self.add_vertex_tuple(v)
            increase_mapping(v, edge_set, v2e)
        if edge_set not in self.edge_to_triangles:
            self.edge_to_triangles[edge_set] = set()
        self.edge_to_length[edge_set] = self.tdistance(*list(edge_set))

    def add_triangle_set(self, triangle_set):
        ln = len(triangle_set)
        if ln == 2:
            return self.add_edge_set(triangle_set)
        if ln < 3:
            # ignore degenerate one point triangle
            return
        assert ln == 3, "too large triangle " + repr(triangle_set)
        v2t = self.vertex_to_triangles
        for v in triangle_set:
            self.add_vertex_tuple(v)
            increase_mapping(v, triangle_set, v2t)
        (A, B, C) = tuple(triangle_set)
        edges = frozenset(frozenset(pair) for pair in  ((A, B), (B, C), (C, A)))
        assert (len(edges) == 3)
        e2t = self.edge_to_triangles
        self.triangle_to_edges[triangle_set] = edges
        h = self.length_edge_heap
        e2l = self.edge_to_length
        for edge in edges:
            self.add_edge_set(edge)
            increase_mapping(edge, triangle_set, e2t)
            if h is not None:
                ln = e2l[edge]
                heapq.heappush(h, (ln, edge))

    def delete_vertex(self, vertex):
        v2e = self.vertex_to_edges
        delete_edges = v2e.get(vertex)
        if delete_edges is not None:
            del v2e[vertex]
            for edge in delete_edges:
                self.delete_edge(edge)

    def delete_edge(self, edge):
        v2e = self.vertex_to_edges
        for vertex in edge:
            if vertex in v2e:
                decrease_mapping(vertex, edge, v2e)
        e2t = self.edge_to_triangles
        delete_triangles = e2t.get(edge)
        if delete_triangles is not None:
            del e2t[edge]
            for triangle in delete_triangles:
                self.delete_triangle(triangle)

    def delete_triangle(self, triangle):
        t2e = self.triangle_to_edges
        edges = t2e.get(triangle)
        del t2e[triangle]
        v2t = self.vertex_to_triangles
        for vertex in triangle:
            decrease_mapping(vertex, triangle, v2t)
        e2t = self.edge_to_triangles
        for edge in edges:
            decrease_mapping(edge, triangle, e2t)

    def doodle(self, pixel_width=500):
        from jp_doodle import dual_canvas
        self.canvas = dual_canvas.DualCanvasWidget(width=pixel_width, height=pixel_width)
        varray = np.array(list(self.vertex_to_edges.keys()))
        vmax = varray.max(axis=0)
        vmin = varray.min(axis=0)
        vdiff = vmax - vmin
        vside = vdiff.max()
        self.frame = self.canvas.frame_region(
            minx=0, miny=0, maxx=pixel_width, maxy=pixel_width, 
            frame_minx=0, frame_miny=0, frame_maxx=vside, frame_maxy=vside)
        self.draw_frame(fit=True)
        return self.canvas

    def draw_poly(self, points, color, fill, lineWidth=1):
        shift = 0.2
        points = np.array(points)[:,:2]
        center = points.mean(axis=0).reshape((1, 2))
        shifted = (1.0 - shift) * points + shift * center
        self.frame.polygon(points=shifted, color=color, fill=fill, lineWidth=lineWidth)

    def draw_frame(self, fit=False):
        radius = 5
        circle_color = "cyan"
        edge_color = "blue"
        triangle_color = "rgba(255,0,255,0.3)"
        frame = self.frame
        frame.reset_frame()
        for v in self.vertex_to_edges:
            (x, y, z) = v
            r = max(0.5 * self.vertex_size.get(v, 0), radius)
            frame.frame_circle(x=x, y=y, r=r, color=circle_color)
        for edge in self.edge_to_triangles:
            w = self.edge_widths.get(edge, 1)
            self.draw_poly(list(edge), color=edge_color, fill=False, lineWidth=w)
        for triangle in self.triangle_to_edges:
            self.draw_poly(list(triangle), color=triangle_color, fill=True)
        if fit:
            self.canvas.fit()

    def add_float_triangle(self, triangle):
        return self.add_triangle_set(self.triangle_set(triangle))

    def triangle_set(self, float_triangle):
        return frozenset(self.vertex_tuple(v) for v in float_triangle)

    def vertex_tuple(self, floats):
        offset = np.array(floats, dtype=np.float) - self.vmin
        scaled = (self.scale_factor * offset) / self.vrange
        gridded = scaled.astype(np.int)
        return tuple(gridded)

    def tedge_distance(self, tedge):
        return self.tdistance(*list(tedge))

    def tdistance(self, tupleA, tupleB):
        return np.linalg.norm(np.array(tupleA) - np.array(tupleB))

    def add_box(self, i, j):
        self.add_float_triangle([(i, j, 0), (i+1, j, 0), (i, j+1, 0), ])
        self.add_float_triangle([(i+1, j+1, 0), (i+1, j, 0), (i, j+1, 0), ])

def jupyter_test():
    T = TopologyFinder([])
    for i in range(-10, 10):
        T.add_box(i, 5)
        T.add_box(5, i)
        T.add_box(5-i, i-6)
        T.add_box(5-i, i-5)
    return T

