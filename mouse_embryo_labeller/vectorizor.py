"""
Compute vector offsets for two 3D label volumes
"""

import numpy as np

from mouse_embryo_labeller.color_list import indexed_color
from mouse_embryo_labeller.timestamp import big_choose

def make_tracks_from_haydens_json_graph(json_graph):
    """
    Read a JSON dump of matlab graph similar to "Gata6Nanog1.json".
    Return a dictionary mapping timestamp numbers to dictionaries mapping timestemps to tracks.

    >>> timestamp_to_mappings = make_tracks_from_haydens_json_graph(json_graph)
    >>> timestamp3_label_to_track = timestamp_to_mappings[3]
    >>> track_for_label_6 = timestamp3_label_to_track[6]
    """
    edges = json_graph['G_based_on_nn_combined']['Edges']
    mapping = {}
    splits = {}
    all_ids = set()
    for thing in edges:
        [src, dst] = thing["EndNodes"]
        all_ids.add(src)
        all_ids.add(dst)
        if src in mapping:
            #print ("split", src, mapping[src], dst)
            splits[src] = mapping[src]
        mapping[src] = dst
    track_starts = sorted(set(mapping.keys()) - set(mapping.values()))
    string_to_track = { s: count+1 for (count, s) in enumerate(track_starts) }
    for start in track_starts:
        track = string_to_track[start]
        next = mapping.get(start)
        while next is not None:
            string_to_track[next] = track
            next = mapping.get(next)
    parsed_strings = {}
    timestamps = set()
    for s in all_ids:
        [ts_string, label_string] = s.split("_")
        parsed = (int(ts_string), int(label_string))
        timestamps.add(parsed[0])
        parsed_strings[s] = parsed
    ts_to_label_to_track = {ts: {} for ts in timestamps}
    for (st, track) in string_to_track.items():
        (ts, label) = parsed_strings[st]
        ts_mapping = ts_to_label_to_track [ts]
        ts_mapping[label] = track
    ts_to_label_to_split_label = {ts: {} for ts in timestamps}
    for (sparent, schild) in splits.items():
        (parent_ts, parent_label) = parsed_strings[sparent]
        (child_ts, child_label) = parsed_strings[schild]
        if child_ts != parent_ts + 1:
            print("WARNING: IGNORED bad timestamp order: ", repr((sparent, schild)))
        else:
            ts_to_label_to_split_label[parent_ts][parent_label] = child_label
    return (ts_to_label_to_track, ts_to_label_to_split_label)


def offset_vectors(vector_maker):
    """
    Compute vectors for vector_maker using the following method:
    For corresponding label volumes compute the vector difference between the "new" center and the "old" center.
    Every voxel position in the "old" volume is assigned the difference vector in the vector field output.  
    All other vectors not in an old volume are zeros.
    """
    self = vector_maker
    A = self.A
    Acenters = self.Acenters
    Bcenters = self.Bcenters
    vectors = np.zeros(A.shape + (3,), dtype=np.float)
    for (v, ca) in Acenters.items():
        cb = Bcenters.get(v)
        if cb is not None:
            d = cb - ca
            (Is, Js, Ks) = np.nonzero( (A == v).astype(np.int) )
            vectors[Is, Js, Ks] = d
    return vectors

def center_vectors(vector_maker):
    """
    Compute vectors for vector_maker using the following method:
    For corresponding label volumes compute the vector difference between the "new" center and the "old" center.
    Every voxel position in the "old" volume is assigned a vector pointing to the center of the new volume.  
    All other vectors not in an old volume are zeros.
    """
    self = vector_maker
    A = self.A
    Acenters = self.Acenters
    Bcenters = self.Bcenters
    vectors = np.zeros(A.shape + (3,), dtype=np.float)
    for (v, ca) in Acenters.items():
        cb = Bcenters.get(v)
        if cb is not None:
            (Is, Js, Ks) = np.nonzero( (A == v).astype(np.int) )
            sources = np.zeros(Is.shape + (3,), dtype=np.float)
            sources[:, 0] = Is
            sources[:, 1] = Js
            sources[:, 2] = Ks
            d = cb.reshape((1,3)) - sources
            vectors[Is, Js, Ks] = d
    return vectors


def blend_vectors(vector_maker):
    """
    Compute vectors for vector_maker using the following method:
    For corresponding label volumes compute the vector difference between the "new" center and the "old" center.
    Every voxel position in the "old" volume is assigned a vector to a blend of the centers in the new volume
    based on how near the position is to centers in the old volume.
    """
    self = vector_maker
    A = self.A
    (II, JJ, KK) = A.shape
    (Is, Js, Ks) = np.meshgrid(np.arange(II), np.arange(JJ), np.arange(KK), indexing='ij')
    def Distance(i, j, k):
        return np.sqrt((i - Is) ** 2 + (j - Js) ** 2 + (k - Ks) ** 2)
    Acenters = self.Acenters
    Bcenters = self.Bcenters
    vectors = np.zeros(A.shape + (3,), dtype=np.float)
    tracks = set(Acenters.keys()) & set(Bcenters.keys())
    offsets = {t: (Bcenters[t] - Acenters[t]) for t in tracks}
    Adistances = {t: Distance(*Acenters[t]) for t in tracks}
    weighting_factors = {t: 1.0 for t in tracks}
    for t in tracks:
        for (t2, D) in Adistances.items():
            if t2 != t:
                weighting_factors[t] = weighting_factors[t] * D
    w2 = 0
    for w in weighting_factors.values():
        w2 += w**2
    normalizer = np.sqrt(w2)
    for t in tracks:
        #print()
        #print ("for track", t)
        weight = weighting_factors[t] / normalizer
        #print (weight, "weights")
        offset = offsets[t]
        #print (offset, "offsets")
        component = weight.reshape(weight.shape + (1,)) * offset.reshape((1,1,1,3))
        #print (component, "component")
        vectors += component
    #print("vectors")
    #print(vectors)
    #sanity check
    lvecs = vectors.reshape((II * JJ * KK, 3))
    M = (np.abs(lvecs)).max(axis=0)
    assert np.all( M < vv(II, JJ, KK)), "bad max: " + repr((M, II, JJ, KK))
    return vectors

VECTORIZE_METHODS = dict(
    offset = offset_vectors,
    center = center_vectors,
    blend = blend_vectors,
)
DEFAULT_VECTORIZE_METHOD = "center"

def unify_tracks(
    A,  # label array
    A_label_2_track,   # mapping of labels in A to track numbers
    B,  # label array
    B_label_2_track,   # mapping of labels in A to track numbers
    splits,
):
    """
    Remap contents of A and B replacing arbitrary labels with corresponding tracks.
    Any labels with no track number assigned will be remapped to a "fresh" number.
    
    returns (A_remapped, B_remapped)
    """
    Alabels = set(np.unique(A))
    Blabels = set(np.unique(B))
    A2t = A_label_2_track.copy()  # copy for modification
    B2t = B_label_2_track.copy()  # copy for modification
    tracks = set(A2t.values()) | set(B2t.values())
    A_remapped = A.copy()
    B_remapped = B.copy()
    max_track = max(*tracks)
    for (Ar, l2t, labels) in [(A_remapped, A2t, Alabels), (B_remapped, B2t, Blabels)]:
        for label in labels:
            if label not in l2t and label>0:
                fresh_track = max_track = max_track + 1
                tracks.add(fresh_track)
                l2t[label] = fresh_track
        max_label = 0
        if len(labels) == 1:
            max_label = list(labels)[0]
        else:
            max_label = max(*labels)
        choices = np.zeros((max_label+1,), dtype=np.int)
        #print(max_label, choices)
        #print(l2t)
        for label in l2t:
            if label in labels:
                choices[label] = l2t[label]
        assert choices[0] == 0, "bad choice for 0 " + repr((choices, l2t))
        Ar[:] = big_choose(Ar, choices)
    # split logic
    for (src, dst) in splits.items():
        dst_track = B2t[dst]
        current_track = A2t[src]
        src_center = center_of_values(A_remapped, current_track)
        dst_center = center_of_values(B_remapped, dst_track)
        current_center = center_of_values(B_remapped, current_track)
        # punt if any region is empty
        if src_center is not None and dst_center is not None and current_center is not None:
            vdst = normalize(dst_center - src_center)
            vcurrent = normalize(current_center - src_center)
            out = np.cross(vdst, vcurrent)
            middle = (vdst + vcurrent)
            split_vector = np.cross(out, middle)
            if split_vector.dot(vdst) < 0:
                split_vector = - split_vector
            mask = (A_remapped == current_track).astype(np.int)
            splitter = Splitter(mask, src_center, split_vector)
            (II, JJ, KK) = splitter.positiveIJKs()
            A_remapped[II, JJ, KK] = dst_track
        else:
            print("WARNING: null centers in split " + repr((src_center, dst_center, current_center)))
    return (A_remapped, B_remapped)

def normalize(v, epsilon=1e-10):
    n = np.linalg.norm(v)
    if n < epsilon:
        return np.array([1,0,0], dtype=np.float)   # arbitrary default unit
    return v / n

def center_of_values(A, value):
    """
    Find the weighted average of indices that have the value in A (or None if missing)
    """
    (Is, Js, Ks) = np.nonzero( (A == value).astype(np.int) )
    N = len(Js)
    if N < 1:
        print("no such value" + repr((value, np.unique(A))))
        return None
    Ind = np.zeros((N, 3), dtype=np.int)
    Ind[:, 0] = Is
    Ind[:, 1] = Js
    Ind[:, 2] = Ks
    center = Ind.mean(axis=0)
    return center


class Splitter:

    def __init__(self, mask, center, normal):
        """
        For center and normal in index dimensions identify the non-zero indices
        in mask that are above and below the plane including the center orthogonal
        to normal
        """
        center = np.array(center)
        normal = np.array(normal)
        normal = normal/np.linalg.norm(normal)
        self.mask = mask
        self.center = center
        self.normal = normal
        [Is, Js, Ks] = np.nonzero(mask)
        ln = len(Is)
        index_vectors = np.zeros((ln, 3), dtype=np.int)
        index_vectors[:, 0] = Is
        index_vectors[:, 1] = Js
        index_vectors[:, 2] = Ks
        shifted_vectors = index_vectors - center.reshape((1,3))
        dots = shifted_vectors.dot(normal.reshape((3,1)))
        self.radius = np.abs(dots).max()
        positive_dots = (dots >= 0).reshape((ln,))
        print ("positive", positive_dots.shape)
        print ("index vectors", index_vectors.shape)
        self.positive_indices = index_vectors[positive_dots]
        self.negative_indices = index_vectors[np.logical_not(positive_dots)]
        print ("negative", self.negative_indices.shape)

    def positiveIJKs(self, indices=None):
        if indices is None:
            indices = self.positive_indices
        return (indices[:,0], indices[:,1], indices[:,2], )

    def negativeIJK(self):
        return self.positiveIJKs(self.negative_indices)

    def widget(self, pixels=600):
        from jp_doodle import nd_frame
        W = nd_frame.swatch3d(pixels=pixels, model_height=2*self.radius)
        self.W = W
        W.arrow(self.center, self.center + (self.radius * self.normal), lineWidth=5, head_length=0.3)
        W.circle(self.center, 10, color="blue")
        for ind in self.positive_indices:
            W.circle(ind, 3, color="green")
        for ind in self.negative_indices:
            W.circle(ind, 3, color="red")
        W.fit(0.6)
        W.orbit_all(1.5 * self.radius, list(self.center))
        return W

def get_track_vector_field(
    old_label_array, 
    old_labels_to_tracks,
    new_label_array,
    new_labels_to_tracks,
    di = (10, 0, 0),  #  xyz offset between A[i,j,k] and A[i+1,j,k]
    dj = (0, 10, 0),  #  xyz offset between A[i,j,k] and A[i,j+1,k]
    dk = (0, 0, 10),  #  xyz offset between A[i,j,k] and A[i,j,k+1]
    method=DEFAULT_VECTORIZE_METHOD,
    ):
    (old_mapped, new_mapped) = unify_tracks(old_label_array, old_labels_to_tracks, new_label_array, new_labels_to_tracks)
    V = VectorMaker(old_mapped, new_mapped, di, dj, dk, method=method)
    return V.scaled_vectors

'''def get_label_vector_field(
    old_label_array, 
    new_label_array,
    di = (10, 0, 0),  #  xyz offset between A[i,j,k] and A[i+1,j,k]
    dj = (0, 10, 0),  #  xyz offset between A[i,j,k] and A[i,j+1,k]
    dk = (0, 0, 10),  #  xyz offset between A[i,j,k] and A[i,j,k+1]
    method=DEFAULT_VECTORIZE_METHOD,
    ):
    V = VectorMaker(old_label_array, new_label_array, di, dj, dk, method=method)
    return V.scaled_vectors  COMMENTED - NOT USED'''

def vv(*args):
    return np.array(args, dtype=np.float)

def ii(*args):
    return np.array(args, dtype=np.int)

class VectorMaker:

    def __init__(
        self,
        A,  # the "old" label volume
        B,  # the "new" label volume
        di = (10, 0, 0),  #  xyz offset between A[i,j,k] and A[i+1,j,k]
        dj = (0, 10, 0),  #  xyz offset between A[i,j,k] and A[i,j+1,k]
        dk = (0, 0, 10),  #  xyz offset between A[i,j,k] and A[i,j,k+1]
        method=DEFAULT_VECTORIZE_METHOD,
    ):
        assert A.shape == B.shape, "Arrays must match: " +repr((A.shape, B.shape))
        assert method in VECTORIZE_METHODS, (
            "no such vectorization method: " + repr(method) +
            ".  Available methods: " + repr(list(VECTORIZE_METHODS.keys()))
        )
        self.vectorize_method = VECTORIZE_METHODS[method]
        self.A = A
        self.B = B
        self.di = vv(*di)
        self.dj = vv(*dj)
        self.dk = vv(*dk)
        self.Acenters = {}
        self.Bcenters = {}
        self.max = self.min = None
        self.find_centers(A, self.Acenters)
        self.find_centers(B, self.Bcenters)
        self.center = self.pos(0.5 * (self.min + self.max))
        radius = (self.max - self.min).max()
        self.radius = self.pos([radius, radius, radius]).max()
        self.compute_vectors()

    def pos(self, ijk):
        (i, j, k) = ijk
        return i * self.di + j * self.dj + k * self.dk

    def scale_indices(self, index_rows):
        shape = index_rows.shape
        assert shape[-1] == 3, "index vectors must be triples: " + repr(shape)
        ln = len(index_rows.ravel())
        N = int(ln / 3)
        index_rows = index_rows.reshape((N, 3))
        ishape = (N, 1)
        Is = index_rows[:, 0].reshape(ishape)
        Js = index_rows[:, 1].reshape(ishape)
        Ks = index_rows[:, 2].reshape(ishape)
        di = self.di.reshape((1,3))
        dj = self.dj.reshape((1,3))
        dk = self.dk.reshape((1,3))
        scaled = Is * di + Js * dj + Ks * dk
        return scaled.reshape(shape)

    def find_centers(self, A, centers):
        u = np.unique(A)
        for v in u:
            if v > 0:
                (Is, Js, Ks) = np.nonzero( (A == v).astype(np.int) )
                N = len(Js)
                Ind = np.zeros((N, 3), dtype=np.int)
                Ind[:, 0] = Is
                Ind[:, 1] = Js
                Ind[:, 2] = Ks
                center = Ind.mean(axis=0)
                centers[v] = center
                m = Ind.min(axis=0)
                M = Ind.min(axis=0)
                if self.max is None:
                    self.max = M
                    self.min = m 
                else:
                    self.max = np.maximum(self.max, M)
                    self.min = np.minimum(self.min, m)

    def compute_vectors(self):
        """Compute vectors using the chosen method."""
        vectorize_method = self.vectorize_method
        self.vectors = vectorize_method(self)
        self.scaled_vectors = self.scale_indices(self.vectors)

    def draw(self, W, limit=500):
        from . import color_list
        #W = self.W
        A = self.A
        Alabels = np.unique(A)
        Blabels = np.unique(self.B)
        labels = list(set(Alabels) | set(Blabels))
        colors = color_list.get_colors(len(labels))
        self.label_to_color = {label: color_list.rgbhtml(color) for (label, color) in zip(labels, colors)}
        self.draw_array(A, radius=5)
        self.draw_array(self.B, radius=5)
        self.draw_centers(self.Acenters, radius=5)
        self.draw_centers(self.Bcenters, radius=7)
        (I, J, K) = A.shape
        vectors = self.vectors
        info = []
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    #v = A[i, j, k]
                    #if v > 0:
                    p = vv(i, j, k)
                    v = vectors[i, j, k]
                    if (np.abs(v)).max() > 0:
                        #dp = p + v
                        loc1 = self.pos(p)
                        #loc2 = self.pos(dp)
                        loc2 = loc1 + self.scaled_vectors[i, j, k]
                        #W.arrow(loc1, loc2, 2)
                        info.append([loc1, loc2])
        ln = len(info)
        if ln < limit:
            indices = range(ln)
        else:
            indices = np.random.choice(ln, limit, replace=True)
        for i in indices:
            [loc1, loc2] = info[i]
            W.arrow(loc1, loc2, 2)
        # draw big arrows between corresponding centers
        Acenters = self.Acenters
        Bcenters = self.Bcenters
        l2c = self.label_to_color
        for (v, cA) in Acenters.items():
            cB = Bcenters.get(v)
            if cB is not None:
                locA = self.pos(cA)
                locB = self.pos(cB)
                color = l2c[v]
                W.arrow(locA, locB, 5, lineWidth=8, color=color)

    def draw_centers(self, centers, radius):
        W = self.W
        l2c = self.label_to_color
        for (v, c) in centers.items():
            pos = self.pos(c)
            c = l2c[v]
            W.rect(pos, radius, radius, c)

    def draw_array(self, A, radius, limit=100):
        W = self.W
        (I, J, K) = A.shape
        l2c = self.label_to_color
        info = []
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    v = A[i,j,k]
                    if v > 0:
                        c = l2c[v]
                        pos = self.pos([i, j, k])
                        #W.circle(pos, radius, c, fill=False)
                        info.append((pos, c))
        ln = len(info)
        if ln < limit:
            indices = range(ln)
        else:
            indices = np.random.choice(ln, limit, replace=True)
        for i in indices:
            (pos, c) = info[i]
            W.circle(pos, radius, c, fill=False)

    def widget(self, pixels=600):
        from jp_doodle import nd_frame
        A = self.A
        B = self.B
        #Acount = len(np.nonzero(A)[0])
        #Bcount = len(np.nonzero(B)[0])
        #count = Acount + Bcount
        #assert count < 2000, "too many labels for this method of display: " + repr(count)
        W = nd_frame.swatch3d(pixels=pixels, model_height=pixels)
        self.W = W
        self.draw(W)
        W.fit(0.6)
        W.orbit_all(1.5 * self.radius, list(self.center))
        return W
