"""
Compute vector offsets for two 3D label volumes
"""

import numpy as np

from mouse_embryo_labeller.color_list import indexed_color
from mouse_embryo_labeller.timestamp import big_choose

def unify_tracks(
    A,  # label array
    A_label_2_track,   # mapping of labels in A to track numbers
    B,  # label array
    B_label_2_track,   # mapping of labels in A to track numbers
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
            if label not in l2t:
                fresh_track = max_track = max_track + 1
                tracks.add(fresh_track)
                l2t[label] = fresh_track
        max_label = max(*labels)
        choices = np.zeros((max_label+1,), dtype=np.int)
        print(max_label, choices)
        print(l2t)
        for label in l2t:
            if label in labels:
                choices[label] = l2t[label]
        Ar[:] = big_choose(Ar, choices)
    return (A_remapped, B_remapped)

def get_label_vector_field(
    old_label_array, 
    new_label_array,
    di = (10, 0, 0),  #  xyz offset between A[i,j,k] and A[i+1,j,k]
    dj = (0, 10, 0),  #  xyz offset between A[i,j,k] and A[i,j+1,k]
    dk = (0, 0, 10),  #  xyz offset between A[i,j,k] and A[i,j,k+1]
    ):
    V = VectorMaker(old_label_array, new_label_array, di, dj, dk)
    return V.scaled_vectors

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
    ):
        assert A.shape == B.shape, "Arrays must match: " +repr((A.shape, B.shape))
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
        self.vectors = vectors
        self.scaled_vectors = self.scale_indices(vectors)

    def draw(self, W):
        from . import color_list
        #W = self.W
        A = self.A
        Alabels = np.unique(A)
        Blabels = np.unique(self.B)
        labels = list(set(Alabels) | set(Blabels))
        colors = color_list.get_colors(len(labels))
        self.label_to_color = {label: color_list.rgbhtml(color) for (label, color) in zip(labels, colors)}
        self.draw_array(A, radius=3)
        self.draw_array(self.B, radius=5)
        self.draw_centers(self.Acenters, radius=3)
        self.draw_centers(self.Bcenters, radius=3)
        (I, J, K) = A.shape
        vectors = self.vectors
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    v = A[i, j, k]
                    if v > 0:
                        p = vv(i, j, k)
                        v = vectors[i, j, k]
                        if np.abs(v.max()) > 0:
                            #dp = p + v
                            loc1 = self.pos(p)
                            #loc2 = self.pos(dp)
                            loc2 = loc1 + self.scaled_vectors[i, j, k]
                            W.arrow(loc1, loc2, 2)

    def draw_centers(self, centers, radius):
        W = self.W
        l2c = self.label_to_color
        for (v, c) in centers.items():
            pos = self.pos(c)
            c = l2c[v]
            W.rect(pos, radius, radius, c)

    def draw_array(self, A, radius):
        W = self.W
        (I, J, K) = A.shape
        l2c = self.label_to_color
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    v = A[i,j,k]
                    if v > 0:
                        c = l2c[v]
                        pos = self.pos([i, j, k])
                        W.circle(pos, radius, c, fill=False)

    def widget(self, pixels=600):
        from jp_doodle import nd_frame
        A = self.A
        B = self.B
        Acount = len(np.nonzero(A)[0])
        Bcount = len(np.nonzero(B)[0])
        count = Acount + Bcount
        assert count < 2000, "too many labels for this method of display: " + repr(count)
        W = nd_frame.swatch3d(pixels=pixels, model_height=pixels)
        self.W = W
        self.draw(W)
        W.fit(0.6)
        W.orbit_all(1.5 * self.radius, list(self.center))
        return W
