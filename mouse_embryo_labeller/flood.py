"""
Identify isolated regions in a 3D mask using "flooding".
"""

import numpy as np
from mouse_embryo_labeller import color_list
from feedWebGL2 import volume
#from IPython.display import display

class Flood3DMask:

    def __init__(self, mask, niter=1000, mark=True, verbose=True):
        self.volume_widget = None
        labels = np.arange(1, mask.size+1).reshape(mask.shape)
        mask = mask.astype(np.int)
        smask = mask * int(1.1 * mask.size)
        combined = np.maximum(labels, smask)
        current = combined.copy()
        if verbose:
            print("Flooding regions", current.shape)
        for itr in range(niter):
            if verbose:
                print ("flooding", itr)
            nc = current.copy()
            # move up
            nc[1:] = np.minimum(nc[1:], nc[:-1])
            # enforce mask
            nc = np.maximum(nc, smask)
            # move down
            nc[:-1] = np.minimum(nc[:-1], nc[1:])
            # enforce mask
            nc = np.maximum(nc, smask)
            # move right
            nc[:, 1:] = np.minimum(nc[:, 1:], nc[:, :-1])
            # enforce mask
            nc = np.maximum(nc, smask)
            # move down
            nc[:, :-1] = np.minimum(nc[:, :-1], nc[:, 1:])
            # move forward
            nc[:, :, 1:] = np.minimum(nc[:, :, 1:], nc[:, :, :-1])
            # enforce mask
            nc = np.maximum(nc, smask)
            # move backward
            nc[:, :, :-1] = np.minimum(nc[:, :, :-1], nc[:, :, 1:])
            # enforce mask
            nc = np.maximum(nc, smask)
            if np.all(current == nc):
                break
            current = nc
        self.flooded = current
        self.labels = np.unique(current)
        # automatically do marking -- it's fast for reasonable cases
        if mark:
            if verbose:
                print("done flooding, now marking labels", len(self.labels))
            self.mark()

    def mark(self):
        """
        Assign small integer labels and colors to connected regions in flooded array.
        """
        self.nmarks = len(self.labels) - 1
        marked_labels = self.labels[:-1]  # last label is mask marker -- exclude it
        #mark_colors = [[255,255,255]] + color_list.color_arrays[:self.nmarks]
        mark_colors = [[255,255,255]] + color_list.get_colors(self.nmarks)
        mark_to_color = {}
        flooded = self.flooded
        marked_array = np.zeros(flooded.shape, dtype=np.float)
        for i in range(self.nmarks):
            mark = i + 1
            label = marked_labels[i]
            mark_to_color[mark] = mark_colors[mark]
            marked_array[:] = np.where(flooded == label, mark, marked_array)
        self.marked_labels = marked_labels
        self.marked_array = marked_array
        self.mark_to_color = mark_to_color
        self.mark_colors = mark_colors

    def volume(self):
        "Prepare volume visualization for marked array."
        volume.widen_notebook()
        W = self.volume_widget = volume.Volume32()
        return W

    def init_volume(
        self, 
        width=1800,
        di=dict(x=0, y=0, z=1),  # xyz offset between ary[0,0,0] and ary[1,0,0]
        dj=dict(x=0, y=1, z=0),  # xyz offset between ary[0,0,0] and ary[0,1,0]
        dk=dict(x=1, y=0, z=0),  # xyz offset between ary[0,0,0] and ary[0,0,1]
        camera_distance_multiple=1.7,
        ):
        "Display and initialize volume visualization of marked array."
        W = self.volume_widget
        if W is None:
            W = self.volume()
            #display(W) -- display is automatic, don't redisplay.
        W.load_3d_numpy_array(
            self.marked_array, 
            threshold=-0.1,
            di=di,
            dj=dj,
            dk=dk,
            camera_distance_multiple=camera_distance_multiple,
        )
        W.load_label_to_color_mapping(self.mark_to_color)
        W.build(width=width)
