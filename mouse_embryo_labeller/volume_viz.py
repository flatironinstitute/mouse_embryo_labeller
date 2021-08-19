"""
3D view of embryo development.
"""

from feedWebGL2 import volume
from mouse_embryo_labeller import tools
from mouse_embryo_labeller import geometry
import ipywidgets as widgets
import time
import numpy as np

class EmbryoVolume:

    def __init__(
        self,
        folder,
        nuclei_names=None,
        width=1600,
        di=dict(x=0, y=0, z=2),  # xyz offset between ary[0,0,0] and ary[1,0,0]
        dj=dict(x=0, y=1, z=0),  # xyz offset between ary[0,0,0] and ary[0,1,0]
        dk=dict(x=1, y=0, z=0),  # xyz offset between ary[0,0,0] and ary[0,0,1]
        camera_distance_multiple=1.0,
        ):
        self.folder = folder
        self.width = width
        self.nuclei_names = None
        if nuclei_names is not None:
            self.nuclei_names = set(nuclei_names)
        self.di = di
        self.dj = dj
        self.dk = dk
        self.camera_distance_multiple = camera_distance_multiple
        self.nc = tools.get_example_nucleus_collection(folder)
        self.tsc = tools.get_example_timestamp_collection(folder, self.nc)
        # get the timestamps that include the nuclei
        timestamps = []
        for timestamp in self.tsc.timestamp_sequence():
            tnames = timestamp.nucleus_names()
            if (tnames) and ((nuclei_names is None) or (self.nuclei_names & tnames)):
                timestamps.append(timestamp)
        self.timestamps = timestamps
        self.id_to_timestamps = {ts.identifier: ts for ts in timestamps}
        self.volume_widget = None
        self.ts_id = None
        self.slice_union = self.combined_slicing()
        self.combo_widget = None

    def make_widget(self, debug=False, width=None):
        if width is not None:
            self.width = width
        volume.widen_notebook()
        tsid_options = [None] + [ts.identifier for ts in self.timestamps]
        self.ts_dropdown = widgets.Dropdown(
            options=tsid_options,
            value=None,
            description='Timestamp:',
            disabled=False,
        )
        self.ts_dropdown.observe(self.ts_dropdown_change, names='value')
        self.volume_widget = volume.Volume32()
        self.status_widget = widgets.HTML(value="Please select a timestamp.")
        display = self.volume_widget
        if debug:
            display = self.volume_widget.debugging_display()
        self.widget = widgets.VBox([
            self.ts_dropdown,
            display,
            self.status_widget,
        ])
        return self.widget

    def make_combo_widget(self, debug=False, side=1000):
        from mouse_embryo_labeller import viz_controller
        self.side = side
        self.debug = debug
        my_widget = self.make_widget(debug=debug, width=side * 4)
        self.labeller = viz_controller.VizController(self.folder, self.tsc, self.nc)
        self.labeller_widget = self.labeller.make_widget(side)
        self.combo_widget = widgets.VBox([
            self.labeller_widget,
            my_widget,
        ])
        return self.combo_widget

    def reset_combo_widget(self):
        # free up memory...
        self.volume_widget.dispose(verbose=False)
        #my_widget = self.make_widget(debug=self.debug, width=self.side * 4)
        #self.combo_widget.children = [
        #    self.labeller_widget,
        #    my_widget,
        #]

    def capture_combo_image(self, tsid, sleep=0.1):
        assert self.combo_widget is not None
        self.reset_combo_widget()
        labeller = self.labeller
        labeller.tree_view_checkbox.value = True
        labeller.timestamp_input.value = tsid
        tree = labeller.time_tree.widget
        image3d = self.capture_image(tsid, sleep)
        treeimg = tree.pixels_array()
        result = combine_images(treeimg, image3d)
        return result

    def status(self, message):
        self.status_widget.value = str(message)

    def ts_dropdown_change(self, change):
        value = self.ts_dropdown.value
        if value is not None:
            if value != self.ts_id:
                self.status("selected: " + repr(value))
                self.load_timestamp(value)
        else:
            self.status("Cannot change to timestamp of None.")

    def load_timestamp(self, ts_id=None):
        self.status("loading timestamp: " + repr(ts_id))
        if ts_id is None:
            ts_id = self.timestamps[0]
        ts = self.id_to_timestamps[ts_id]
        label_to_color = ts.label_colors(self.nuclei_names)
        ts.load_truncated_arrays()
        l3d = ts.l3d_truncated
        # release memory
        ts.reset_all_arrays()
        #sl = geometry.positive_slicing(l3d)
        sl = self.slice_union
        sliced = geometry.apply_slicing(sl, l3d)
        W = self.volume_widget
        W.load_3d_numpy_array(
            sliced, 
            threshold=-0.1,
            di=self.di,
            dj=self.dj,
            dk=self.dk,
            camera_distance_multiple=self.camera_distance_multiple,
        )
        W.load_label_to_color_mapping(label_to_color)
        W.build(width=self.width)
        self.ts_id = ts_id
        if ts_id != self.ts_dropdown.value:
            self.ts_dropdown.value = ts_id
        self.status("loaded timestamp: " + repr(ts_id))

    def combined_slicing(self):
        "Union of non-empty slices for all relevant timestamps"
        print("computing slicing.")
        slice_union = None
        for ts in self.timestamps:
            #ts.load_truncated_arrays()
            #l3d = ts.l3d_truncated
            # release memory
            #ts.reset_all_arrays()
            #sl = geometry.positive_slicing(l3d)
            sl = ts.nuclei_mask_slicing(self.nuclei_names)
            if slice_union is None:
                slice_union = sl
            else:
                slice_union = geometry.unify_slicing(slice_union, sl)
        print("done computing slicing", slice_union)
        return slice_union

    def capture_image(self, tsid, sleep=0.1):
        self.load_timestamp(tsid)
        time.sleep(sleep)
        self.volume_widget.sync()
        img = self.volume_widget.get_pixels()
        return img

    def capture_images(self, sleep=0.1):
        images = []
        for ts in self.timestamps:
            tsid = ts.identifier
            img = self.capture_image(tsid, sleep)
            images.append(img)
            print("loaded", ts)
        return images

    def capture_combo_images(self, stride=1, sleep=0.1):
        images = []
        timestamps = self.timestamps
        for index in range(0, len(timestamps), stride):
            ts = timestamps[index]
            tsid = ts.identifier
            img = self.capture_combo_image(tsid, sleep)
            images.append(img)
            print("loaded", ts)
        return images
        

def save_images_to_gif(images, filename="animation.gif", duration=1, bookends=True):
    import imageio
    if bookends:
        im0 = images[0]
        black = np.zeros(im0.shape, dtype=np.uint8)
        images = [black] + list(images) + [black]
    imageio.mimsave(filename, images, format="GIF", duration=duration)

def combine_images(left_image, right_image):
    (LH, LW) = left_image.shape[:2]
    (RH, RW) = right_image.shape[:2]
    width = LW + RW
    height = max(LH, RH)
    result = np.zeros((height, width, 4), dtype=np.uint8)
    result[:] = 255
    left3 = left_image[:, :, :3]
    sh = int((height-LH)/2)
    result[sh:sh+LH, :LW, :3] = np.where(left3, left3, 255)
    right3 = right_image[:, :, :3]
    sh = int((height-RH)/2)
    result[sh:sh+RH, LW: width, :3] = np.where(right3, right3, 255)
    result[:, :, 3] = 255
    return result

