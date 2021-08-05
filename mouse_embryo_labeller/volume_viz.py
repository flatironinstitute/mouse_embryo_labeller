"""
3D view of embryo development.
"""

from feedWebGL2 import volume
from mouse_embryo_labeller import tools
from mouse_embryo_labeller import geometry
import ipywidgets as widgets

class EmbryoVolume:

    def __init__(
        self,
        folder,
        nuclei_names=None,
        width=1600,
        di=dict(x=0, y=0, z=2),  # xyz offset between ary[0,0,0] and ary[1,0,0]
        dj=dict(x=0, y=1, z=0),  # xyz offset between ary[0,0,0] and ary[0,1,0]
        dk=dict(x=1, y=0, z=0),  # xyz offset between ary[0,0,0] and ary[0,0,1]
        ):
        self.folder = folder
        self.width = width
        self.nuclei_names = None
        if nuclei_names is not None:
            self.nuclei_names = set(nuclei_names)
        self.di = di
        self.dj = dj
        self.dk = dk
        self.nc = tools.get_example_nucleus_collection(folder)
        self.tsc = tools.get_example_timestamp_collection(folder, self.nc)
        # get the timestamps that include the nuclei
        timestamps = []
        for timestamp in self.tsc.timestamp_sequence():
            tnames = timestamp.nucleus_names()
            if (nuclei_names is None) or (self.nuclei_names & tnames):
                timestamps.append(timestamp)
        self.timestamps = timestamps
        self.id_to_timestamps = {ts.identifier: ts for ts in timestamps}
        self.volume_widget = None
        self.ts_id = None

    def make_widget(self, debug=False):
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
        sl = geometry.positive_slicing(l3d)
        sliced = geometry.apply_slicing(sl, l3d)
        W = self.volume_widget
        W.load_3d_numpy_array(
            sliced, 
            threshold=-0.1,
            di=self.di,
            dj=self.dj,
            dk=self.dk,
        )
        W.load_label_to_color_mapping(label_to_color)
        W.build(width=self.width)
        self.ts_id = ts_id
        if ts_id != self.ts_dropdown.value:
            self.ts_dropdown.value = ts_id
        self.status("loaded timestamp: " + repr(ts_id))

    def capture_images(self, sleep=1):
        import time
        images = []
        for ts in self.timestamps:
            tsid = ts.identifier
            self.load_timestamp(tsid)
            time.sleep(sleep)
            self.volume_widget.sync()
            img = self.volume_widget.get_pixels()
            images.append(img)
            print("loaded", ts)
        return images

