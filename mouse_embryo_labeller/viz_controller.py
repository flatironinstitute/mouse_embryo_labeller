
import ipywidgets as widgets
from jp_doodle import array_image
from jp_doodle.data_tables import widen_notebook


class VizController:

    """
    Coordinator for tracker visualization tool.
    """

    def __init__(self, timestamp_collection, nucleus_collection):
        self.timestamp_collection = timestamp_collection
        self.nucleus_collection = nucleus_collection
        self.selected_timestamp_id = self.timestamp_collection.first_id()
        self.selected_nucleus = None
        ts = self.timestamp()
        self.selected_layer = ts.nlayers() - 1

    def timestamp(self):
        return self.timestamp_collection.get_timestamp(self.selected_timestamp_id)

    def make_widget(self, side=300):
        widen_notebook()
        self.side = side
        ts = self.timestamp()
        self.prev_button = widgets.Button(description="< Prev")
        self.prev_button.on_click(self.go_previous)
        self.timestamp_html = widgets.HTML(value=repr(self.selected_timestamp_id))
        self.next_button = widgets.Button(description="Next >")
        self.next_button.on_click(self.go_next)
        self.layers_slider = widgets.IntSlider(
            value=self.selected_layer,
            min=0,
            max=ts.nlayers() - 1,
            step=1,
            description="layer",
        )
        self.layers_slider.observe(self.redraw_on_change, names='value')
        self.extruded_checkbox = widgets.Checkbox(
            value=True,
            description="extruded",
        )
        self.extruded_checkbox.observe(self.redraw_on_change, names='value')
        top_bar = widgets.HBox([
            self.prev_button,
            self.timestamp_html,
            self.next_button,
            self.layers_slider,
            self.extruded_checkbox,
        ])
        rimage = self.raster_image(ts)
        self.raster_display = array_image.show_array(rimage, height=side, width=side)
        limage = self.label_image(ts)
        self.labelled_image_display = array_image.show_array(limage, height=side, width=side, scale=False)
        image_assembly = widgets.HBox([
            self.labelled_image_display,
            self.raster_display,
        ])
        self.left_assambly = widgets.VBox([
            top_bar,
            image_assembly,
        ])
        self.nucleus_chooser = self.nucleus_collection.create_widget(height=side)
        self.widget = widgets.HBox([
            self.left_assambly,
            self.nucleus_chooser,
        ])
        # fix up buttons, etcetera
        self.redraw()
        return self.widget

    def redraw_on_change(self, change):
        if change['new'] != change['old']:
            self.redraw()

    def go_next(self, button):
        (prv, nxt) = self.previous_next()
        assert nxt is not None, "No next timestamp: " + repr(self.selected_timestamp_id)
        self.selected_timestamp_id = nxt
        self.redraw()

    def go_previous(self, button):
        (prv, nxt) = self.previous_next()
        assert prv is not None, "No previous timestamp: " + repr(self.selected_timestamp_id)
        self.selected_timestamp_id = prv
        self.redraw()

    def previous_next(self):
        tsc = self.timestamp_collection
        tsid = self.selected_timestamp_id
        return tsc.previous_next(tsid)

    def redraw(self):
        #tsc = self.timestamp_collection
        tsid = self.selected_timestamp_id
        self.timestamp_html.value = repr(tsid)
        (prv, nxt) = self.previous_next()
        self.prev_button.disabled = (prv is None)
        self.next_button.disabled = (nxt is None)
        ts = self.timestamp()
        rimage = self.raster_image(ts)
        self.raster_display.set_image(rimage)
        limage = self.label_image(ts)
        self.labelled_image_display.set_image(limage)

    def label_image(self, ts):
        color_mapping = ts.colorization_mapping()
        extruded = self.extruded_checkbox.value
        layer = self.layers_slider.value
        return ts.colorized_label_slice(
            color_mapping_array=color_mapping,
            slice_i=layer, 
            extruded=extruded,
            )

    def raster_image(self, ts):
        extruded = self.extruded_checkbox.value
        layer = self.layers_slider.value
        return ts.raster_slice_with_boundary(layer, extruded)