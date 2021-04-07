
import ipywidgets as widgets
from jp_doodle import array_image
from jp_doodle.data_tables import widen_notebook
from jp_doodle import color_chooser
from mouse_embryo_labeller import nucleus


class VizController:

    """
    Coordinator for tracker visualization tool.
    """

    def __init__(self, folder, timestamp_collection, nucleus_collection):
        self.folder = folder
        self.timestamp_collection = timestamp_collection
        self.nucleus_collection = nucleus_collection
        self.selected_timestamp_id = self.timestamp_collection.first_id()
        self.selected_nucleus_id = None
        ts = self.timestamp()
        self.selected_layer = ts.nlayers() - 1

    def timestamp(self):
        return self.timestamp_collection.get_timestamp(self.selected_timestamp_id)

    def make_widget(self, side=350):
        widen_notebook()
        self.side = side
        ts = self.timestamp()
        self.info = widgets.HTML(value="Nucleus Labeller Tool")
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
        # nucleus creation controls
        #self.nucleus_info = widgets.HTML(value="Enter name.")
        self.nucleus_name_input = widgets.Text(
            value='',
            placeholder='identifier',
            description='New Nucleus:',
            disabled=False
        )
        self.nucleus_name_input.observe(self.nucleus_name_change, names='value')
        self.new_button = widgets.Button(description="New", disabled=True)
        self.new_button.on_click(self.new_click)
        self.child_button = widgets.Button(description="Child", disabled=True)
        self.color_selector = color_chooser.chooser(side=0.5 * side, callback=self.change_color)
        make_nucleus_assembly = widgets.HBox([
            self.nucleus_name_input,
            self.color_selector,
            self.child_button,
            self.new_button,
            #self.nucleus_info,
        ])
        self.left_assambly = widgets.VBox([
            self.info,
            top_bar,
            image_assembly,
            make_nucleus_assembly,
        ])
        self.nucleus_chooser = self.nucleus_collection.create_widget(height=side * 1.5)
        self.widget = widgets.HBox([
            self.left_assambly,
            self.nucleus_chooser,
        ])
        # reset colors
        self.change_color()
        # fix up buttons, etcetera
        self.redraw()
        return self.widget

    def new_click(self, button):
        self.info.value = "New clicked."
        identifier = self.nucleus_name_input.value
        color = self.color_selector.color_array
        parent_id = None
        n = nucleus.Nucleus(identifier, color, parent_id)
        self.nucleus_collection.add_nucleus(n)
        self.nucleus_collection.save_json(self.folder)
        self.selected_nucleus_id = identifier
        self.nucleus_collection.set_widget_options(callback=None, selected=identifier)

    def nucleus_name_change(self, change):
        self.color_selector.reset_color_choice()
        self.child_button.disabled = True
        self.new_button.disabled = True

    def change_color(self, color_array=None, html_color=None):
        self.color_array = color_array
        self.html_color = html_color
        name = self.nucleus_name_input.value
        if (name != "") and (self.nucleus_collection.get_nucleus(name, check=False) is None):
            self.new_button.disabled = False
            self.child_button.disabled = (self.selected_nucleus_id is None)

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