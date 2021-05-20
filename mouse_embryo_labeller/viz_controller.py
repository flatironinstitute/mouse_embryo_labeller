
import ipywidgets as widgets
from jp_doodle import array_image
from jp_doodle.data_tables import widen_notebook
from jp_doodle import color_chooser
from mouse_embryo_labeller import nucleus
from mouse_embryo_labeller import color_list
import os


ARROWS = [LEFT, UP, RIGHT, DOWN] = "LEFT UP RIGHT DOWN".split()

NAME_TO_ARROW_KEY_NUMBER = {
    LEFT: 37,
    UP: 38,
    RIGHT: 39,
    DOWN: 40,
}
NUM_TO_KEY = {str(num): key for (key, num) in NAME_TO_ARROW_KEY_NUMBER.items()}

KEYPRESS_JS = """

element.keypress_handler = function(event) {
    var num = event.which;
    var key = NUM_TO_KEY[num];
    if (key) {
        keypress_callback(key);
        event.preventDefault();  // disallow defaults for handled keys
    } else {
        keypress_callback("unknown key press. " + num)
    }
};

var f = element.parent();
f.attr('tabindex', 0);
f.keydown(element.keypress_handler);

element.keypress_focus = function () {
    f.focus();
};

element.keypress_focus();
"""

def attach_keypress_handler(to_proxy_widget, keypress_callback, num_to_key=NUM_TO_KEY):
    to_proxy_widget.js_init(KEYPRESS_JS, NUM_TO_KEY=num_to_key, keypress_callback=keypress_callback)


class VizController:

    """
    Coordinator for tracker visualization tool.
    """

    def __init__(self, folder, timestamp_collection, nucleus_collection):
        self.folder = folder
        self.timestamp_collection = timestamp_collection
        self.nucleus_collection = nucleus_collection
        self.selected_timestamp_id = self.timestamp_collection.first_id()
        #self.selected_nucleus_id = None
        self.last_timestamp = None
        ts = self.timestamp()
        self.selected_layer = ts.nlayers() - 1
        self.nucleus_info = widgets.HTML(value="Please select or create a nucleus.")
        #self.set_nucleus_id(None)  # nope, not initialized yet
        self.selected_nucleus_id = None
        nucleus_collection.set_controller(self)

    switch_count = 0

    def timestamp(self):
        last = self.last_timestamp
        current = self.timestamp_collection.get_timestamp(self.selected_timestamp_id)
        if last is not current:
            # clear the old timestamp (don't waste memory) amd load the new one
            #print("SWITCHING", repr([last, current]))
            if last is not None:
                last.reset_all_arrays()
            current.load_truncated_arrays()
            self.switch_count += 1
        self.last_timestamp = current
        return current

    def get_nucleus(self):
        nid = self.selected_nucleus_id
        if nid is None:
            return None
        else:
            return self.nucleus_collection.get_nucleus(nid)

    def make_widget(self, side=350):
        widen_notebook()
        self.side = side
        ts = self.timestamp()

        self.info = widgets.HTML(value="Nucleus <br> Labeller <br> Tool.")
        #self.nucleus_info = widgets.HTML(value="Please select or create an nucleus.") # moved to __init__
        self.delete_button = widgets.Button(description="DELETE", disabled=True)
        self.delete_button.on_click(self.delete_click)
        info_area1 = widgets.HBox([self.info], layout=widgets.Layout(border='solid'))
        reparent_assembly = self.nucleus_collection.reparent_assembly()
        join_assembly = self.nucleus_collection.join_assembly()
        info_area = widgets.HBox([
            info_area1, 
            self.nucleus_info, 
            self.delete_button,
            reparent_assembly,
            join_assembly, 
            ])

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
        self.colorize_checkbox = widgets.Checkbox(
            value=False,
            description="pseudocolors",
        )
        self.colorize_checkbox.observe(self.redraw_on_change, names='value')
        layers_assembly = widgets.VBox([
            self.layers_slider,
            self.colorize_checkbox,
        ])
        self.extruded_checkbox = widgets.Checkbox(
            value=True,
            description="extruded",
        )
        self.extruded_checkbox.observe(self.redraw_on_change, names='value')
        self.blur_checkbox = widgets.Checkbox(
            value=True,
            description="blur",
        )
        checkboxen = widgets.VBox([
            self.extruded_checkbox,
            self.blur_checkbox,
        ])
        self.blur_checkbox.observe(self.redraw_on_change, names='value')
        top_bar = widgets.HBox([
            self.prev_button,
            self.timestamp_html,
            self.next_button,
            layers_assembly,
            checkboxen,
        ])
        rimage = self.raster_image(ts)
        self.raster_display = array_image.show_array(
            rimage, 
            height=side, 
            width=side,
            hover_text_callback=self.raster_hover_text,
        )
        limage = self.label_image(ts)
        self.labelled_image_display = array_image.show_array(
            limage, 
            height=side, 
            width=side, 
            scale=False,
            hover_text_callback=self.label_image_hover,
        )
        self.labelled_image_display.image_display.on("click", self.label_image_click)
        image_assembly = widgets.HBox([
            self.raster_display,
            self.labelled_image_display,
        ])
        attach_keypress_handler(self.raster_display, self.keypress_callback)
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
        self.child_button.on_click(self.child_click)
        self.color_selector = color_chooser.chooser(side=0.5 * side, callback=self.change_color)
        self.color_display = widgets.HTML(value="no color")
        self.color_assembly = widgets.VBox([
            self.color_display,
            self.color_selector,
        ], layout=widgets.Layout(border='solid', visibility="hidden"))
        make_nucleus_assembly = widgets.HBox([
            self.nucleus_name_input,
            self.color_assembly,
            self.child_button,
            self.new_button,
            #self.nucleus_info,
        ])
        self.left_assambly = widgets.VBox([
            info_area,
            top_bar,
            image_assembly,
            make_nucleus_assembly,
        ])
        self.nucleus_chooser = self.nucleus_collection.create_widget(
            height=side * 1.5,
            callback=self.set_nucleus_id,
        )
        self.widget = widgets.HBox([
            self.left_assambly,
            self.nucleus_chooser,
        ])
        # reset colors etc
        self.change_color()
        self.hide_color_selector()
        # fix up buttons, etcetera
        self.redraw()
        return self.widget

    def keypress_callback(self, txt):
        self.info.value = "Key press: " + repr(txt)
        current_layer = self.current_layer()
        if txt == UP:
            self.change_layer(current_layer + 1)
        elif txt == DOWN:
            self.change_layer(current_layer - 1)
        elif txt == RIGHT:
            self.go_next(None)
        elif txt == LEFT:
            self.go_previous(None)
        else:
            self.info.value = "Use \u21e6 \u21e8 for time; \u21e7 \u21e9 for layers"

    def current_layer(self):
        return self.layers_slider.value

    def change_layer(self, new_value):
        ts = self.timestamp()
        if new_value < 0:
            self.info.value = "No previous layer: " + repr(new_value)
        elif new_value >= ts.nlayers():
            self.info.value = "No next layer: " + repr(new_value)
        else:
            self.info.value = "Change layer: " + repr(new_value)
            self.layers_slider.value = new_value

    def hide_color_selector(self):
        #self.color_selector.element.hide()
        self.color_assembly.layout.visibility = "hidden"

    def show_color_selector(self):
        #self.color_selector.element.show()
        self.color_assembly.layout.visibility = "visible"

    def delete_click(self, button):
        del_id = self.selected_nucleus_id
        n = self.get_nucleus()
        assert n is not None, "can't delete no nucleus found " + repr(del_id)
        #self.selected_nucleus_id = None
        self.set_nucleus_id(None)
        self.timestamp_collection.forget_nucleus(n, self)
        self.nucleus_collection.forget_nucleus_id(del_id, self.folder)
        self.nucleus_collection.set_widget_options(callback=None, selected=None)
        self.redraw()
        self.info.value = "DELETED " + repr(del_id)

    def set_nucleus_id(self, identifier):
        if self.nucleus_collection.get_nucleus(identifier, check=False):
            self.selected_nucleus_id = identifier
        else:
            self.selected_nucleus_id = None
        self.nucleus_collection.set_selected_nucleus(self.selected_nucleus_id)
        self.nucleus_collection.set_widget_options(callback=None, selected=self.selected_nucleus_id)
        self.show_nucleus_selection()

    def show_nucleus_selection(self):
        n = self.get_nucleus()
        info = self.nucleus_info
        if n is None:
            info.value = "No nucleus currently selected."
            self.delete_button.disabled = True
        else:
            #info.value = '<span style="background-color:%s">NUCLEUS</span> %s' % (n.html_color(), n.identifier)
            info.value = n.html_info(self.nucleus_collection)
            self.delete_button.disabled = False

    def label_image_click(self, event):
        position = event['model_location']
        x = int(position["x"])
        y = int(position["y"])
        self.info.value = "Click: " + repr((x, y))
        ts = self.timestamp()
        extruded = self.extruded_checkbox.value
        layer = self.layers_slider.value
        label = ts.get_label(layer, y, x, extruded=extruded)
        self.info.value = "label " + repr(label)
        if label == 0:
            self.info.value = "Cannot assign unlabelled area."
            return
        self.assign_nucleus(label)

    def assign_nucleus(self, label):
        ts = self.timestamp()
        n = self.get_nucleus()
        ts.assign_nucleus(label, n)
        tsid = self.selected_timestamp_id
        if n is None:
            self.info.value = "Reset nucleus assignment for label %s in timestamp %s." % (label, tsid)
        else:
            self.info.value = "Assigned label %s to nucleus %s in timestamp %s." % (label, n.identifier, tsid)
        # save the assignment
        path = self.ts_assignment_path()
        ts.save_mapping(path)
        self.redraw()

    def ts_assignment_path(self, tsid=None):
        if tsid is None:
            tsid = self.selected_timestamp_id
        assert tsid is not None, "cannot save -- no selected ts"
        filename = "ts%s.json" % tsid
        return os.path.join(self.folder, filename)

    def raster_hover_text(self, x, y, array):
        ts = self.timestamp()
        layer = self.layers_slider.value
        #tsid = self.selected_timestamp_id
        #prefix = repr((y, x)) + ": "
        extruded = self.extruded_checkbox.value
        intensity = ts.get_intensity(layer, y, x, extruded)
        return "%s,%s : %s" % (y, x, intensity)

    def label_image_hover(self, x, y, array):
        ts = self.timestamp()
        layer = self.layers_slider.value
        tsid = self.selected_timestamp_id
        prefix = repr((y, x)) + ": "
        suffix = "error"
        extruded = self.extruded_checkbox.value
        try:
            label = ts.get_label(layer, y, x, extruded=extruded)
            if label == 0:
                suffix = "unlabelled:0"
            else:
                n = ts.get_nucleus(label)
                if n is not None:
                    suffix = "%s :: %s" % (label, n.identifier)
                else:
                    suffix = repr(label) + " no nucleus"
        except Exception:
            raise
        return prefix + suffix

    def child_click(self, button):
        return self.new_click(button, parent_id=self.selected_nucleus_id)

    def new_click(self, button, parent_id=None):
        self.info.value = "New clicked."
        identifier = self.nucleus_name_input.value
        #color = self.color_selector.color_array
        color = self.color_array
        n = nucleus.Nucleus(identifier, color, parent_id)
        self.nucleus_collection.add_nucleus(n)
        self.nucleus_collection.save_json(self.folder)
        # only switch to nucleus if no parent
        if parent_id is None:
            #self.selected_nucleus_id = identifier
            self.set_nucleus_id(identifier)
        self.nucleus_collection.set_widget_options(callback=None, selected=identifier)
        self.show_nucleus_selection()
        self.nucleus_name_input.value = ""
        self.new_button.disabled = True
        self.child_button.disabled = True
        self.color_selector.reset_color_choice()
        self.color_display.value = "No color."
        self.hide_color_selector()

    def nucleus_name_change(self, change):
        if self.nucleus_name_input.value:
            self.show_color_selector()
            self.color_selector.reset_color_choice()
            self.info.value = "Please select a color for nucleus."
        else:
            self.hide_color_selector()
        self.child_button.disabled = True
        self.new_button.disabled = True
        self.pick_unused_color()

    def pick_unused_color(self):
        color_array = color_list.color_arrays[-1]
        in_use = self.nucleus_collection.colors_in_use()
        for array in color_list.color_arrays:
            if tuple(array) not in in_use:
                color_array = array
        self.change_color(color_array)

    def change_color(self, color_array=None, html_color=None):
        self.color_array = color_array
        if color_array is not None:
            self.html_color = color_list.rgbhtml(color_array)
            self.color_display.value = color_list.colordiv(color_array)
        name = self.nucleus_name_input.value
        if (color_array is not None) and (name != "") and (self.nucleus_collection.get_nucleus(name, check=False) is None):
            self.new_button.disabled = False
            self.child_button.disabled = (self.selected_nucleus_id is None)

    def redraw_on_change(self, change):
        if change['new'] != change['old']:
            self.redraw()

    def go_next(self, button):
        (prv, nxt) = self.previous_next()
        if nxt is None:
            self.info.value = "No next timestamp: " + repr(self.selected_timestamp_id)
            return
        self.selected_timestamp_id = nxt
        self.info.value = "next timestamp: " + repr(nxt)
        self.redraw()

    def go_previous(self, button):
        (prv, nxt) = self.previous_next()
        if prv is None:
            self.info.value = "No previous timestamp: " + repr(self.selected_timestamp_id)
            return
        self.selected_timestamp_id = prv
        self.info.value = "previous timestamp: " + repr(prv)
        self.redraw()

    def previous_next(self):
        tsc = self.timestamp_collection
        tsid = self.selected_timestamp_id
        return tsc.previous_next(tsid)

    def redraw(self):
        #tsc = self.timestamp_collection
        self.show_nucleus_selection()
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
        blur = self.blur_checkbox.value
        colorize = self.colorize_checkbox.value
        return ts.raster_slice_with_boundary(layer, extruded, blur=blur, colorize=colorize)
 