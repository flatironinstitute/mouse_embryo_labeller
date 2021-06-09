
from mouse_embryo_labeller.geometry import positive_extent_info
from mouse_embryo_labeller.timestamp_collection import load_preprocessed_timestamps
import ipywidgets as widgets
from jp_doodle import array_image
from jp_doodle.data_tables import widen_notebook
from jp_doodle import color_chooser
from mouse_embryo_labeller import nucleus, nucleus_collection
from mouse_embryo_labeller import color_list
import os

INVISIBLE = "rgba(0,0,0,0)"

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

VISIBILITY_MAP = {
    True: "visible",
    False: "hidden",
}

def attach_keypress_handler(to_proxy_widget, keypress_callback, num_to_key=NUM_TO_KEY):
    to_proxy_widget.js_init(KEYPRESS_JS, NUM_TO_KEY=num_to_key, keypress_callback=keypress_callback)


class VizController:

    """
    Coordinator for tracker visualization tool.
    """

    def __init__(self, folder, timestamp_collection, nucleus_collection):
        self.widget = None
        self.folder = folder
        self.timestamp_collection = timestamp_collection
        self.nucleus_collection = nucleus_collection
        self.selected_timestamp_index = 0
        #self.selected_nucleus_id = None
        self.last_timestamp = None
        ts = self.timestamp()
        self.selected_layer = ts.nlayers() - 1
        self.nucleus_info = widgets.HTML(value="Please select or create a nucleus.")
        #self.set_nucleus_id(None)  # nope, not initialized yet
        self.selected_nucleus_id = None
        nucleus_collection.set_controller(self)
        self.snapshot_folder = os.path.join(folder, "snapshots")
        self.reverted_folder = os.path.join(folder, "reverted")
        # indicator -- tree view or image view?
        self.show_tree_view = False
        self.colorize_raster = False
        self.show_max_intensities = True
        self.time_tree = TimeTreeWidget(self.nucleus_collection, self.timestamp_collection, self)

    def reload(self):
        "Reload data sources from files."
        self.nucleus_collection = nucleus_collection.collection_from_json(self.folder)
        self.nucleus_collection.set_controller(self)
        self.timestamp_collection = load_preprocessed_timestamps(self.folder, self.nucleus_collection)
        self.selected_timestamp_index = 0
        self.last_timestamp = None
        ts = self.timestamp()
        self.selected_layer = ts.nlayers() - 1
        self.selected_nucleus_id = None

    def calculate_stats(self):
        self.nucleus_collection.reset_stats()
        self.timestamp_collection.reset_stats()
        self.nucleus_collection.assign_children()
        self.nucleus_collection.assign_widths()
        self.timestamp_collection.assign_indices()
        self.nucleus_collection.assign_positions()

    switch_count = 0

    def timestamp(self):
        last = self.last_timestamp
        current = self.timestamp_collection.get_indexed_timestamp(self.selected_timestamp_index)
        if last is not current:
            # clear the old timestamp (don't waste memory) amd load the new one
            #print("SWITCHING", repr([last, current]))
            if last is not None:
                last.reset_all_arrays()
            current.load_truncated_arrays()
            self.switch_count += 1
        self.last_timestamp = current
        return current

    def current_timestamp_id(self):
        return self.timestamp().identifier

    def get_nucleus(self):
        nid = self.selected_nucleus_id
        if nid is None:
            return None
        else:
            return self.nucleus_collection.get_nucleus(nid)

    last_side = 350

    def make_widget(self, side=None):
        if side is None:
            side = self.last_side
        self.last_side = side
        widen_notebook()
        self.side = side
        ts = self.timestamp()

        self.info = widgets.HTML(value="Nucleus <br> Labeller <br> Tool.")
        #self.nucleus_info = widgets.HTML(value="Please select or create an nucleus.") # moved to __init__
        self.delete_button = widgets.Button(description="DELETE", disabled=True)
        self.delete_button.on_click(self.delete_click)
        self.change_name_input = widgets.Text(
            value='',
            placeholder='identifier',
            description='Rename',
            disabled=False
        )
        self.change_name_input.observe(self.change_name_change, names='value')
        self.rename_button = widgets.Button(description="RENAME", disabled=True)
        self.rename_button.on_click(self.rename_click)
        nucleus_info_area = widgets.VBox([
            self.nucleus_info,
            self.delete_button,
            self.change_name_input,
            self.rename_button,
        ])
        info_area1 = widgets.HBox([self.info], layout=widgets.Layout(border='solid'))
        reparent_assembly = self.nucleus_collection.reparent_assembly()
        join_assembly = self.nucleus_collection.join_assembly()
        split_assembly = self.nucleus_collection.split_assembly()
        info_area = widgets.HBox([
            info_area1, 
            nucleus_info_area,
            reparent_assembly,
            join_assembly,
            split_assembly,
            ])
        #self.prev_button = widgets.Button(description="< Prev")
        #self.prev_button.on_click(self.go_previous)
        #self.next_button = widgets.Button(description="Next >")
        #self.next_button.on_click(self.go_next)
        self.timestamp_input = widgets.BoundedIntText(
            value=self.selected_timestamp_index,
            min=0,
            max=len(self.timestamp_collection.id_sequence),
            step=1,
            description='Timestamp:',
            disabled=False
        )
        self.timestamp_input.observe(self.redraw_on_change, names='value')
        #self.timestamp_input.on_submit(self.redraw_on_change)  # doesn't work
        self.tree_view_checkbox = widgets.Checkbox(
            value=self.show_tree_view,
            description="tree_view",
        )
        self.tree_view_checkbox.observe(self.tree_view_change, names='value')
        #timestamp_buttons = widgets.HBox([
        #    #self.prev_button,
        #    self.timestamp_html,
        #    #self.next_button,
        #])
        #timestamp_assembly = widgets.VBox([timestamp_buttons, self.tree_view_checkbox])
        self.layers_slider = widgets.IntSlider(
            value=self.selected_layer,
            min=0,
            max=ts.nlayers() - 1,
            step=1,
            description="layer",
        )
        self.layers_slider.observe(self.redraw_on_change, names='value')
        self.colorize_checkbox = widgets.Checkbox(
            value=self.colorize_raster,
            description="pseudocolors",
        )
        self.colorize_checkbox.observe(self.redraw_on_change, names='value')
        self.max_intensity_checkbox = widgets.Checkbox(
            value=self.show_max_intensities,
            description="max intensity",
        )
        self.max_intensity_checkbox.observe(self.redraw_on_change, names='value')
        #layers_assembly = widgets.VBox([
        #    self.layers_slider,
        #    self.colorize_checkbox,
        #])
        coords_visible = VISIBILITY_MAP[not self.show_tree_view]
        coords_assembly = widgets.HBox([
            self.timestamp_input,
            self.layers_slider,
            self.tree_view_checkbox,
        ], )
        self.extruded_checkbox = widgets.Checkbox(
            value=True,
            description="extruded",
        )
        self.extruded_checkbox.observe(self.redraw_on_change, names='value')
        self.blur_checkbox = widgets.Checkbox(
            value=True,
            description="blur",
        )
        self.raster_assembly = widgets.HBox([
            self.max_intensity_checkbox,
            self.colorize_checkbox,
            self.blur_checkbox,
        ], layout=dict(visibility=coords_visible))
        #LHSpresentation_assembly = widgets.HBox([
        #    self.tree_view_checkbox,
        #    self.raster_assembly,
        #])
        LHSpresentation_assembly = self.raster_assembly
        LHScontrols_assembly = widgets.VBox([
            coords_assembly,
            LHSpresentation_assembly,
        ])
        #RHScontrols_assembly = widgets.VBox([
        #    self.extruded_checkbox,
        #    self.blur_checkbox,
        #])
        RHScontrols_assembly = self.extruded_checkbox
        self.blur_checkbox.observe(self.redraw_on_change, names='value')
        top_bar = widgets.HBox([
            #self.prev_button,
            #self.timestamp_html,
            #self.next_button,
            #timestamp_assembly,
            #layers_assembly,
            LHScontrols_assembly,
            RHScontrols_assembly,
        ])
        rimage = self.raster_image(ts)
        if self.show_tree_view:
            self.raster_display = None
            self.calculate_stats()
            #self.time_tree = TimeTreeWidget(self.nucleus_collection, self.timestamp_collection, self)
            self.left_box = self.time_tree.make_widget(side, side)
        else:
            #self.time_tree = None
            self.raster_display = array_image.show_array(
                rimage, 
                height=side, 
                width=side,
                hover_text_callback=self.raster_hover_text,
            )
            self.left_box = self.raster_display
        #self.left_box = self.left_box.debugging_display() # DEBUG
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
            self.left_box,
            self.labelled_image_display,
        ])
        #attach_keypress_handler(self.left_box, self.keypress_callback)
        attach_keypress_handler(self.labelled_image_display, self.keypress_callback)
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
        self.left_assembly = widgets.VBox([
            info_area,
            top_bar,
            image_assembly,
            make_nucleus_assembly,
        ])
        self.nucleus_chooser = self.nucleus_collection.create_widget(
            height=side * 1.5,
            callback=self.set_nucleus_id,
        )
        self.snapshot_button = widgets.Button(description="Snapshot", disabled=False)
        self.snapshot_button.on_click(self.snapshot_click)
        self.revert_buttton = widgets.Button(description="Revert", disabled=False)  # xxx make disabled "smart"
        self.revert_buttton.on_click(self.revert_click)
        self.right_assembly = widgets.VBox([
            self.nucleus_chooser,
            self.snapshot_button,
            self.revert_buttton,
        ])
        widget_children = [
            self.left_assembly,
            self.right_assembly,
        ]
        if self.widget is None:
            self.widget = widgets.HBox(widget_children)
        else:
            self.widget.children = widget_children
        # reset colors etc
        self.change_color()
        self.hide_color_selector()
        # fix up buttons, etcetera
        self.redraw()
        return self.widget

    def tree_view_change(self, change):
        value = change["new"]
        if value != change["old"]:
            self.show_tree_view = value
            # redisplay all
            self.make_widget()
            self.info.value = "tree view <br> " + repr(value)

    def revert_click(self, button):
        from mouse_embryo_labeller import tools
        import shutil
        # find the most recent snapshot
        source_folder = self.snapshot_folder
        snapshots = list(sorted(os.listdir(source_folder)))
        if not snapshots:
            self.info.value = "No snapshots"
            return
        most_recent = snapshots[-1]
        # snap current state to "reverted"
        self.snapshot_click(button, destination=self.reverted_folder)
        # move most recent snapshot files to local...
        from_folder = os.path.join(source_folder, most_recent)
        to_folder = self.folder
        tools.copy_json_files(from_folder, to_folder)
        # move the reverted timestamp to the reverted area
        reverted_folder = os.path.join(self.reverted_folder, most_recent)
        shutil.move(from_folder, reverted_folder)
        # ingest the files...
        self.reload()
        # redisplay...
        self.make_widget()
        self.info.value = "reverted <br> " + repr(most_recent)

    def snapshot_click(self, button, destination=None):
        from mouse_embryo_labeller import tools
        timestamp = tools.timestamp_string_now()
        from_folder = self.folder
        if destination is None:
            destination = self.snapshot_folder
        to_folder = os.path.join(destination, timestamp)
        os.makedirs(to_folder, exist_ok=True)
        tools.copy_json_files(from_folder, to_folder)
        self.info.value = "Snap <br> " + repr(timestamp)

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
        #self.redraw()
        self.make_widget()
        self.info.value = "DELETED " + repr(del_id)

    def relabel_and_delete(self, old_id, replacement_id):
        old_nucleus = self.nucleus_collection.get_nucleus(old_id)
        replacement_nucleus = self.nucleus_collection.get_nucleus(replacement_id)
        self.timestamp_collection.relabel(old_nucleus, replacement_nucleus, self)
        self.nucleus_collection.forget_nucleus_id(old_id, self.folder)
        self.nucleus_collection.set_widget_options(callback=None, selected=None)
        #self.redraw()
        self.make_widget()
        self.info.value = "JOINED " + repr(old_id) + ">>" + repr(replacement_id)

    def split_right(self, split_id):
        old_nucleus = self.get_nucleus()
        new_nucleus = self.nucleus_collection.get_nucleus(split_id, check=False)
        from_timestamp_id = self.current_timestamp_id()
        self.timestamp_collection.split_right(from_timestamp_id, old_nucleus, new_nucleus)
        self.redraw()
        self.info.value = "SPLIT <br> %s: %s>>%s" % (from_timestamp_id, self.selected_nucleus_id, split_id)

    def set_nucleus_id(self, identifier):
        if self.nucleus_collection.get_nucleus(identifier, check=False):
            self.selected_nucleus_id = identifier
        else:
            self.selected_nucleus_id = None
        #self.nucleus_collection.set_selected_nucleus(self.selected_nucleus_id) -- causes infinite recursion.
        self.nucleus_collection.set_widget_options(callback=None, selected=self.selected_nucleus_id)
        self.show_nucleus_selection()
        self.check_highlights()

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
        tsid = self.current_timestamp_id()
        if n is None:
            self.info.value = "Reset nucleus assignment <br> for label %s in timestamp %s." % (label, tsid)
        else:
            self.info.value = "Assigned label <br> %s to nucleus %s in timestamp %s." % (label, n.identifier, tsid)
        # save the assignment
        path = self.ts_assignment_path()
        ts.save_mapping(path)
        self.redraw()

    def ts_assignment_path(self, tsid=None):
        if tsid is None:
            tsid = self.current_timestamp_id()
        assert tsid is not None, "cannot save -- no selected ts"
        filename = "ts%s.json" % tsid
        return os.path.join(self.folder, filename)

    def raster_hover_text(self, x, y, array):
        ts = self.timestamp()
        layer = self.layers_slider.value
        #prefix = repr((y, x)) + ": "
        extruded = self.extruded_checkbox.value
        intensity = ts.get_intensity(layer, y, x, extruded)
        return "%s,%s : %s" % (y, x, intensity)

    def label_image_hover(self, x, y, array):
        ts = self.timestamp()
        layer = self.layers_slider.value
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
        self.calculate_stats()
        # only switch to nucleus if no parent
        select_id = parent_id
        if select_id is None:
            select_id = identifier
        self.set_nucleus_id(select_id)
        self.nucleus_collection.set_widget_options(callback=None, selected=identifier)
        self.show_nucleus_selection()
        self.nucleus_name_input.value = ""
        self.new_button.disabled = True
        self.child_button.disabled = True
        self.color_selector.reset_color_choice()
        self.color_display.value = "No color."
        self.hide_color_selector()

    def change_name_change(self, change):
        value = self.change_name_input.value
        valid = self.nucleus_collection.valid_new_name(value)
        if valid and (self.selected_nucleus_id is not None):
            self.rename_button.disabled = False
        else:
            self.rename_button.disabled = True

    def rename_click(self, button):
        identifier = self.change_name_input.value
        current_nucleus = self.get_nucleus()
        old_id = current_nucleus.identifier
        parent_id = current_nucleus.parent_id
        color = current_nucleus.color
        # create a new nucleus track
        n = nucleus.Nucleus(identifier, color, parent_id)
        self.nucleus_collection.add_nucleus(n)
        # subsume the current nucleus
        self.set_nucleus_id(identifier)
        self.relabel_and_delete(old_id, identifier)
        # reset input widgets
        self.change_name_input.value = ""
        self.rename_button.disabled = True

    def nucleus_name_change(self, change):
        value = self.nucleus_name_input.value
        valid = self.nucleus_collection.valid_new_name(value)
        if valid:
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
        self.show_tree_view = self.tree_view_checkbox.value
        self.colorize_raster = self.colorize_checkbox.value
        self.show_max_intensities = self.max_intensity_checkbox.value
        self.selected_layer = self.layers_slider.value
        self.selected_timestamp_index = self.timestamp_input.value
        if change['new'] != change['old']:
            self.redraw() 

    def go_next(self, button):
        (prv, nxt) = self.previous_next()
        if nxt is None:
            self.info.value = "No next timestamp: " + repr(self.current_timestamp_id())
            return
        self.selected_timestamp_index = self.timestamp_collection.index_of_id(nxt) # nxt
        self.info.value = "next timestamp: " + repr(nxt)
        self.timestamp_input.value = self.selected_timestamp_index
        #self.redraw() -- triggered automatically

    def go_previous(self, button):
        (prv, nxt) = self.previous_next()
        if prv is None:
            self.info.value = "No previous timestamp: " + repr(self.current_timestamp_id())
            return
        self.selected_timestamp_index = self.timestamp_collection.index_of_id(prv) # prv
        self.info.value = "previous timestamp: " + repr(prv)
        self.timestamp_input.value = self.selected_timestamp_index
        #self.redraw() -- triggered automatically

    def previous_next(self):
        tsc = self.timestamp_collection
        tsid = self.current_timestamp_id()
        return tsc.previous_next(tsid)

    def check_highlights(self):
        if self.show_tree_view:
            #self.time_tree.set_highlights(self.current_timestamp_id(), self.selected_nucleus_id)
            #self.time_tree.make_frame()
            # just redraw the time tree
            self.time_tree.make_widget()

    def set_ids(self, timestamp_id=None, nucleus_id=None):
        if nucleus_id is not None:
            self.set_nucleus_id(nucleus_id)
        if timestamp_id is not None:
            self.selected_timestamp_index = self.timestamp_collection.index_of_id(timestamp_id)
            self.timestamp_input.value = self.selected_timestamp_index
            #self.redraw() -- triggered automatically
        else:
            self.redraw()

    def redraw(self):
        self.calculate_stats()
        #tsc = self.timestamp_collection
        self.show_nucleus_selection()
        (prv, nxt) = self.previous_next()
        #self.prev_button.disabled = (prv is None)
        #self.next_button.disabled = (nxt is None)
        ts = self.timestamp()
        if self.raster_display is not None:
            rimage = self.raster_image(ts)
            self.raster_display.set_image(rimage)
        self.check_highlights()
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
        max_intensity = self.max_intensity_checkbox.value
        layer = self.layers_slider.value
        blur = self.blur_checkbox.value
        colorize = self.colorize_checkbox.value
        return ts.raster_slice_with_boundary(layer, max_intensity, blur=blur, colorize=colorize)

class TimeTreeWidget:

    def __init__(self, nucleus_collection, timestamp_collection, controller=None):
        self.nucleus_collection = nucleus_collection
        self.timestamp_collection = timestamp_collection
        self.controller = controller
        self.widget = None
        self.frame = None
        self.min_ts_index = 0
        self.max_ts_index = timestamp_collection.max_index()
        self.min_position = 0
        self.max_position = len(nucleus_collection.nuclei) - 1
        self.position_to_nuclei = None
        self.nuclei_range_order = None
        self.assembly = None

    def sort_nuclei_in_range(self):
        position_to_nuclei_in_range = self.nucleus_collection.get_position_to_nuclei_in_range(self.min_ts_index, self.max_ts_index)
        positions = sorted(position_to_nuclei_in_range.keys())
        range_positions_to_nuclei = {}
        for (range_index, position) in enumerate(positions):
            nucleus = position_to_nuclei_in_range[position]
            range_positions_to_nuclei[range_index] = nucleus
            nucleus.range_position = range_index
        return range_positions_to_nuclei

    last_range_nucleus = None

    def nucleus_at_range_position(self, int_y):
        result = self.nuclei_in_range.get(int_y)
        self.last_range_nucleus = (int_y, result)
        return result

    def compute_nuclei_range(self):
        self.nuclei_in_range = self.sort_nuclei_in_range()
        self.nuclei_in_range_ids = set(x.identifier for x in self.nuclei_in_range.values())

    making_widget = False

    def make_widget(self, width=None, height=None):
        from jp_doodle import dual_canvas
        from jp_doodle import bounded_value_slider
        try:
            self.making_widget = True
            self.compute_nuclei_range()
            ts = self.controller.timestamp()
            if width is None:
                width = self.width
                height = self.height
            self.width = width
            self.height = height
            self.widget = dual_canvas.DualCanvasWidget(width=width, height=height)
            self.timestamp_index = 0
            self.nucleus_position = 0
            if ts is not None:
                self.timestamp_index = self.timestamp_collection.get_index(ts)
                # check boundaries
                reset = False
                if self.min_ts_index > self.timestamp_index:
                    self.min_ts_index = self.timestamp_index
                    reset = True
                if self.max_ts_index < self.timestamp_index:
                    self.max_ts_index = self.timestamp_index
                    reset = True
                self.reset_timestamps = (ts, self.min_ts_index, self.timestamp_index, self.max_ts_index) # debug
                if reset:
                    self.reset_position_extrema()
            else:
                raise ValueError("no timestamp?") # debug
                pass
            self.controller.info.value = "timestamps: <br>" + repr((self.min_ts_index, self.timestamp_index, self.max_ts_index))
            #self.make_frame()
            self.ts_slider = bounded_value_slider.BoundedValueSlider(
                length=self.width,
                minimum=0,
                maximum=self.timestamp_collection.max_index(),
                initial=self.timestamp_index,
                on_stop=self.ts_slide, 
                integral=True, 
                aspect_ratio=0.05,
                forbidden="red",
                border=20,
                call_on_init=False,
            )
            self.ts_slider.set_values(low=self.min_ts_index, high=self.max_ts_index)
            #self.nuclei_in_range = self.sort_nuclei_in_range(self.min_ts_index, self.max_ts_index)
            #self.nuclei_in_range_ids = set(x.identifier for x in self.nuclei_in_range.values())
            nuclei_range_positions = list(self.nuclei_in_range.keys())
            initial_position = 0
            nucleus = self.controller.get_nucleus()
            if (nucleus is not None) and (nucleus.identifier in self.nuclei_in_range_ids):
                initial_position = nucleus.range_position
            low_position = 0
            high_position = 1
            if nuclei_range_positions:
                low_position = min(nuclei_range_positions)
                high_position = max(nuclei_range_positions)
                if low_position == high_position:
                    high_position = low_position + 1
            self.position_slider = bounded_value_slider.BoundedValueSlider(
                length=self.width,
                minimum=low_position,
                maximum=high_position,
                initial=initial_position,
                on_stop=self.position_slide, 
                integral=True, 
                horizontal=False,
                aspect_ratio=0.04,
                border=30,
            )
            if self.min_position < low_position:
                self.min_position = low_position
            if self.min_position > high_position:
                self.min_position = high_position
            if self.max_position < self.min_position:
                self.max_position = self.min_position
            if self.max_position > high_position:
                self.max_position = high_position
            if self.max_position == self.min_position:
                self.max_position = self.min_position + 1  # avoid div 0
            self.position_slider.set_values(low=self.min_position, high=self.max_position)
            self.make_frame()
            widget = self.widget
            position_slider = self.position_slider
            ts_slider = self.ts_slider
            display_debug = False
            if display_debug:  # DEBUGGING
                widget = self.widget.debugging_display()
                position_slider = self.position_slider.debugging_display()
                ts_slider = self.ts_slider.debugging_display()
            top_assembly = widgets.HBox([
                position_slider,
                widget,
            ])
            assembly_children = [
                top_assembly,
                ts_slider,
            ]
            if self.assembly is None:
                self.assembly = widgets.VBox(assembly_children)
            else:
                self.assembly.children = assembly_children
            return self.assembly
        finally:
            self.making_widget = False

    def make_frame(self):
        nucleus = self.controller.get_nucleus()
        ts = self.controller.timestamp()
        widget = self.widget
        widget.reset_canvas()
        #fwidth = self.timestamp_collection.width()
        #nuclei_range_positions = list(self.nuclei_in_range.keys())
        #fheight = max(nuclei_range_positions)
        frame_minx = self.min_ts_index - 1
        frame_maxx = self.max_ts_index + 1
        frame_miny = self.min_position - 1
        frame_maxy = self.max_position + 1
        #fheight = frame_maxy - frame_miny
        #fwidth = frame_maxx - frame_minx
        margin = 0.2
        self.frame = self.widget.frame_region(
            minx=0, miny=0, maxx=self.width, maxy=self.height, 
            frame_minx=frame_minx - margin, 
            frame_miny=frame_miny - margin, 
            frame_maxx=frame_maxx + margin, 
            frame_maxy=frame_maxy + margin,
            )
        with self.widget.delay_redraw():
            max_height = self.nucleus_collection.height() + 2
            max_width = self.timestamp_collection.width() + 2
            self.timetamp_highlight = self.frame.frame_rect(
                self.timestamp_index, -1, w=0.9, h=max_height, fill=False, color="black", name=True)
            ncolor = INVISIBLE
            if self.controller:
                if nucleus and (nucleus.range_position is not None):
                    ncolor = "red"
                    self.nucleus_position = nucleus.range_position
            self.nucleus_highlight = self.frame.frame_rect(-1, self.nucleus_position, w=max_width, h=0.9, fill=False, color=ncolor, name=True)
            self.nucleus_collection.draw_nuclei(self.frame, in_range=True)
            info = "Time Tree: " + repr((nucleus, ts, self.min_ts_index, self.timestamp_index, self.max_ts_index))
            self.text = self.widget.text(0, 0, info, name=True, events=False)
            # invisible event rectangle
            self.event_rect = self.frame.frame_rect(-1, -1, max_width, max_height, color="rgba(0,0,0,0)", name=True)
            self.event_rect.on("mousemove", self.mouse_move)
            self.event_rect.on("click", self.click)
            #widget.fit()

    def position_slide(self, positions):
        if self.making_widget:
            return  # ignore callbacks before
        old = (self.min_position, self.max_position)
        self.min_position = positions["LOW"]
        self.max_position = positions["HIGH"]
        new = (self.min_position, self.max_position)
        curr = positions["CURRENT"]
        nucleus = self.nucleus_at_range_position(curr)
        if nucleus is not None:
            old_nucleus = self.controller.get_nucleus()
            if nucleus is not old_nucleus:
                self.controller.set_ids(nucleus_id=nucleus.identifier)
            elif old != new:
                self.make_widget()
        else:
            if old != new:
                self.make_widget()

    def reset_position_extrema(self):
        # reset the position range
        self.compute_nuclei_range()
        self.min_position = 0
        self.max_position = 0
        nuclei_range_positions = list(self.nuclei_in_range.keys())
        if nuclei_range_positions:
            self.max_position = max(nuclei_range_positions)

    def ts_slide(self, positions):
        old = (self.min_ts_index, self.max_ts_index)
        self.min_ts_index = positions["LOW"]
        self.max_ts_index = positions["HIGH"]
        new = (self.min_ts_index, self.max_ts_index)
        if (old != new):
            self.reset_position_extrema()
        old_ts = self.timestamp_index
        self.timestamp_index = positions["CURRENT"]
        #self.make_frame()
        if (old_ts != self.timestamp_index):
            self.controller.set_ids(timestamp_id=self.timestamp_index)
        else:
            #self.controller.redraw()
            if old != new:
                #print("old", old, "new", new)
                self.make_widget()

    def set_highlights(self, timestamp_id, nucleus_id):
        self.compute_nuclei_range()
        ids = (timestamp_id, nucleus_id)
        self.text.change(text="set_highlignts: " +repr(ids))
        if nucleus_id:
            nucleus = self.nucleus_collection.get_nucleus(nucleus_id)
            self.nucleus_position = nucleus.range_position
            if self.nucleus_position is not None:
                self.nucleus_highlight.change(y=self.nucleus_position, color="blue")
            else:
                self.nucleus_highlight.change(color=INVISIBLE)
        else:
            self.nucleus_highlight.change(color=INVISIBLE)
        self.timestamp_index = self.timestamp_collection.index_of_id(timestamp_id)
        self.timetamp_highlight.change(x=self.timestamp_index, color="red")

    def ids_at(self, x, y):
        timestamp_id = self.timestamp_collection.id_at_index(int(x))
        #nucleus_id = self.nucleus_collection.id_at_position(int(y))
        nucleus_id = None
        nucleus = self.nucleus_at_range_position(int(y))
        if nucleus is not None:
            nucleus_id = nucleus.identifier
        return (timestamp_id, nucleus_id)

    def click(self, event):
        self.compute_nuclei_range()
        position = event['model_location']
        x = int(position["x"])
        y = int(position["y"])
        (timestamp_id, nucleus_id) = ids = self.ids_at(x,y)
        if self.controller is not None:
            self.text.change(text="set ids: " + repr(ids))
            self.controller.set_ids(timestamp_id, nucleus_id)

    def mouse_move(self, event):
        position = event['model_location']
        x = int(position["x"])
        y = int(position["y"])
        ids = self.ids_at(x,y)
        self.text.change(text=repr(ids))
