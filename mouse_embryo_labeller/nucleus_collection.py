
import numpy
import ipywidgets as widgets
import jp_proxy_widget
import json
import os
from mouse_embryo_labeller import nucleus, color_list
import ipywidgets as widgets

DEFAULT_FILENAME = "nuclei.json"

def collection_from_json(from_folder, from_filename=DEFAULT_FILENAME):
    from_path = os.path.join(from_folder, from_filename)
    f = open(from_path)
    json_ob = json.load(f)
    f.close()
    nuclei = [nucleus.nucleus_from_json(x) for x in json_ob]
    result = NucleusCollection(filename=from_filename, nuclei=nuclei)
    return result


class NucleusCollection: 

    """'
    Container for collection of nuclei tracked in the visualization.
    """
    def __init__(self, filename=DEFAULT_FILENAME, nuclei=()):
        self.filename = filename
        self.nuclei =  []
        self.id_to_nucleus = {}
        self.controller = None
        self.last_position = None
        for n in nuclei:
            self.add_nucleus(n)
        self.reset_stats()

    def sorted_nuclei(self, id2n=None):
        "nuclei sorted by increasing id"
        if id2n is None:
            id2n = self.id_to_nucleus
        ids = sorted(id2n.keys())
        return [id2n[ident] for ident in ids]

    def get_position_to_nuclei_in_range(self, low_timestamp_index, high_timestamp_index):
        result = {}
        for n in self.nuclei:
            position = n.position
            assert position is not None, "no position for nucleus: %s" % n
            if n.intersects_index_range(low_timestamp_index, high_timestamp_index):
                result[position] = n
        return result

    def reset_stats(self):
        for n in self.nuclei:
            n.reset_stats()

    def id_at_position(self, position):
        result = None
        ns = self.nuclei
        if ns:
            result = ns[0].identifier
            for n in ns:
                if n.position == position:
                    result = n.identifier
        return result

    def height(self):
        return len(self.nuclei)

    def assign_children(self):
        for n in self.nuclei:
            n.children = []
        for n in self.nuclei:
            pid = n.parent_id
            if pid is not None:
                p = self.id_to_nucleus.get(pid)
                if p is not None:
                    p.children.append(n.identifier)
                else:
                    print("Warning: invalid parent id removed: " + repr((n.identifier, pid)))
                    n.parent_id = None  # patch the data
                
    def assign_widths(self):
        # children must be assigned
        # also test for circularity
        unassigned = set(self.id_to_nucleus.keys())
        for n in self.nuclei:
            n.width = None
        while unassigned:
            progress = False
            for nid in list(unassigned):
                ok = True
                width = 1
                node = self.id_to_nucleus[nid]
                for cid in node.children:
                    c = self.id_to_nucleus.get(cid)
                    if c.width is not None:
                        width = width + c.width
                    else:
                        ok = False
                        break
                if ok:
                    node.width = width
                    unassigned.remove(node.identifier)
                    progress = True
            assert progress, "Circular node parent relationship: " + repr(list(unassigned))

    def assign_positions(self, check_children=False):
        if check_children:
            self.assign_children()
            self.assign_widths()
        # after assign_width
        cursor = 0
        for n in reversed(self.sorted_nuclei()):
            if n.parent_id is None:
                cursor = self.assign_position(n, cursor)
        self.last_position = cursor

    def assign_position(self, nucleus, cursor):
        children = nucleus.children
        # put the nucleus between multiple children
        if children:
            child0 = self.id_to_nucleus[children[0]]
            cursor = self.assign_position(child0, cursor)
        nucleus.position = cursor + 1
        cursor = nucleus.position
        for cid in children[1:]:
            child = self.id_to_nucleus[cid]
            cursor = self.assign_position(child, cursor)
        nucleus.last_descendent_position = cursor
        return cursor

    def draw_nuclei(self, on_frame, labels=False, in_range=False):
        for n in self.nuclei:
            n.draw_rectangles(on_frame, in_range=in_range)
        for n in self.nuclei:
            n.draw_link(on_frame, self, in_range=in_range)
        if labels:
            for n in self.nuclei:
                n.draw_label(on_frame, in_range=in_range)

    def valid_new_name(self, name):
        if name:
            return name not in self.id_to_nucleus
        return False

    #def set_selected_nucleus(self, id):  -- not used
    #    if id is not None:
    #        assert id in self.id_to_nucleus, "no such id in collection: " + repr(id)
    #    #self.selected_id = id
    #    self.controller.set_nucleus_id(id)
    #    self.check_buttons()

    def get_selected_nucleus(self):
        id = self.controller.selected_nucleus_id
        if id is None:
            return None
        return self.get_nucleus(id)

    def reparent_selected_nucleus(self, new_parent_id):
        if new_parent_id is not None:
            assert new_parent_id in self.id_to_nucleus, "bad parent id: " + repr(new_parent_id)
        selected = self.get_selected_nucleus()
        assert selected is not None
        selected.reparent(new_parent_id)
        to_folder = self.controller.folder
        self.save_json(to_folder)

    def set_controller(self, controller):
        self.controller = controller

    def colors_in_use(self):
        return set(tuple(n.color) for n in self.nuclei)

    save_json_folder = None

    def save_json(self, to_folder=None, to_filename=None):
        json_ob = [n.json_description() for n in self.nuclei]
        if to_folder is None:
            to_folder = self.save_json_folder
        assert to_folder is not None, "to folder is required"
        self.save_json_folder = to_folder
        if to_filename is None:
            to_filename = self.filename
        to_path = os.path.join(to_folder, to_filename)
        self.manifest_path = to_path
        f = open(to_path, "w")
        json.dump(json_ob, f, indent=2)
        f.close()

    def add_nucleus(self, nucleus):
        id2n = self.id_to_nucleus
        identifier = nucleus.identifier
        assert identifier not in id2n, "duplicate nucleus id: " + repr(identifier)
        id2n[identifier] = nucleus
        self.nuclei.append(nucleus)

    def forget_nucleus_id(self, nid, save_folder):
        id2n = self.id_to_nucleus
        ns = self.nuclei
        # unparent any nuclei with this nucleus as a parent
        for n in ns:
            if n.parent_id == nid:
                n.parent_id = None
        delete_n = id2n[nid]
        self.id_to_nucleus = {identifier: nucleus for (identifier, nucleus) in id2n.items() if identifier != nid}
        self.nuclei = [n for n in ns if n is not delete_n]
        self.save_json(save_folder)

    def get_nucleus(self, identifier, check=True):
        n = self.id_to_nucleus.get(identifier)
        if n is None and check:
            assert n is not None, "No nucleus with identifier: " + repr(identifier)
        return n

    def get_or_make_nucleus(self, identifier):
        result = self.get_nucleus(identifier, check=False)
        if result is None:
            index = len(self.id_to_nucleus)
            color = color_list.indexed_color(index)  # maybe should look for color not in use. xxx
            result = nucleus.Nucleus(identifier, color)
            self.add_nucleus(result)
        return result

    def create_widget(self, height=500, width=100, callback=None):
        widget = jp_proxy_widget.JSProxyWidget()
        widget.js_init("""
            element.empty();
            element.embryo_callback = null
            element.info = $("<div>Nucleus ids</div>").appendTo(element);
            element.embryo_select = $('<select/>').appendTo(element);
            element.embryo_select.height(height);
            element.embryo_select.width(width);
            element.embryo_select.change(function(e) {
                var selected = element.embryo_select.val();
                element.info.html("Selected: " + selected);
                if (element.embryo_callback) {
                    element.embryo_callback(selected);
                }
            });
            element.set_embryo_options = function(options, callback, selected) {
                var select = element.embryo_select;
                if (callback) {
                    element.embryo_callback = callback;
                }
                element.embryo_select.empty();
                var ln = options.length;
                var ln1 = ln - 1;
                select.attr('size', ln)
                for (var i=0; i<ln; i++) {
                    var option = options[i];
                    $(option).appendTo(select)
                }
                if (selected) {
                    element.info.html("at " + selected + " of " + ln1);
                } else {
                    element.info.html("nuclei: " + ln1);
                }
            };
        """, height=height, width=width)
        self.widget = widget
        self.widget_height = height
        self.widget_width = width
        self.set_widget_options(callback, widget=widget);
        return widget

    none_option = "NONE"

    def set_widget_options(self, callback, selected=None, widget=None, id_to_nucleus=None):
        if widget is None:
            widget = self.widget
        if id_to_nucleus is None:
            id_to_nucleus = self.controller.id_to_visible_nuclei()
        assert widget is not None, "no widget initialized"
        nucleus = self.get_selected_nucleus()
        nulloption = '<option value="NONE">NONE</option>'
        nuclei = self.sorted_nuclei(id2n=id_to_nucleus)
        options = [n.html_option() for n in nuclei]
        identifiers = [n.identifier for n in nuclei]
        options = [nulloption] + options
        widget.element.set_embryo_options(options, callback, selected)
        none_option = self.none_option
        drop_down_option = [none_option] + identifiers
        self.join_dropdown.options = drop_down_option
        self.join_dropdown.value = none_option
        self.join_button.disabled = True
        self.reparent_dropdown.options = drop_down_option
        self.reparent_dropdown.value = none_option
        self.split_dropdown.options = drop_down_option
        self.split_dropdown.value = none_option
        self.reparent_button.disabled = (nucleus is None)
        self.split_button.disabled = (nucleus is None)

    def join_click(self, button):
        join_from_id = self.join_dropdown.value
        join_to_id = self.controller.selected_nucleus_id
        valid_ids = self.id_to_nucleus
        assert (join_from_id in valid_ids) and (join_to_id in valid_ids), "Invalid ids: " + repr([join_from_id, join_to_id])
        self.controller.relabel_and_delete(join_from_id, join_to_id)

    def reparent_click(self, button):
        new_parent = self.reparent_dropdown.value
        if new_parent == self.none_option:
            new_parent = None
        else:
            assert new_parent in self.id_to_nucleus, "no such nucleus identifier: " + repr(new_parent)
        self.reparent_selected_nucleus(new_parent)
        # update display
        self.set_widget_options(callback=None, selected=None)
        #self.controller.redraw()
        self.controller.make_widget()
        child = self.controller.selected_nucleus_id
        self.controller.info.value = repr(child) + "<br> reparented " + repr(new_parent)

    def reparent_assembly(self):
        self.reparent_info = widgets.HTML(value="Change Parent")
        self.reparent_dropdown = widgets.Dropdown(
            options=['NONE'],
            value='NONE',
            #description='new parent',
            disabled=False,
            layout={'width': "150px"}
            )
        self.reparent_dropdown.observe(self.reparent_select, names="value")
        self.reparent_button = widgets.Button(description="Reparent", disabled=True)
        self.reparent_button.on_click(self.reparent_click)
        assembly = widgets.VBox([
            self.reparent_info,
            self.reparent_dropdown,
            self.reparent_button,
        ], layout=widgets.Layout(border='solid'))
        return assembly

    def split_assembly(self):
        self.split_info = widgets.HTML(value="Split Right")
        self.split_dropdown = widgets.Dropdown(
            options=['NONE'],
            value='NONE',
            #description='new parent',
            disabled=False,
            layout={'width': "150px"}
            )
        self.split_dropdown.observe(self.split_select, names="value")
        self.split_button = widgets.Button(description="Split", disabled=True)
        self.split_button.on_click(self.split_click)
        assembly = widgets.VBox([
            self.split_info,
            self.split_dropdown,
            self.split_button,
        ], layout=widgets.Layout(border='solid'))
        return assembly

    def split_select(self, change):
        # do nothing -- user must click "split" button
        pass

    def split_click(self, button):
        split_id = self.split_dropdown.value
        if split_id == self.none_option:
            split_id = None
        self.controller.split_right(split_id)

    def join_select(self, change):
        none_option = self.none_option
        selected_id = self.controller.selected_nucleus_id
        self.join_button.disabled = ((change["new"] == none_option) and (selected_id is not None))

    def reparent_select(self, change):
        #self.check_buttons()
        pass # do nothing -- the reparent button should only be enabled if selected id is set

    def check_buttons(self):
        selected = self.controller.selected_nucleus_id
        if selected is None:
            self.reparent_button.disabled = True
            self.join_button.disabled = True
        else:
            none_option = self.none_option
            # allow reparent to "none"
            self.reparent_button.disabled = False
            self.join_button.disabled = (self.join_dropdown.value == none_option)

    def join_assembly(self):
        self.join_info = widgets.HTML(value="Subsume nucleus")
        none_option = self.none_option
        self.join_dropdown = widgets.Dropdown(
            options=[none_option],
            value=none_option,
            #description='new parent',
            disabled=False,
            layout={'width': "150px"}
            )
        self.join_dropdown.observe(self.join_select, names="value")
        self.join_button = widgets.Button(description="Join", disabled=True)
        self.join_button.on_click(self.join_click)
        assembly = widgets.VBox([
            self.join_info,
            self.join_dropdown,
            self.join_button,
        ], layout=widgets.Layout(border='solid'))
        return assembly


    