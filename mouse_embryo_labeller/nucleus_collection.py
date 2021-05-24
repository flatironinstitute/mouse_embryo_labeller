
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
        self.selected_id = None
        self.controller = None
        for n in nuclei:
            self.add_nucleus(n)

    def valid_new_name(self, name):
        if name:
            return name not in self.id_to_nucleus
        return False

    def set_selected_nucleus(self, id):
        if id is not None:
            assert id in self.id_to_nucleus, "no such id in collection: " + repr(id)
        self.selected_id = id

    def get_selected_nucleus(self):
        id = self.selected_id
        if id is None:
            return None
        return self.get_nucleus(id)

    def reparent_selected_nucleus(self, new_parent_id):
        selected = self.get_selected_nucleus()
        assert selected is not None
        selected.reparent(new_parent_id)
        to_folder = self.controller.folder
        self.save_json(to_folder)

    def set_controller(self, controller):
        self.controller = controller

    def colors_in_use(self):
        return set(tuple(n.color) for n in self.nuclei)

    def save_json(self, to_folder, to_filename=None):
        json_ob = [n.json_description() for n in self.nuclei]
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
                    element.info.html("nucleii: " + ln1);
                }
            };
        """, height=height, width=width)
        self.widget = widget
        self.set_widget_options(callback, widget=widget);
        return widget

    none_option = "NONE"

    def set_widget_options(self, callback, selected=None, widget=None):
        if widget is None:
            widget = self.widget
        assert widget is not None, "no widget initialized"
        nucleus = self.get_selected_nucleus()
        nulloption = '<option value="NONE">NONE</option>'
        options = [n.html_option() for n in self.nuclei]
        identifiers = [n.identifier for n in self.nuclei]
        options = [nulloption] + options
        widget.element.set_embryo_options(options, callback, selected)
        none_option = self.none_option
        drop_down_option = [none_option] + identifiers
        self.join_dropdown.options = drop_down_option
        self.join_dropdown.value = none_option
        self.join_button.disabled = True
        self.reparent_dropdown.options = drop_down_option
        self.reparent_dropdown.value = none_option
        self.reparent_button.disabled = (nucleus is None)

    def join_click(self, button):
        join_from_id = self.join_dropdown.value
        join_to_id = self.selected_id
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
        self.controller.redraw()
        self.controller.info.value = "reparented " + repr(new_parent)

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

    def join_select(self, change):
        button = self.join_button
        new = change["new"]
        if new != self.none_option:
            if self.selected_id is not None:
                button.disabled = False
                return
            else:
                #print("no selection")
                pass
        else:
            #print("bad change: ", new)
            pass
        # default
        button.disabled = True

    def reparent_select(self, change):
        pass # do nothing until user presses the reparent button

    def join_assembly(self):
        self.join_info = widgets.HTML(value="Subsume nucleus")
        self.join_dropdown = widgets.Dropdown(
            options=['NONE'],
            value='NONE',
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


    