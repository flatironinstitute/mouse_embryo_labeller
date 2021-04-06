
import numpy
import ipywidgets as widgets
import jp_proxy_widget
import json
import os
from mouse_embryo_labeller import nucleus

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
        for n in nuclei:
            self.add_nucleus(n)

    def save_json(self, to_folder, to_filename=None):
        json_ob = [n.json_description() for n in self.nuclei]
        if to_filename is None:
            to_filename = self.filename
        to_path = os.path.join(to_folder, to_filename)
        f = open(to_path, "w")
        json.dump(json_ob, f)
        f.close()

    def add_nucleus(self, nucleus):
        id2n = self.id_to_nucleus
        identifier = nucleus.identifier
        assert identifier not in id2n, "duplicate nucleus id: " + repr(identifier)
        id2n[identifier] = nucleus
        self.nuclei.append(nucleus)

    def get_nucleus(self, identifier):
        return self.id_to_nucleus[identifier]

    def get_nucleus(self, identifier, check=True):
        n = self.id_to_nucleus.get(identifier)
        if n is None and check:
            assert n is not None, "No nucleus with identifier: " + repr(identifier)
        return n

    def create_widget(self, height=500, width=100, callback=None):
        widget = jp_proxy_widget.JSProxyWidget()
        widget.js_init("""
            element.empty();
            element.embryo_callback = null
            element.info = $("<div>Embryo ids</div>").appendTo(element);
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
                select.attr('size', ln)
                for (var i=0; i<ln; i++) {
                    var option = options[i];
                    $(option).appendTo(select)
                }
                if (selected) {
                    element.info.html("at " + selected + " of " + ln);
                } else {
                    element.info.html("embryos: " + ln);
                }
            };
        """, height=height, width=width)
        self.widget = widget
        self.set_widget_options(callback, widget=widget);
        return widget

    def set_widget_options(self, callback, selected=None, widget=None):
        if widget is None:
            widget = self.widget
        assert widget is not None, "no widget initialized"
        options = [n.html_option() for n in self.nuclei]
        widget.element.set_embryo_options(options, callback, selected);

    