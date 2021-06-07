
import numpy as np
import ipywidgets as widgets
from mouse_embryo_labeller.color_list import rgbhtml

class Nucleus:

    """
    Information about a tracked nucleus.
    """
    def __init__(self, identifier, color, parent_id=None):
        self.identifier = identifier
        self.parent_id = parent_id
        self.color = np.zeros((3,), dtype=np.int)
        self.color[:] = color
        self.reset_stats()

    def __repr__(self):
        return "N(%s)" % repr(self.identifier)

    def reset_stats(self):
        self.children = None
        self.width = None
        self.position = None
        self.last_descendent_position = None
        self.timestamp_indices = set()

    def draw_rectangles(self, on_frame, position=None):
        color = self.html_color()
        if position is None:
            position = self.position
        #for index in self.timestamp_indices:
        #    on_frame.frame_rect(x=index, y=position+0.25, h=0.5, w=0.8, color=color)
        indices = sorted(self.timestamp_indices)
        if len(indices) == 0:
            return
        start_index = None
        last_index = None
        def emit_rectangle():
            w = last_index - start_index + 0.8
            on_frame.frame_rect(x=start_index, y=position+0.25, h=0.5, w=w, color=color)
        for index in indices:
            if start_index is None:
                start_index = index
                last_index = start_index
            else:
                if index > last_index + 1:
                    emit_rectangle()
                    start_index = index
                    last_index = start_index
                else:
                    last_index = index
        if start_index is not None:
            emit_rectangle()

    def intersects_index_range(self, low_index, high_index):
        for index in self.timestamp_indices:
            if low_index <= index <= high_index:
                return True
        return False

    def min_index(self):
        result = 0
        if self.timestamp_indices:
            result = min(self.timestamp_indices)
        return result

    def draw_label(self, on_frame):
        color = self.html_color()
        position = self.position
        index = self.min_index()
        on_frame.text(index-0.1, position+0.5, str(self.identifier) + " ", align="right", valign="center", color=color, background="#eee")

    def draw_link(self, on_frame, nucleus_collection):
        parent_id = self.parent_id
        if parent_id is not None:
            parent = nucleus_collection.get_nucleus(parent_id)
            color = self.html_color()
            position = self.position
            index = self.min_index()
            p_position = parent.position
            p_index = parent.min_index() #min(parent.timestamp_indices)
            on_frame.line(index+0.5, position+0.5, p_index+0.5, p_position+0.5, color=color, lineWidth=5)

    def add_timestamp_index(self, index):
        self.timestamp_indices.add(index)

    def html_color(self):
        return "rgb(%s,%s,%s)" % tuple(self.color)

    def reparent(self, new_parent_id):
        self.parent_id = new_parent_id

    def html_info(self, nucleus_collection):
        i = self.identifier
        pid = self.parent_id
        c = self.html_color()
        prefix =  '<span style="background-color:%s">NUCLEUS</span> %s ' % (c, i)
        suffix = ""
        if pid:
            p = nucleus_collection.get_nucleus(pid, check=False)
            if p is None:
                suffix = "BAD PARENT " + repr(pid)
            else:
                cp = p.html_color()
                suffix = '<span style="background-color:%s">PARENT</span> %s ' % (cp, pid)
        return prefix + suffix

    def html_option(self, selected=False):
        s = ""
        if selected:
            s = " selected "
        c = self.html_color()
        i = self.identifier
        p = self.parent_id
        txt = i
        if p:
            txt = "%s&lt;%s" % (i, p)
        return '<option value="%s" %s style="background-color:%s">%s</option>' % (
            i, s, c, txt,
        )

    def json_description(self):
        return dict(
            identifier=self.identifier,
            color=self.color.tolist(),
            #html_color: self.html_color(),
            parent_id=self.parent_id,
        )

def nucleus_from_json(json_ob):
    return Nucleus(
        identifier=json_ob["identifier"],
        color=json_ob["color"],
        parent_id=json_ob["parent_id"]
        )