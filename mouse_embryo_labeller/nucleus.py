
import numpy as np
import ipywidgets as widgets

class Nucleus:

    """
    Information about a tracked nucleus.
    """
    def __init__(self, identifier, color, parent_id=None):
        self.identifier = identifier
        self.parent_id = parent_id
        self.color = np.zeros((3,), dtype=np.int)
        self.color[:] = color

    def html_color(self):
        return "rgb(%s,%s,%s)" % tuple(self.color)

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