
import numpy as np
import json
import os
from mouse_embryo_labeller import timestamp

class TimestampCollection:

    """
    Container for timestamps in the microscopy sequence.
    """
    def __init__(self):
        self.id_to_timestamp = {}
        self.id_sequence = []

    def reset_stats(self):
        pass  # placeholder -- nothing to be done here yet.

    def width(self):
        return len(self.id_sequence)

    def get_index(self, ts_id):
        return self.id_sequence.index(ts_id)

    def max_index(self):
        return len(self.id_sequence) - 1

    def id_at_index(self, index):
        s = self.id_sequence
        index = max(0, index)
        index = min(len(s)-1, index)
        return s[index]

    def assign_indices(self):
        for (index, id) in enumerate(self.id_sequence):
            ts = self.id_to_timestamp[id]
            ts.assign_index(index)

    def get_timestamp(self, identifier):
        return self.id_to_timestamp[identifier]

    def get_indexed_timestamp(self, index):
        identifier = self.id_sequence[index]
        return self.id_to_timestamp[identifier]

    def index_of_id(self, identifier):
        return self.id_sequence.index(identifier)

    def first_id(self):
        return self.id_sequence[0]

    def add_timestamp(self, ts):
        identifier = ts.identifier
        self.id_to_timestamp[identifier] = ts
        self.id_sequence.append(identifier)

    def split_right(self, from_timestamp_id, old_nucleus, new_nucleus):
        found = False
        for ts_id in self.id_sequence:
            if ts_id == from_timestamp_id:
                found = True
            if found:
                ts = self.id_to_timestamp[ts_id]
                changed = ts.relabel(old_nucleus, new_nucleus)
                if changed:
                    ts.save_mapping()
        assert found, "timestamp for split not found: " + repr(from_timestamp_id)

    def forget_nucleus(self, n, controller):
        for (ts_id, ts) in self.id_to_timestamp.items():
            found = ts.forget_nucleus(n)
            if found:
                save_path = controller.ts_assignment_path(ts_id)
                ts.save_mapping(save_path)

    def relabel(self, old_nucleus, replacement_nucleus, controller):
        for (ts_id, ts) in self.id_to_timestamp.items():
            found = ts.relabel(old_nucleus, replacement_nucleus)
            if found:
                save_path = controller.ts_assignment_path(ts_id)
                ts.save_mapping(save_path)

    def previous_next(self, ts_id):
        prv = nxt = None
        seq = self.id_sequence
        i = seq.index(ts_id)
        if i > 0:
            prv = seq[i - 1]
        if i < len(seq) - 1:
            nxt = seq[i + 1]
        return (prv, nxt)

    """
    def truncate_all(self):
        ids = self.id_sequence
        assert len(ids) > 0, "nothing to truncate."
        maxlabels = None
        for (identity, ts) in self.id_to_timestamp.items():
            r = ts.raster3d
            if maxlabels is None:
                maxlabels = r 
            else:
                maxlabels = np.maximum(r, maxlabels)
        for (identity, ts) in self.id_to_timestamp.items():
            ts.get_truncated_arrays(maxlabels)
            """
            
    def store_all(self, pattern, verbose=True):
        manifest = []
        #json_pattern = pattern + ".json"
        #npz_pattern = pattern + ".npz"
        for (identity, ts) in self.id_to_timestamp.items():
            #json_path = json_pattern % identity
            #npz_path = npz_pattern % identity
            #entry = {
            #    "identity": identity,
            #    "json_path": filename_only(json_path),
            #    "npz_path": filename_only(npz_path),
            #}
            entry = ts.manifest
            manifest.append(entry)
            #if verbose:
            #    print ("storing ts", (json_path, npz_path))
            #ts.save_mapping(json_path)
            #ts.save_truncated_arrays(npz_path)
        manifest_path = manifest_file_path(pattern)
        self.manifest_path = manifest_path
        if verbose:
            print("   Storing manifest", manifest_path, len(manifest))
        f = open(manifest_path, "w")
        json.dump(manifest, f, indent=2)
        f.close()

def manifest_file_path(from_pattern):
    return (from_pattern % ("_manifest")) + ".json"

def load_preprocessed_timestamps(from_folder, nucleus_collection, filename="ts_manifest.json"):
    def expand(fn):
        return os.path.join(from_folder, fn)
    mpath = expand(filename)
    f = open(mpath)
    manifest = json.load(f)
    f.close()
    result = TimestampCollection()
    for description in manifest:
        ts = timestamp.Timestamp(description["identity"])
        # don't hog memory -- load arrays only on demand!
        #ts.load_truncated_arrays(expand(description["npz_path"]))
        ts.save_array_path(expand(description["npz_path"]))
        #ts.load_truncated_arrays(expand(description["npz_path"]))
        ts.load_mapping(expand(description["json_path"]), nucleus_collection)
        result.add_timestamp(ts)
    return result

def filename_only(path):
    return os.path.split(path)[-1]
