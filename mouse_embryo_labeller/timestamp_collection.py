
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

    def get_timestamp(self, identifier):
        return self.id_to_timestamp[identifier]

    def first_id(self):
        return self.id_sequence[0]

    def add_timestamp(self, ts):
        identifier = ts.identifier
        self.id_to_timestamp[identifier] = ts
        self.id_sequence.append(identifier)

    def previous_next(self, ts_id):
        prv = nxt = None
        seq = self.id_sequence
        i = seq.index(ts_id)
        if i > 0:
            prv = seq[i - 1]
        if i < len(seq) - 1:
            nxt = seq[i + 1]
        return (prv, nxt)

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
            
    def store_all(self, pattern, verbose=True):
        manifest = []
        json_pattern = pattern + ".json"
        npz_pattern = pattern + ".npz"
        for (identity, ts) in self.id_to_timestamp.items():
            json_path = json_pattern % identity
            npz_path = npz_pattern % identity
            entry = {
                "identity": identity,
                "json_path": filename_only(json_path),
                "npz_path": filename_only(npz_path),
            }
            manifest.append(entry)
            if verbose:
                print ("storing ts", (json_path, npz_path))
            ts.save_mapping(json_path)
            ts.save_truncated_arrays(npz_path)
        manifest_path = manifest_file_path(pattern)
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
        ts.load_truncated_arrays(expand(description["npz_path"]))
        ts.load_mapping(expand(description["json_path"]), nucleus_collection)
        result.add_timestamp(ts)
    return result

def filename_only(path):
    return os.path.split(path)[-1]
