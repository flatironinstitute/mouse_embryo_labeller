"""
Miscellaneous functionality.
"""

import os
from PIL import Image, ImageSequence
import numpy as np
from mouse_embryo_labeller import timestamp
from mouse_embryo_labeller import timestamp_collection
from mouse_embryo_labeller import nucleus_collection

EXAMPLE_FOLDER = "../example_data/"

def preprocess_sample_data(
    stride=4, 
    destination=EXAMPLE_FOLDER,
    labels_pattern="/Users/awatters/misc/lisa2/mouse-embryo-nuclei/H9/H9_%s_Labels.tiff",
    intensities_pattern="/Users/awatters/misc/lisa2/mouse-embryo-nuclei/H9OriginalIntensityImages/klbOut_CH1_%06d.klb",
    sanity_limit=10000,
    ):
    "preprocess the example H9 data, subsample j and k dimensions by stride"
    try:
        import pyklb
    except ImportError:
        print ("Please install pyklb or fix any install problems.")
        print ("Install problem fix at: https://github.com/bhoeckendorf/pyklb/issues/3")
        raise
    tsc = timestamp_collection.TimestampCollection()
    ts_pattern = destination + "/ts%s"
    json_pattern = ts_pattern + ".json"
    npz_pattern = ts_pattern + ".npz"
    for i in range(sanity_limit):
        labels_path = labels_pattern % i
        intensities_path = intensities_pattern % i
        le = os.path.exists(labels_path)
        ie = os.path.exists(intensities_path)
        print()
        print("labels:", labels_path)
        print("intensities", intensities_path)
        if not (le and ie):
            if le:
                raise ValueError("labels without intensities found")
            if ie:
                raise ValueError("intensities without labels found")
            print ("files not found... finishing")
            break
        img = pyklb.readfull(intensities_path)
        labels = load_tiff_array(labels_path)
        assert img.shape == labels.shape, "bad shapes " + repr((img.shape, labels.shape))
        # truncate j and k
        s_img = img[:, ::stride, ::stride]
        s_labels = labels[:, ::stride, ::stride]
        ts = timestamp.Timestamp(i)
        ts.add_source_arrays(s_img, s_labels)
        tsc.add_timestamp(ts)
        # store and discard the timestamp arrays to prevent memory leakage.
        ts.get_truncated_arrays(test_array=None)
        json_path = json_pattern % ts.identifier
        npz_path = npz_pattern % ts.identifier
        ts.save_mapping(json_path)
        ts.save_truncated_arrays(npz_path, discard=True)
        # store the manifest for this timestamp
        ts.manifest = {
                "identity": ts.identifier,
                "json_path": timestamp_collection.filename_only(json_path),
                "npz_path": timestamp_collection.filename_only(npz_path),
            }
    #print("now truncating all time slices...")
    #tsc.truncate_all()
    #ts_pattern = destination + "/ts%s"
    print("storing timestamps with pattern", repr(ts_pattern))
    tsc.store_all(ts_pattern)
    print("Creating empty nucleus collection...")
    nc = nucleus_collection.NucleusCollection()
    nc.save_json(destination)
    print("done.")

def get_example_nucleus_collection(from_folder=EXAMPLE_FOLDER):
    return nucleus_collection.collection_from_json(from_folder)

def get_example_timestamp_collection(from_folder=EXAMPLE_FOLDER, nucleus_collection=None):
    return timestamp_collection.load_preprocessed_timestamps(from_folder, nucleus_collection)

def load_tiff_array(tiff_path):
    im = Image.open(tiff_path)
    L = []
    for i, page in enumerate(ImageSequence.Iterator(im)):
        a = np.array(page)
        # only the first channel
        a = a[:, :, 0]
        # flip j and k
        a = a.transpose()
        L.append(a)
    All = np.zeros( (len(L),) + L[0].shape, dtype=np.int)
    for (i, aa) in enumerate(L):
        All[i] = aa
    return All

if __name__=="__main__":
    preprocess_sample_data()
