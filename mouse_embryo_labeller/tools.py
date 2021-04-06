"""
Miscellaneous functionality.
"""

import os
from PIL import Image, ImageSequence
import numpy as np
from mouse_embryo_labeller import timestamp
from mouse_embryo_labeller import timestamp_collection

def preprocess_sample_data(
    stride=4, 
    destination="../example_data/",
    labels_pattern="/Users/awatters/misc/LisaBrown/mouse-embryo-nuclei/H9/H9_%s_Labels.tiff",
    intensities_pattern="/Users/awatters/misc/LisaBrown/mouse-embryo-nuclei/H9OriginalIntensityImages/klbOut_CH1_%06d.klb",
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
    print("now truncating all time slices...")
    tsc.truncate_all()
    ts_pattern = destination + "/ts%s"
    print("storing timestamps with pattern", repr(ts_pattern))
    tsc.store_all(ts_pattern)
    print("done.")

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
