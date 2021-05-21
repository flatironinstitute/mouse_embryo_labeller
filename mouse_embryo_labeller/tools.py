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
        # This logic is moved to FileSystemHelper... not refactored here yet.
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

class FileSystemHelper:
    "Refactored common file system operations for reuse."

    def __init__(
        self,
        destination,
        ):
        self.destination = destination
        self.tsc = timestamp_collection.TimestampCollection()
        self.nc = nucleus_collection.NucleusCollection()
        self.ts_pattern = self.destination + "/ts%s"
        self.json_pattern = self.ts_pattern + ".json"
        self.npz_pattern = self.ts_pattern + ".npz"

    def load_existing_collections(self):
        self.tsc = get_example_timestamp_collection(self.destination)
        self.nc = get_example_nucleus_collection(self.destination)

    def trackname(self, tracknum):
        return "TR" + repr(tracknum)

    def load_track_data(self, tsnum2label2track):
        for tsnum in tsnum2label2track:
            label2track = tsnum2label2track[tsnum]
            ts = self.tsc.get_timestamp(tsnum)
            for (labelnum, tracknum) in label2track.items():
                trackname = self.trackname(tracknum)
                nucleus = self.nc.get_or_make_nucleus(trackname)
                ts.assign_nucleus(labelnum, nucleus)
            json_path = self.json_pattern % ts.identifier
            ts.save_mapping(json_path)
        self.nc.save_json(self.destination)

    def add_timestamp(self, identifier, img_array, labels_array):
        ts = timestamp.Timestamp(identifier)
        ts.add_source_arrays(img_array, labels_array)
        self.tsc.add_timestamp(ts)
        # store and discard the timestamp arrays to prevent memory leakage.
        ts.get_truncated_arrays(test_array=None)
        json_path = self.json_pattern % ts.identifier
        npz_path = self.npz_pattern % ts.identifier
        ts.save_mapping(json_path)
        ts.save_truncated_arrays(npz_path, discard=True)
        # store the manifest for this timestamp
        ts.manifest = {
                "identity": ts.identifier,
                "json_path": timestamp_collection.filename_only(json_path),
                "npz_path": timestamp_collection.filename_only(npz_path),
            }
        return ts

    def get_pyklb_image(self, intensities_path):
        try:
            import pyklb
        except ImportError:
            print ("Please install pyklb or fix any install problems.")
            print ("Install problem fix at: https://github.com/bhoeckendorf/pyklb/issues/3")
            raise
        return pyklb.readfull(intensities_path)

    def stored_timestamp_collection(self):
        self.tsc.store_all(self.ts_pattern)
        return self.tsc

    def stored_nucleus_collection(self):
        self.nc.save_json(self.destination)
        return self.nc

TRACKED_FOLDER = "../tracked_example"
LABELS_PATTERN = "/Users/awatters/misc/LisaBrown/tracks/fucci/fucci_%(ts_number)s_Labels.tiff"
INTENSITIES_SUBDIR = "folder_Cam_Long_%(ts_number)05d/klbOut_Cam_Long_%(ts_number)05d.klb"
INTENSITIES_PATTERN = "/Users/awatters/misc/LisaBrown/tracks/images/fucci_images_BigSet/" + INTENSITIES_SUBDIR

def preprocess_tracked_data(
    stride=4,
    destination=TRACKED_FOLDER,
    labels_pattern=LABELS_PATTERN,
    intensities_pattern=INTENSITIES_PATTERN,
    sanity_limit=100000,
    ):
    helper = FileSystemHelper(destination)
    if not os.path.isdir(destination):
        os.mkdir(destination)
    nc = helper.stored_nucleus_collection()
    print("Stored empty nucleus collection", nc.manifest_path)
    tracks_done = False
    for ts_number in range(sanity_limit):
        D = {"ts_number": ts_number}
        labels_path = labels_pattern % D
        if not os.path.exists(labels_path):
            print ("::: Done creating timestamps: no labels file", labels_path)
            tracks_done = True
            break
        # otherwise proceed...
        print("::: found timestamp labels file", labels_path)
        intensities_path = intensities_pattern % D
        assert os.path.isfile(intensities_path), repr(labels_path) + " no matching intensities " + repr(intensities_path)
        img = helper.get_pyklb_image(intensities_path)
        labels = load_tiff_array(labels_path)
        assert img.shape == labels.shape, "bad shapes " + repr((img.shape, labels.shape))
        # truncate j and k
        s_img = img[:, ::stride, ::stride]
        s_labels = labels[:, ::stride, ::stride]
        ts = helper.add_timestamp(ts_number, s_img, s_labels)
        print("::: timestamp", ts.manifest)
    assert tracks_done, "Track loop didn't terminate normally.  Too many tracks?"
    tsc = helper.stored_timestamp_collection()
    print("Stored timestamp collection", tsc.manifest_path)
    return helper

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

def read_track_table_to_tsnum2label2track(
    filepath="/Users/awatters/misc/LisaBrown/tracks/TrackAnalysis/BigTrackTable_0_138.csv"
    ):
    tsnum2label2track = {}
    f = open(filepath)
    done = False
    count = 0
    num_ts = None
    for track_number in range(100000):
        rdline = f.readline()
        if not rdline:
            done = True
            break
        ts_entries = rdline.strip().split()
        nentries = len(ts_entries)
        if num_ts is not None:
            assert num_ts == len(ts_entries), "bad line length? " + repr([track_number, nentries, num_ts])
        num_ts = nentries
        for (ts_num, label_str) in enumerate(ts_entries):
            label_num = int(label_str.split(".")[0])
            if label_num > 0:
                label2track = tsnum2label2track.get(ts_num, {})
                label2track[label_num] = track_number
                tsnum2label2track[ts_num] = label2track
    assert done, "Not finished reading tracks? " + repr(count)
    return tsnum2label2track

if __name__=="__main__":
    preprocess_sample_data()
