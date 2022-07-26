"""
Miscellaneous functionality.
"""

import os
import glob
import shutil
from PIL import Image, ImageSequence
import numpy as np
from skimage import io
from mouse_embryo_labeller import timestamp
from mouse_embryo_labeller import timestamp_collection
from mouse_embryo_labeller import nucleus_collection
import pandas as pd

EXAMPLE_FOLDER = "../example_data/"

def preprocess_sample_data(
    stride=4, 
    destination=EXAMPLE_FOLDER,
    labels_pattern="/Users/awatters/misc/lisa2/mouse-embryo-nuclei/H9/H9_%s_Labels.tiff",
    intensities_pattern="/Users/awatters/misc/lisa2/mouse-embryo-nuclei/H9OriginalIntensityImages/klbOut_CH1_%06d.klb",
    sanity_limit=10000,
    ):
    print("banana")
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
        intensities_path = intensities_pattern % (i,i)
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
        #ts.manifest = {
        #        "identity": ts.identifier,
        #        "json_path": timestamp_collection.filename_only(json_path),
        #        "npz_path": timestamp_collection.filename_only(npz_path),
        #    }
        ts.get_manifest()
    #print("now truncating all time slices...")
    #tsc.truncate_all()
    #ts_pattern = destination + "/ts%s"
    print("storing timestamps with pattern", repr(ts_pattern))
    tsc.store_all(ts_pattern)
    print("Creating empty nucleus collection...")
    nc = nucleus_collection.NucleusCollection()
    nc.save_json(destination)
    print("done.")


def to_even(x):
    return 2*int(round(x/2))    
    
    
def preprocess_sample_data_Kohrman(
    stride=4, 
    destination=EXAMPLE_FOLDER,
    labels_pattern="/Users/awatters/misc/lisa2/mouse-embryo-nuclei/H9/H9_%s_Labels.tiff",
    intensities_pattern="/Users/awatters/misc/lisa2/mouse-embryo-nuclei/H9OriginalIntensityImages/klbOut_CH1_%06d.klb",
    cropbox_path = '/media/posfailab/Chomky_Drive1/Zsombor/220218/stack_10_channel_0_obj_left/cropped/cropboxes/',
    sanity_limit=10000,
    ):
    offset = 150 #THIS WILL BREAK IF MASHA CHANGES OFFSET!!
    vpairs = pd.read_csv(os.path.join(cropbox_path, 'vpairs.csv'), index_col = [0])
    hpairs = pd.read_csv(os.path.join(cropbox_path, 'hpairs.csv'), index_col = [0])
    vpairs = tuple(map(int, vpairs['all'][0][1:-1].split(', ')))
    hpairs = tuple(map(int, hpairs['all'][0][1:-1].split(', ')))
    v1 = max(hpairs[0]-offset,0)
    v2 = min(hpairs[1]+offset, 2048)
    h1 = max(vpairs[0]-offset,0)
    h2 = min(vpairs[1]+offset, 2048)
    
    """
    import glob, os
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd
    import time

    from skimage.transform import rescale
    import scipy.ndimage as ndimage
    import imageio
    import h5py

    import matplotlib.pyplot as plt
    import seaborn as sns
    from skimage.io import imsave
    #from pyklb import readfull
    """
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
    """
    if crop_KLBs:
        if "Long" in intensities_pattern:
            which_cam = 'Long'
        else:
            which_cam = 'Short'
        images = glob.glob(root + '/out/folder_Cam_'+which_cam+'*/klbOut_Cam_'+which_cam+'*.klb')
    """    
        
        
    for i in range(sanity_limit):
        labels_path = labels_pattern % i
        #intensities_helper = intensities_subdir 
        intensities_path = intensities_pattern %(i,i)
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
        img = img[:, h1:h2, v1:v2]
            
        labels = load_tiff_array(labels_path)
        labels = np.transpose(labels,(0,2,1))
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
        #return "TR" + repr(tracknum)
        return "TR%05d" % tracknum

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

CHANNELS_FOLDER = "../channels_example"
CHANNEL0_PATTERN = "/Users/awatters/misc/LisaBrown/pole_cell/560/Crop_out_stack0_chan0_camFused_tp%(ts_number)05d.h5.tiff"
CHANNEL1_PATTERN = "/Users/awatters/misc/LisaBrown/pole_cell/560/Crop_out_stack0_chan1_camFused_tp%(ts_number)05d.h5.tiff"
CHANNEL_LABEL_PATTERN = "/Users/awatters/misc/LisaBrown/pole_cell/560/LabelDownSample_out_stack0_chan1_camFuse_%(ts_number)05d.tif"

def preprocess_channel_data(
    stride=4,
    destination=CHANNELS_FOLDER,
    labels_pattern=CHANNEL_LABEL_PATTERN,
    channel0_pattern=CHANNEL0_PATTERN,
    channel1_pattern=CHANNEL1_PATTERN,
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
            assert ts_number > 0, "NOTHING FOUND!!"
            break
        # otherwise proceed...
        print("::: found timestamp labels file", labels_path)
        c_path = channel0_pattern % D
        assert os.path.isfile(c_path), repr(labels_path) + " no matching intensities " + repr(c_path)
        #img = helper.get_pyklb_image(intensities_path)
        channel0 = load_tiff_array(c_path)
        c_path = channel1_pattern % D
        assert os.path.isfile(c_path), repr(labels_path) + " no matching intensities " + repr(c_path)
        #img = helper.get_pyklb_image(intensities_path)
        channel1 = load_tiff_array(c_path)
        labels = load_tiff_array(labels_path)
        assert channel0.shape == channel1.shape, "channel shapes don't match " + repr([channel0.shape, channel1.shape])
        # XXXX the label shape and channel shapes don't match in the sample data 
        # XXXX HACKY CORRECTION HERE:
        if channel0.shape != labels.shape:
            channel0 = channel0[:, ::2, ::2]
            channel1 = channel1[:, ::2, ::2]
        assert channel0.shape == labels.shape, "channel shape doesn't match label shape " + repr((channel0.shape, labels.shape))
        # truncate j and k
        #s_img = img[:, ::stride, ::stride]
        s_labels = labels[:, ::stride, ::stride]
        s_img = np.zeros(s_labels.shape + (3,), dtype=np.float)  # three channels required but only using 2
        s_img[:, :, :, 0] = scale_channel(channel0[:, ::stride, ::stride])
        s_img[:, :, :, 1] = scale_channel(channel1[:, ::stride, ::stride])
        ts = helper.add_timestamp(ts_number, s_img, s_labels)
        print("::: timestamp", ts.manifest)
    assert tracks_done, "Track loop didn't terminate normally.  Too many tracks?"
    tsc = helper.stored_timestamp_collection()
    print("Stored timestamp collection", tsc.manifest_path)
    return helper

def scale_channel(channel_array, maximum=256.0):
    result = np.zeros(channel_array.shape, dtype=np.float)
    factor = maximum * 1.0 / channel_array.max()
    result[:] = factor * channel_array
    return result

def check_folder(folder):
    folder = os.path.expanduser(folder)
    folder = os.path.abspath(folder)
    assert os.path.isdir(folder), "cannot find folder: " + repr(folder)
    return folder

MADDY_FOLDER = '/Users/awatters/misc/Abraham_Kohrman/maddy_data'

def prepare_collection_for_maddy_data(
    folder=MADDY_FOLDER,
    collection_subfolder = "/collection",
    tif_file_pattern="/*_rescaled_*.tif",
    intensities_file_pattern="/%05d_rescaled_low.tif",
    masks_file_pattern="/%05d_rescaled_low_cp_masks.tif",
    stride=2,
    ):
    folder = check_folder(folder)
    tiff_glob_pattern = folder + tif_file_pattern
    tiff_paths = glob.glob(tiff_glob_pattern)
    timestamp_numbers = set()
    for tiff_path in tiff_paths:
        fn = os.path.split(tiff_path)[-1]
        sprefix = fn.split("_")[0]
        iprefix = int(sprefix)
        timestamp_numbers.add(iprefix)
    collection_destination = folder + collection_subfolder
    print ("=== Making collections in", collection_destination, "for", len(timestamp_numbers), "timestamps")
    if not os.path.isdir(collection_destination):
        os.mkdir(collection_destination)
    helper = FileSystemHelper(collection_destination)
    nc = helper.stored_nucleus_collection()
    print ("Stored empty nucleus collection: ", nc.manifest_path)
    for timestamp_number in sorted(timestamp_numbers):
        intensities_path = folder + (intensities_file_pattern % timestamp_number)
        assert os.path.isfile(intensities_path), "intensities not found: " + repr(intensities_path)
        masks_path = folder + (masks_file_pattern % timestamp_number)
        assert os.path.isfile(masks_path), "masks not found: " + repr(masks_path)
        intensities = io.imread(intensities_path)
        # intensities are 64bit -- could convert to smaller format.
        masks = io.imread(masks_path)
        assert masks.shape == intensities.shape, "arrays don't match: " + repr([masks.shape, intensities.shape])
        si = intensities[::stride, ::stride, ::stride]
        sm = masks[::stride, ::stride, ::stride]
        ts = helper.add_timestamp(timestamp_number, si, sm)
        print("::: timestamp", ts.manifest)
    tsc = helper.stored_timestamp_collection()
    print("Stored timestamp collection", tsc.manifest_path)
    return helper

class ForAaronCollector:

    """
    Helper for generating collections structured like this example:
    rusty://mnt/home/akohrman/ceph/for_Aaron/
    """

    def __init__(
        self,
        destination_folder="/Users/awatters/misc/Abraham_Kohrman/for_Aaron/collection",
        root="/Users/awatters/misc/Abraham_Kohrman/for_Aaron",
        img_pattern="220309/stack_0_channel_0_obj_left/out/folder_Cam_Long_%(n)05d.lux/klbOut_Cam_Long_%(n)05d.lux.klb",
        label_pattern="220309_out/st0/klbOut_Cam_Long_%(n)05d.lux.label.tif",
    ):
        self.root = root
        self.full_image_pattern = os.path.join(root, img_pattern)
        self.full_label_pattern = os.path.join(root, label_pattern)
        self.destination_folder = destination_folder
        self.helper = FileSystemHelper(destination_folder)
        self.combined_slicing = None

    def label_path(self, ts_number):
        subst = dict(n=ts_number)
        return self.full_label_pattern % subst

    def image_path(self, ts_number):
        subst = dict(n=ts_number)
        return self.full_image_pattern % subst

    def get_slicing(self, sanity_limit=10000):
        """
        Determine the slicing containing all labels and check file consistency.
        """
        from . import geometry
        print()
        print("Determining label geometry and checking file matches.")
        done = False
        max_ts = None
        combined_slicing = None
        for ts_number in range(0, sanity_limit):
            subst = dict(n=ts_number)
            label_path = self.full_label_pattern % subst
            img_path = self.full_image_pattern % subst
            if os.path.isfile(label_path):
                assert os.path.isfile(img_path), "No matching image file: " + repr((label_path, img_path))
                label_array = self.get_label_array(label_path)
                max_ts = ts_number
                slicing = geometry.positive_slicing(label_array)
                print (ts_number, "for", repr(label_path), "slicing", slicing)
                if combined_slicing is not None:
                    combined_slicing = geometry.unify_slicing(combined_slicing, slicing)
                else:
                    combined_slicing = slicing
            else:
                done = True
                break
        assert done, "Too many label paths found: " + repr(max_ts)
        assert combined_slicing is not None, "No label paths found: " + repr(self.full_label_pattern)
        print("combined slicing", combined_slicing)
        self.combined_slicing = combined_slicing
        self.max_ts = max_ts

    def prepare_collections(self):
        if self.combined_slicing is None:
            self.get_slicing()
        for ts_number in range(self.max_ts):
            print()
            print ("Preparing ts", ts_number)
            not finished

    def get_label_array(self, from_path):
        matrix_bad_axes = load_tiff_array(from_path)
        matrix_good_axes = np.swapaxes(matrix_bad_axes, 1, 2)
        return matrix_good_axes

class ParseMatlabJSONDump:

    """
    Read a JSON dump of matlab graph similar to "Gata6Nanog1.json".
    """

    def __init__(self, json_graph):
        # the json graph dump should be a dictionary with one entry at the top level
        [(self.graph_name, self.graph_info)] = list(json_graph.items())
        self.node2timestamp = {}
        self.name2node = {}
        self.time_stamps = set()
        # collect nodes and timestamps
        edge_info = self.graph_info["Edges"]
        for edge_map in edge_info:
            [src, dst] = edge_map["EndNodes"]
            self.add_node(src)
            self.add_node(dst)
        nodes_info = self.graph_info["Nodes"]
        for node_map in nodes_info:
            node_name = node_map["Name"]
            self.add_node(node_name)
        self.ts_to_labels = {ts: set() for ts in self.time_stamps}
        # collect labels in timestamps
        for (ts, label) in self.node2timestamp:
            self.ts_to_labels[ts].add(label)
        # determine parent/child relationship
        self.node2parent = {}
        self.node2children = {node: [] for node in self.node2timestamp}
        for edge_map in edge_info:
            [src_name, dst_name] = edge_map["EndNodes"]
            src_node = self.name2node[src_name]
            dst_node = self.name2node[dst_name]
            assert dst_node not in self.node2parent, "Multiple parent? " + repr((dst_node, src_node, self.node2parent[dst_node]))
            self.node2parent[dst_node] = src_node
            self.node2children[src_node].append(dst_node)
            #assert len(self.node2children[src_node]) <= 2, "Too many children? " + repr([src_node, self.node2children[src_node]])
        # find non-split nodes
        parent_to_single_child = {}
        for (node, children) in self.node2children.items():
            if len(children) == 1:
                parent_to_single_child[node] = children[0]
        # find tracks
        start_node_to_track = {}
        start_nodes = set(parent_to_single_child.keys()) - set(parent_to_single_child.values())
        for start_node in start_nodes:
            track = []
            current_node = start_node
            while current_node is not None:
                track.append(current_node)
                current_node = parent_to_single_child.get(current_node)
            start_node_to_track[start_node] = track
        self.start_node_to_track = start_node_to_track
        
    def add_node(self, node_name):
        [ts_str, label_str] = node_name.split("_")
        ts = int(ts_str)
        label = int(label_str)
        node = (ts, label)
        self.node2timestamp[node] = ts
        self.time_stamps.add(ts)
        self.name2node[node_name] = node
    

GROUND_TRUTH_FOLDER = '/Users/awatters/misc/LisaBrown/embryo/WholeEmbryo'

class GroundTruthProcessor:

    def __init__(self, folder=GROUND_TRUTH_FOLDER):
        folder = os.path.expanduser(folder)
        folder = os.path.abspath(folder)
        assert os.path.isdir(folder), "cannot find folder: " + repr(folder)
        self.folder = folder

    def delete_collections(self):
        pattern = self.folder + "collection_*"
        to_delete = glob.glob(pattern)
        print("Deleting", len(to_delete), "collections.")
        for path in to_delete:
            shutil.rmtree(path)
            print("   ", repr(path), "deleted.")

    def make_all_collections(self, stride=4):
        prefixes = set()
        for filename in os.listdir(self.folder):
            if "collection" in filename:
                continue  # don't process collections
            split = filename.split("_")
            if len(split) == 2:
                prefixes.add(split[0])
        print("Now making", len(prefixes), "collections.")
        destinations = []
        for prefix in sorted(prefixes):
            d = self.make_collection(prefix)
            destinations.append(d)
        return destinations

    def make_collection(self, prefix, stride=4):
        # prefix like "F22"
        assert "_" not in prefix, "prefix should not incude underscore: " + repr(prefix)
        folder_pattern = self.folder + "/" + str(prefix) + "_*"
        prefix_folders = glob.glob(folder_pattern)
        assert len(prefix_folders) > 0, "no folders found for prefix: " + repr(folder_pattern)
        suffix_integers = []
        for path in prefix_folders:
            suffix = path.split("_")[-1]
            try:
                intsuffix = int(suffix)
            except ValueError:
                print("   SKIPPING BAD SUFFIX", repr((suffix, path)))
                continue
            suffix_integers.append(intsuffix)
        suffix_integers.sort()
        prefix_destination = self.folder + "/collection_" + prefix
        print()
        print("============ Creating collection", prefix_destination)
        if not os.path.isdir(prefix_destination):
            os.mkdir(prefix_destination)
        helper = FileSystemHelper(prefix_destination)
        nc = helper.stored_nucleus_collection()
        print("Stored empty nucleus collection", nc.manifest_path)
        ts_counter = 0
        for si in suffix_integers:
            suffix_folder = "%s/%s_%s" % (self.folder, prefix, si)
            print("processing folder", suffix_folder)
            assert os.path.isdir(suffix_folder), "suffix folder not found: " + repr(suffix_folder)
            if len(os.listdir(suffix_folder)) == 0:
                print ("   SKIPPING EMPTY FOLDER", suffix_folder)
                continue
            im_in = self.get_array(suffix_folder, "*/masks/*_masks_in_*.npy", stride)
            im_mi = self.get_array(suffix_folder, "*/masks/*_masks_mi_*.npy", stride)
            im_im = self.get_array(suffix_folder, "*/images/*_image_*.npy", stride)
            assert im_in.shape == im_mi.shape and im_mi.shape == im_im.shape, "bad shapes "+repr((im_in.shape, im_mi.shape, im_im.shape))
            labels = im_in.copy()
            intensities = scale_channel(im_im)
            #channels = np.zeros(intensities.shape + (3,), dtype=intensities.dtype)
            #channels[:, :, :, 1] = intensities
            #channels[:, :, :, 2] = intensities
            mi_unique = set(np.unique(im_mi)) - {0}
            if mi_unique:
                in_unique = set(np.unique(im_in)) - {0}
                common = mi_unique & in_unique
                assert not common, "normal and mitotic labels intersect " + repr(common)
                mitotic = (im_mi > 0)
                labels = np.where(mitotic, im_mi, labels)
                #channels[:, :, :, 0] = np.where(mitotic, 255, 0)
            #ts = helper.add_timestamp(ts_counter, channels, labels)
            ts = helper.add_timestamp(ts_counter, intensities, labels)
            if mi_unique:
                print("mitotic labels", mi_unique)
                ts.special_labels = list(mi_unique)
                ts.save_mapping()
            #print("   channels", channels.shape, "labels", labels.shape)
            print("::: timestamp", ts.manifest)
            ts_counter += 1
        tsc = helper.stored_timestamp_collection()
        print("Stored timestamp collection", tsc.manifest_path)
        return prefix_destination

    def get_array(self, suffix_folder, glob_pattern, stride):
        full_pattern = suffix_folder + "/" + glob_pattern
        matches = glob.glob(full_pattern)
        assert len(matches) == 1, "Wrong match count: " + repr((full_pattern, matches))
        [path] = matches
        assert os.path.isfile(path), "not a file: " + repr(path)
        result = np.load(path)
        return result[:, ::stride, ::stride]

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
        if len(a.shape) == 3:
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

def timestamp_string_now():
    "sortable millisecond timestamp for file naming."
    import time
    return "%015d" % int(time.time() * 1000)

def copy_json_files(from_folder, to_folder):
    import glob
    import shutil
    from_pattern = os.path.join(from_folder, "*.json")
    from_paths = glob.glob(from_pattern)
    for from_path in from_paths:
        fn = os.path.split(from_path)[-1]
        to_path = os.path.join(to_folder, fn)
        shutil.copyfile(from_path, to_path)

if __name__=="__main__":
    preprocess_sample_data()
