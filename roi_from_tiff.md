

## Creating FIJI ROI files from TIFF label volumes

New as of 10 Nov 2021, the `mouse_embryo_labeller.fiji_roi` module and the `roi_from_tiff` command line
script provide the ability to generate FIJI ROI files automatically for nucleus label TIFF segmentation files.

The `setup` script for `mouse_embryo_labeller` installs the `roi_from_tiff` command line program when the module is installed

```bash
% cd mouse_embryo_labeller
% pip install -e .
```

If you installed `mouse_embryo_labeller` before 10 Nov 2021 you will need to run the install script again
to create the command line program.

The program reads a TIFF file containing 3D segmented nucleus labels and generates an ROI file for every
label and layer where the label occurs in the layer.

```
$ cd my_data_folder
$ roi_from_tiff 162111Labels.tiff

Dumping ROI files for 162111Labels.tiff

Dumping to folder 162111Labels_ROI
wrote 162111Labels_ROI/label_48_layer_3.roi
wrote 162111Labels_ROI/label_48_layer_4.roi
wrote 162111Labels_ROI/label_17_layer_5.roi
wrote 162111Labels_ROI/label_48_layer_5.roi
wrote 162111Labels_ROI/label_8_layer_6.roi
wrote 162111Labels_ROI/label_17_layer_6.roi
wrote 162111Labels_ROI/label_48_layer_6.roi
wrote 162111Labels_ROI/label_51_layer_6.roi
... many lines deleted ...
wrote 162111Labels_ROI/label_43_layer_42.roi
wrote 162111Labels_ROI/label_4_layer_43.roi
wrote 162111Labels_ROI/label_14_layer_43.roi
wrote 162111Labels_ROI/label_25_layer_43.roi
wrote 162111Labels_ROI/label_4_layer_44.roi
Finished dumping ROI files.
$ _
```

The resulting files should be suitable to be read into the FIJI interface to encircle the labeled regions
in each layer.

The functionality for creating the files can also be run from a Python script of Jupyter notebook using
the `fiji_roi.dump_tiff_to_roi(tiff_path)` function or other functionality from the `fiji_roi` module.
Please see the source code for `fiji_roi.py` for more information.
