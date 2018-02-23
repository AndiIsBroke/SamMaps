import numpy as np
import pandas as pd
from os.path import exists, splitext

from timagetk.algorithms import apply_trsf
from timagetk.components import imread, imsave
from timagetk.plugins import registration
from timagetk.wrapping import bal_trsf

# from timagetk.wrapping.bal_trsf import BalTransformation

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    sys.path.append('/data/Meristems/Carlos/SamMaps/scripts/TissueLab/')
elif platform.uname()[1] == "calculus":
    sys.path.append('/projects/SamMaps/scripts/SamMaps_git/scripts/TissueLab/')
else:
    raise ValueError("Unknown custom path to 'SamMaps/scripts/TissueLab/' for this system...")

from nomenclature import splitext_zip
from nomenclature import get_nomenclature_name
from nomenclature import get_nomenclature_channel_fname

dirname = "/data/Meristems/Carlos/"
nomenclature_file = dirname + "SamMaps/nomenclature.csv"

# PARAMETERS:
# -----------
# -1- CZI input infos:
czi_dirname = dirname + "PIN_maps/microscopy/20171110 MS-E35 LD qDII-CLV3-PIN1-PI/RAW/"
czi_time_series = ['qDII-CLV3-PIN1-PI-E35-LD-SAM4-T0.czi', 'qDII-CLV3-PIN1-PI-E35-LD-SAM4-T5.czi', 'qDII-CLV3-PIN1-PI-E35-LD-SAM4-T10.czi', 'qDII-CLV3-PIN1-PI-E35-LD-SAM4-T14.czi']
temp_reg = [(14, 10), (10, 5), (5, 0)]
# -3- OUTPUT directory:
image_dirname = dirname + "PIN_maps/nuclei_images/"
# -4- Define CZI channel names, the microscope orientation, nuclei and membrane channel names and extra channels that should also be registered:
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
membrane_ch_name = 'PI'

# By default we register all other channels:
extra_signal = list(set(channel_names) - set([membrane_ch_name]))
# By default do not recompute deformation when an associated file exist:
force = False


def get_res_trsf_fname(base_fname, t_ref, t_float, trsf_types):
    """
    Return a formatted result transformation filename.
    """
    if isinstance(trsf_types, str):
        trsf_types = [trsf_types]
    base_fname, ext = splitext_zip(base_fname)
    compo_trsf = '_o_'.join(trsf_types)
    return base_fname + "-T{}_on_T{}-{}.trsf".format(t_float, t_ref, compo_trsf)


def get_res_img_fname(base_fname, t_ref, t_float, trsf_types):
    """
    Return a formatted result image filename.
    """
    if isinstance(trsf_types, str):
        trsf_types = [trsf_types]
    base_fname, ext = splitext_zip(base_fname)
    if ext == "":
        ext = '.inr'
    compo_trsf = '_o_'.join(trsf_types)
    return base_fname + "-T{}_on_T{}-{}{}".format(t_float, t_ref, compo_trsf, ext)


trsf_type = 'rigid'
base_fname = "qDII-CLV3-PIN1-PI-E35-LD-SAM4"
czi_base_fname = base_fname + "-T{}.czi"

# - Initialise index dictionaries to keep track of images locations:
file_index = {}
time_index = {}

for n, (t_ref, t_float) in enumerate(temp_reg):
    print "\n\n# - Registering t{} on t{}...".format(t_float, t_ref)
    if n == 0:
        # -- Get the reference CZI file names:
        ref_czi_fname = czi_base_fname.format(t_ref)
        # -- Get the reference INR file names:
        ref_path_suffix, ref_im_fname = get_nomenclature_channel_fname(ref_czi_fname, nomenclature_file, membrane_ch_name)
        # -- Add last time-point filename to the index dictionaries:
        file_index[n] = image_dirname + ref_path_suffix + ref_im_fname
        time_index[n] = t_ref
    else:
        # Use the previously registered image to register all image in the same reference frame:
        ref_path_suffix, ref_im_fname = res_path_suffix, res_img_fname
    # -- Get the float CZI file names:
    float_czi_fname = czi_base_fname.format(t_float)
    # -- Get the float INR file names:
    float_path_suffix, float_im_fname = get_nomenclature_channel_fname(float_czi_fname, nomenclature_file, membrane_ch_name)

    # -- Read image files:
    ref_im = imread(image_dirname + ref_path_suffix + ref_im_fname)
    float_im = imread(image_dirname + float_path_suffix + float_im_fname)
    print "Floating image filename: {}".format(float_im_fname)
    print "Reference image filename: {}".format(ref_im_fname)

    # --- Get RESULT transformation matrix and image filename:
    res_path_suffix = float_path_suffix + "registrations/"
    res_trsf_fname = get_res_trsf_fname(float_im_fname, t_ref, t_float, trsf_type)
    res_img_fname = get_res_img_fname(float_im_fname, t_ref, t_float, trsf_type)

    # -- Add transformed filenames to the index dictionaries:
    file_index[n+1] = image_dirname + res_path_suffix + res_img_fname
    time_index[n+1] = t_float

    # -- Skip estimation if associated *.trsf file exists and should not be overwritten:
    if not exists(image_dirname + res_path_suffix + res_trsf_fname) or force:
        # --- Performs REGISTRATION of t_float on t_ref:
        print "Estimated transformation filename: {}".format(res_trsf_fname)
        print "Registrated image filename: {}".format(res_img_fname)
        py_hl = 3  # defines highest level of the blockmatching-pyramid
        py_ll = 0  # defines lowest level of the blockmatching-pyramid
        res_trsf, res_img = registration(float_im, ref_im, method=trsf_type+'_registration', pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll, try_plugin=False)
        # --- Save estimated transformation matrix and registered image:
        print "Saving registered image..."
        imsave(image_dirname + res_path_suffix + res_img_fname, res_img)
        res_trsf.write(image_dirname + res_path_suffix + res_trsf_fname)
    else:
        res_trsf = bal_trsf.BalTransformation()
        res_trsf.read(image_dirname + res_path_suffix + res_trsf_fname)


    # --- Apply estimated transformation to other channels of the floating CZI:
    if extra_signal:
        print "\nGot EXTRA signal to which to apply estimated transformation: {}".format(', '.join(extra_signal))
        for x_sig_name in extra_signal:
            # ---- Get and read the extra signal images:
            x_sig_path_suffix, x_sig_fname = get_nomenclature_channel_fname(float_czi_fname, nomenclature_file, x_sig_name)
            res_x_sig_fname = get_res_img_fname(x_sig_fname, t_ref, t_float, trsf_type)
            if not exists(image_dirname + x_sig_path_suffix + res_x_sig_fname) or force:
                print "\t- {}".format(x_sig_fname)
                x_sig_img = imread(image_dirname + x_sig_path_suffix + x_sig_fname)
                # ---- Defines output filename, apply and save transformation:
                print "Will save registered image: {}".format(res_x_sig_fname)
                res_x_sig_img = apply_trsf(x_sig_img, res_trsf)
                res_x_sig_path_suffix = x_sig_path_suffix + "registrations/"
                imsave(image_dirname + res_x_sig_path_suffix + res_x_sig_fname, res_x_sig_img)

    # --- Apply estimated transformation to segmented image:
    f,e = splitext_zip(float_im_fname)
    seg_img_fname = f + "_segmented" + e
    res_seg_suffix = float_path_suffix
    if exists(image_dirname + res_seg_suffix + seg_img_fname):
        f,e = splitext_zip(res_img_fname)
        res_seg_img_fname = f + "_segmented" + e
        if not exists(image_dirname + res_seg_suffix + seg_img_fname) or force:
            seg_img = imread(image_dirname + res_seg_suffix + seg_img_fname)
            res_seg_im = apply_trsf(seg_img, res_trsf)
            imsave(image_dirname + res_seg_suffix + res_seg_img_fname, res_seg_im)

    # -- Freeing some memory:
    del res_trsf, res_img, x_sig_img, res_x_sig_img, seg_img, res_seg_im

# - Transform into generic name:
open_file_index = {k: v.replace('_'+membrane_ch_name, '_{}') for k, v in file_index.items()}
index = pd.DataFrame().from_dict({'time': time_index, 'filenames': open_file_index})
index = index.sort_values('time')
index.to_csv(image_dirname + base_fname + "-index.csv", index=False)
