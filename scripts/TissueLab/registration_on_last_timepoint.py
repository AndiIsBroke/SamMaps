# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from os import mkdir
from os.path import exists, splitext, split

from timagetk.algorithms import apply_trsf
from timagetk.algorithms import compose_trsf
from timagetk.algorithms.trsf import save_trsf, read_trsf
from timagetk.components import imread, imsave
from timagetk.plugins import registration
from timagetk.plugins import sequence_registration


import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
    dirname = "/data/Meristems/Carlos/PIN_maps/"
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
    dirname = "/projects/SamMaps/"
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/TissueLab/')

from nomenclature import splitext_zip
from nomenclature import get_nomenclature_name
from nomenclature import get_nomenclature_channel_fname
from nomenclature import get_nomenclature_segmentation_name
from nomenclature import get_res_img_fname
from nomenclature import get_res_trsf_fname
from equalization import z_slice_contrast_stretch
# Nomenclature file location:
nomenclature_file = SamMaps_dir + "nomenclature.csv"
# OUTPUT directory:
image_dirname = dirname + "nuclei_images/"

# - DEFAULT variables:
POSS_TRSF = ['rigid', 'affine', 'deformable']
# Time steps list in hours:
DEF_TIMESTEPS = [0, 5, 10, 14]
# CZI list of channel names:
DEF_CH_NAMES = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
# Microscope orientation:
DEF_ORIENT = -1  # '-1' == inverted microscope!
# Reference channel name used to compute tranformation matrix:
DEF_REF_CH = 'PI'
# List of channels for which to apply the transformation, by default we register all channels:
DEF_SUPP_CHANNELS = list(set(DEF_CH_NAMES) - set([DEF_REF_CH]))


# PARAMETERS:
# -----------
import argparse
parser = argparse.ArgumentParser(description='Consecutive backward registration.')
# positional arguments:
parser.add_argument('xp_id', type=str,
                    help="basename of the experiment (CZI file) without time-step and extension")
parser.add_argument('trsf_type', type=str, default='rigid',
                    help="type of registration to compute, default is 'rigid', valid options are {}".format(POSS_TRSF))
# optional arguments:
parser.add_argument('--time_steps', type=int, nargs='+', default=DEF_TIMESTEPS,
                    help="list of time steps (in hours) to use, '{}' by default".format(DEF_TIMESTEPS))
parser.add_argument('--channel_names', type=str, nargs='+', default=DEF_CH_NAMES,
                    help="list of channel names for the CZI, '{}' by default".format(DEF_CH_NAMES))
parser.add_argument('--ref_ch_name', type=str, default=DEF_REF_CH,
                    help="CZI channel name used to performs the registration, '{}' by default".format(DEF_REF_CH))
parser.add_argument('--extra_channels', type=str, nargs='+', default=DEF_SUPP_CHANNELS,
                    help="list of channel names for which to apply the estimated transformation, '{}' by default".format(DEF_SUPP_CHANNELS))
parser.add_argument('--microscope_orientation', type=int, default=DEF_ORIENT,
                    help="orientation of the microscope (i.e. set '-1' when using an inverted microscope), '{}' by default".format(DEF_ORIENT))
parser.add_argument('--force', action='store_true',
                    help="if given, force computation of registration matrix even if they already exists, else skip it, 'False' by default")
args = parser.parse_args()

# - Variables definition from argument parsing:
base_fname = args.xp_id
trsf_type = args.trsf_type
try:
    assert trsf_type in POSS_TRSF
except:
    raise ValueError("Unknown tranformation type '{}', valid options are: {}".format(trsf_type, POSS_TRSF))

# - Variables definition from optional arguments:
time_steps = args.time_steps
print "Got '{}' as list time steps.".format(time_steps)
channel_names = args.channel_names
print "Got '{}' as list of CZI channel names.".format(channel_names)
ref_ch_name = args.ref_ch_name
print "Got '{}' as the reference channel name.".format(ref_ch_name)

extra_channels = args.extra_channels
if extra_channels != []:
    try:
        assert np.alltrue([ch in channel_names for ch in extra_channels])
    except:
        raise ValueError("Optional argument '--extra_channels' contains unknow channel names!")
    if ref_ch_name != DEF_REF_CH:
        extra_channels = list(set(extra_channels) - set([ref_ch_name]))
    print "Got '{}' as list of channel to which transformation will be applyed.".format(extra_channels)
else:
    print "Estimated transformations will be applied only to the reference intensity image."

force =  args.force
if force:
    print "WARNING: any existing files will be overwritten!"
else:
    print "Existing files will be kept."

microscope_orientation = args.microscope_orientation
if microscope_orientation == -1:
    print "INVERTED microscope specification!"
elif microscope_orientation == 1:
    print "UPRIGHT microscope specification!"
else:
    raise ValueError("Unknown microscope specification, use '1' for upright, '-1' for inverted!")

# Examples
# --------
# python registration_on_last_timepoint.py 'qDII-CLV3-PIN1-PI-E35-LD-SAM4' 'rigid'
# python registration_on_last_timepoint.py 'qDII-CLV3-PIN1-PI-E35-LD-SAM4' 'deformable'
# python registration_on_last_timepoint.py 'qDII-CLV3-PIN1-PI-E35-LD-SAM4' 'deformable' --time_steps 0 5 10

# - Define variables AFTER argument parsing:
czi_time_series = ['{}-T{}.czi'.format(base_fname, t) for t in time_steps]
czi_base_fname = base_fname + "-T{}.czi"

print "\n# - Building list of images for which to apply registration process:"
list_img_fname, list_img = [], []
for n, t in enumerate(time_steps):
    # -- Get the INR file names:
    path_suffix, img_fname = get_nomenclature_channel_fname(czi_base_fname.format(t), nomenclature_file, ref_ch_name)
    print "  - Time-point {}, adding image {}...".format(n, img_fname)
    img_fname = image_dirname + path_suffix + img_fname
    list_img_fname.append(img_fname)

t_ref = time_steps[-1]
t_float_list = time_steps[:-1]
time_reg_list = [(t_ref, t) for t in t_float_list]
time2index = {t: n for n, t in enumerate(time_steps)}

# Test if we really have a sequence to register:
# not_sequence = True if time2index[t_ref] - time2index[t_float_list[0]] > 1 else False
not_sequence = True if len(time_steps) == 2 else False
if not_sequence:
    print "NOT A SEQUENCE!"
else:
    print "SEQUENCE"

# - Build the list of result transformation filenames to check if they exist (if, not they will be computed):
# res_trsf_list = []
seq_res_trsf_list = []
for t_ref, t_float in time_reg_list:  # 't' here refer to 't_float'
    float_img_path, float_img_fname = split(list_img_fname[time2index[t_float]])
    float_img_path += "/"
    # - Get the result image file name & path (output path), and create it if necessary:
    res_img_fname = get_res_img_fname(float_img_fname, t_ref, t_float, trsf_type)
    res_path = float_img_path + '{}_registrations/'.format(trsf_type)
    # - Get sequence registration result trsf filename and write trsf:
    # res_trsf_list.append(res_path + get_res_trsf_fname(float_img_fname, t_ref, t_float, trsf_type))
    seq_res_trsf_list.append(res_path + get_res_trsf_fname(float_img_fname, t_ref, t_float, "sequence_"+trsf_type))

print ""
for f in seq_res_trsf_list:
    print "Existing tranformation file {}: {}\n".format(f, exists(f))

list_img = []
print "\n# - Loading list of images for which to apply registration process:"
for n, img_fname in enumerate(list_img_fname):
    print "  - Time-point {}, reading image {}...".format(n, img_fname)
    im = imread(img_fname)
    if ref_ch_name.find('raw') != -1:
        im = z_slice_contrast_stretch(im)
    else:
        pass
    list_img.append(im)

list_comp_trsf, list_res_img = [], []
# if not np.all([exists(f) for f in res_trsf_list]) or force:
if not np.all([exists(f) for f in seq_res_trsf_list]) or force:
    if not_sequence:
        list_comp_trsf, list_res_img = registration(list_img[0], list_img[1], method='{}_registration'.format(trsf_type), try_plugin=False)
    else:
        print "\n# - Computing sequence {} registration:".format(trsf_type.upper())
        list_comp_trsf, list_res_img = sequence_registration(list_img, method='sequence_{}_registration'.format(trsf_type), try_plugin=False)
    # - Save estimated tranformations:
    for seq_trsf, seq_trsf_fname in zip(list_comp_trsf, seq_res_trsf_list):
        print "Saving computed SEQUENCE {} transformation file: {}".format(trsf_type.upper(), seq_trsf_fname)
        save_trsf(seq_trsf, seq_trsf_fname)
else:
    for seq_trsf_fname in seq_res_trsf_list:
        print "Loading existing SEQUENCE {} transformation file: {}".format(trsf_type.upper(), seq_trsf_fname)
        read_trsf(seq_trsf, seq_trsf_fname)
        list_comp_trsf.append(seq_trsf)
        list_res_img.append(apply_trsf(imread(float_img_path + float_img_fname), seq_trsf))

# - Get the reference file name & path:
ref_im = list_img[-1]  # reference image is the last time-point
ref_img_path, ref_img_fname = split(list_img_fname[-1])
composed_trsf = zip(list_comp_trsf, t_float_list)
for n, (trsf, t) in enumerate(composed_trsf):  # 't' here refer to 't_float'
    # - Get the float file name & path:
    float_im = list_img[time2index[t]]
    float_img_path, float_img_fname = split(list_img_fname[time2index[t]])
    float_img_path += "/"
    # - Get the result image file name & path (output path), and create it if necessary:
    res_img_fname = get_res_img_fname(float_img_fname, t_ref, t, trsf_type)
    res_path = float_img_path + '{}_registrations/'.format(trsf_type)
    if not exists(res_path):
        mkdir(res_path)
    # - Get result trsf filename and write trsf:
    res_trsf_fname = get_res_trsf_fname(float_img_fname, t_ref, t, trsf_type)

    if not exists(res_path + res_trsf_fname) or force:
        if not_sequence or t == time_steps[-2]:
            # -- No need to "adjust" for time_steps[-2]/t_ref registration since it is NOT a composition:
            print "\n# - Saving {} t{}/t{} registration:".format(trsf_type.upper(), time2index[t], time2index[t_ref])
            res_trsf = trsf
            res_im = list_res_img[-2]
        else:
            # -- One last round of vectorfield using composed transformation as init_trsf:
            print "\n# - Final {} registration adjustment for t{}/t{} composed transformation:".format(trsf_type.upper(), time2index[t], time2index[t_ref])
            py_hl = 3  # defines highest level of the blockmatching-pyramid
            py_ll = 0  # defines lowest level of the blockmatching-pyramid
            print '  - t_{}h floating fname: {}'.format(t, float_img_fname)
            print '  - t_{}h reference fname: {}'.format(t_ref, ref_img_fname)
            # print '  - {} t_{}h/t_{}h composed-trsf as initialisation'.format(trsf_type, t, t_ref)
            print ""
            # res_trsf, res_im = registration(float_im, ref_im, method='{}_registration'.format(trsf_type), init_trsf=trsf, pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll, try_plugin=False)
            res_trsf, res_im = registration(list_res_img[time2index[t]], ref_im, method='{}_registration'.format(trsf_type), pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll, try_plugin=False)
            # res_trsf, res_im = registration(float_im, ref_im, method='{}_registration'.format(trsf_type), left_trsf=trsf, pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll, try_plugin=False)
            res_trsf = compose_trsf([trsf, res_trsf], template_img=ref_im)
            print ""

        # - Save result image and tranformation:
        print "Writing image file: {}".format(res_img_fname)
        imsave(res_path + res_img_fname, apply_trsf(imread(float_img_path + float_img_fname), res_trsf))
        print "Writing trsf file: {}".format(res_trsf_fname)
        save_trsf(res_trsf, res_path + res_trsf_fname)
    else:
        print "Existing image file: {}".format(res_img_fname)
        print "Loading existing {} transformation file: {}".format(trsf_type.upper(), res_trsf_fname)
        read_trsf(res_trsf, res_path + res_trsf_fname)

    # -- Apply estimated transformation to other channels of the floating CZI:
    if extra_channels:
        print "\nApplying estimated {} transformation on '{}' to other channels: {}".format(trsf_type.upper(), ref_ch_name, ', '.join(extra_channels))
        for x_ch_name in extra_channels:
            # --- Get the extra channel filenames:
            x_ch_path_suffix, x_ch_fname = get_nomenclature_channel_fname(czi_base_fname.format(t), nomenclature_file, x_ch_name)
            # --- Defines output filename:
            res_x_ch_fname = get_res_img_fname(x_ch_fname, t_ref, t, trsf_type)
            if not exists(res_path + res_x_ch_fname) or True:
                print "  - {}\n  --> {}".format(x_ch_fname, res_x_ch_fname)
                # --- Read the extra channel image file:
                x_ch_img = imread(image_dirname + x_ch_path_suffix + x_ch_fname)
                # --- Apply and save registered image:
                res_x_ch_img = apply_trsf(x_ch_img, res_trsf)
                imsave(res_path + res_x_ch_fname, res_x_ch_img)
            else:
                print "  - existing file: {}".format(res_x_ch_fname)
    else:
        print "No supplementary channels to register."
