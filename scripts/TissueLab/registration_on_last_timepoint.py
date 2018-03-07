# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from os import mkdir
from os.path import exists, splitext, split

from timagetk.algorithms import apply_trsf
from timagetk.algorithms import compose_trsf
from timagetk.components import imread, imsave
from timagetk.plugins import registration
from timagetk.wrapping import bal_trsf

# from timagetk.wrapping.bal_trsf import BalTransformation
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

# XP = 'E37'
XP = sys.argv[1]
# SAM = '5'
SAM = sys.argv[2]
# trsf_type = 'deformable'
trsf_type = sys.argv[3]

# Examples
# --------
# python SamMaps/scripts/TissueLab/rigid_registration_on_last_timepoint.py 'E35' '4' 'rigid'
# python SamMaps/scripts/TissueLab/rigid_registration_on_last_timepoint.py 'E37' '5' 'vectorfield'

nomenclature_file = SamMaps_dir + "nomenclature.csv"

# PARAMETERS:
# -----------
# -1- CZI input infos:
base_fname = "qDII-CLV3-PIN1-PI-{}-LD-SAM{}".format(XP, SAM)
time_steps = [0, 5, 10, 14]
czi_time_series = ['{}-T{}.czi'.format(base_fname, t) for t in time_steps]
# -3- OUTPUT directory:
image_dirname = dirname + "nuclei_images/"
# -4- Define CZI channel names, the microscope orientation, nuclei and membrane channel names and extra channels that should also be registered:
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
membrane_ch_name = 'PI'
membrane_ch_name += '_raw'

czi_base_fname = base_fname + "-T{}.czi"

# By default we register all other channels:
extra_channels = list(set(channel_names) - set([membrane_ch_name]))
# By default do not recompute deformation when an associated file exist:
force = False


from timagetk.plugins import sequence_registration

print "\n# - Building list of images for which to apply registration process:"
list_img_fname, list_img = [], []
for n, t in enumerate(time_steps):
    # -- Get the INR file names:
    path_suffix, img_fname = get_nomenclature_channel_fname(czi_base_fname.format(t), nomenclature_file, membrane_ch_name)
    print "  - Time-point {}, adding image {}...".format(n, img_fname)
    img_fname = image_dirname + path_suffix + img_fname
    list_img_fname.append(img_fname)

t_ref = time_steps[-1]
t_float_list = time_steps[:-1]
time_reg_list = [(t_ref, t) for t in t_float_list]
time2index = {t: n for n, t in enumerate(time_steps)}

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

print [exists(f) for f in seq_res_trsf_list]

list_img = []
print "\n# - Loading list of images for which to apply registration process:"
for n, img_fname in enumerate(list_img_fname):
    print "  - Time-point {}, reading image {}...".format(n, img_fname)
    im = imread(img_fname)
    if membrane_ch_name.find('raw') != -1:
        im = z_slice_contrast_stretch(im)
    else:
        pass
    list_img.append(im)

list_comp_trsf, list_res_img = [], []
# if not np.all([exists(f) for f in res_trsf_list]) or force:
if not np.all([exists(f) for f in seq_res_trsf_list]) or force:
    print "\n# - Computing sequence {} registration:".format(trsf_type.upper())
    list_comp_trsf, list_res_img = sequence_registration(list_img, method='sequence_{}_registration'.format(trsf_type), try_plugin=False)
    for seq_trsf, seq_trsf_fname in zip(list_comp_trsf, seq_res_trsf_list):
        print "Saving existing SEQUENCE {} transformation file: {}".format(trsf_type.upper(), seq_trsf_fname)
        seq_trsf.write(seq_trsf_fname)
else:
    for seq_trsf_fname in seq_res_trsf_list:
        print "Loading existing SEQUENCE {} transformation file: {}".format(trsf_type.upper(), seq_trsf_fname)
        seq_trsf = bal_trsf.BalTransformation()
        seq_trsf.read(seq_trsf_fname)
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
        if t == time_steps[-2]:
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
        res_trsf.write(res_path + res_trsf_fname)
    else:
        print "Existing image file: {}".format(res_img_fname)
        print "Loading existing {} transformation file: {}".format(trsf_type.upper(), res_trsf_fname)
        res_trsf = bal_trsf.BalTransformation()
        res_trsf.read(res_path + res_trsf_fname)

    # -- Apply estimated transformation to other channels of the floating CZI:
    if extra_channels:
        print "\nApplying estimated {} transformation on '{}' to other channels: {}".format(trsf_type.upper(), membrane_ch_name, ', '.join(extra_channels))
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

    # -- Apply estimated transformation to segmented image:
    if trsf_type == 'rigid':
        seg_path_suffix, seg_img_fname = get_nomenclature_segmentation_name(czi_base_fname.format(t), nomenclature_file, membrane_ch_name)
        if exists(image_dirname + seg_path_suffix + seg_img_fname):
            print "\nApplying estimated {} transformation on '{}' to segmented image:".format(trsf_type.upper(), membrane_ch_name)
            res_seg_img_fname = get_res_img_fname(seg_img_fname, t_ref, t, trsf_type)
            if not exists(res_path + seg_img_fname) or True:
                print "  - {}\n  --> {}".format(seg_img_fname, res_seg_img_fname)
                # --- Read the segmented image file:
                seg_im = imread(image_dirname + seg_path_suffix + seg_img_fname)
                res_seg_im = apply_trsf(seg_im, res_trsf, param_str_2=' -nearest -param')
                # --- Apply and save registered segmented image:
                imsave(res_path + res_seg_img_fname, res_seg_im)
            else:
                print "  - existing file: {}".format(res_seg_img_fname)
        else:
            print "Could not find segmented image:\n  '{}'".format(image_dirname + seg_path_suffix + seg_img_fname)
