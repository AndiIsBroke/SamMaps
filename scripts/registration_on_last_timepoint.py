# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from os import sep
from os import mkdir
from os.path import splitext
from os.path import split
from os.path import exists

from timagetk.algorithms import apply_trsf
from timagetk.algorithms import compose_trsf
from timagetk.io.io_trsf import save_trsf
from timagetk.io.io_trsf import read_trsf
from timagetk.io import imsave
from timagetk.plugins import registration
from timagetk.plugins import sequence_registration

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/lib/')

from nomenclature import splitext_zip
from nomenclature import get_res_img_fname
from nomenclature import get_res_trsf_fname
from equalization import z_slice_contrast_stretch
from segmentation_pipeline import read_image

# - DEFAULT variables:
POSS_TRSF = ['rigid', 'affine', 'deformable']
# Microscope orientation:
DEF_ORIENT = -1  # '-1' == inverted microscope!


# PARAMETERS:
# -----------
import argparse
parser = argparse.ArgumentParser(description='Consecutive backward registration from last time-point.')
# positional arguments:
parser.add_argument('images', type=str, nargs='+',
                    help="list of images filename to register.")
                    # optional arguments:
parser.add_argument('--trsf_type', type=str, default='rigid',
                    help="type of registration to compute, default is 'rigid', valid options are {}".format(POSS_TRSF))
parser.add_argument('--time_steps', type=int, nargs='+',
                    help="list of time steps, should be sorted as the list of images to register!")
parser.add_argument('--extra_im', type=str, nargs='+', default=None,
                    help="list of extra images to which the registration should also be applied to, should be sorted as the list of images to register!")
parser.add_argument('--microscope_orientation', type=int, default=DEF_ORIENT,
                    help="orientation of the microscope (i.e. set '-1' when using an inverted microscope), '{}' by default".format(DEF_ORIENT))
parser.add_argument('--time_unit', type=str, default='h',
                    help="Time unist of the time-steps, in hours (h) by default.")
parser.add_argument('--force', action='store_true',
                    help="if given, force computation of registration matrix even if they already exists, else skip it, 'False' by default")
args = parser.parse_args()

################################################################################
# - Parameters & variables definition:
################################################################################

# - Variables definition from argument parsing:
imgs2reg = args.images
trsf_type = args.trsf_type
try:
    assert trsf_type in POSS_TRSF
except AssertionError:
    raise ValueError("Unknown tranformation type '{}', valid options are: {}".format(trsf_type, POSS_TRSF))

# - Variables definition from optional arguments parsing:
try:
    time_steps = args.time_steps
    print "Got '{}' as list time steps.".format(time_steps)
except:
    time_steps = range(len(imgs2reg))
try:
    assert len(time_steps) == len(imgs2reg)
except AssertionError:
    raise ValueError("Not the same number of images ({}) and time-steps ({}).".format(len(imgs2reg), len(time_steps)))
extra_im = args.extra_im
if extra_im:
    try:
        assert len(extra_im) == len(imgs2reg)
    except AssertionError:
        raise ValueError("Not the same number of images ({}) and extra images ({}).".format(len(imgs2reg), len(extra_im)))
time_unit = args.time_unit
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

# - Blockmatching parameters:
################################################################################
py_hl = 3  # defines highest level of the blockmatching-pyramid
if trsf_type == 'rigid':
    py_ll = 1  # defines lowest level of the blockmatching-pyramid
else:
    py_ll = 0  # defines lowest level of the blockmatching-pyramid


################################################################################
# - Checkpoints:
################################################################################

# - Make sure the images are sorted chronologically:
################################################################################
index_ts = np.argsort(time_steps)
time_steps = [time_steps[i] for i in index_ts]
imgs2reg = [imgs2reg[i] for i in index_ts]

print "\n# - Checking the list of images for which to performs the registration process:"
list_img_fname, list_img = [], []
for n, t in enumerate(time_steps):
    # -- Get the INR file names:
    img_fname = imgs2reg[n]
    print "  - Time-point {} ({}{}), adding image: {}...".format(n, t, time_unit, img_fname)
    list_img_fname.append(img_fname)

t_ref = time_steps[-1]
t_float_list = time_steps[:-1]
time_reg_list = [(t_ref, t) for t in t_float_list]
time2index = {t: n for n, t in enumerate(time_steps)}

# - Make sure the destination folder exists:
################################################################################
float_img_path, _ = split(list_img_fname[0])
try:
    dest_folder = float_img_path + '/{}_registrations/'.format(trsf_type)
    print("Creating output folder:", dest_folder)
    mkdir(dest_folder)
except OSError as e:
    print(e)
    pass

# - Make sure we do have a sequence to register:
################################################################################
# not_sequence = True if time2index[t_ref] - time2index[t_float_list[0]] > 1 else False
not_sequence = True if len(time_steps) <= 2 else False


# - Get the filenames for transformations & check if they already exist or not:
################################################################################

# - Build the list of result transformation filenames:
# res_trsf_list = []
seq_res_trsf_list = []
for t_ref, t_float in time_reg_list:
    float_img_path, float_img_fname = split(list_img_fname[time2index[t_float]])
    if float_img_path != "":
        float_img_path += "/"
    # - Get the result image file name & path (output path), and create it if necessary:
    res_img_fname = get_res_img_fname(float_img_fname, t_ref, t_float, trsf_type)
    # - Get sequence registration result trsf filename and write trsf:
    # res_trsf_list.append(dest_folder + get_res_trsf_fname(float_img_fname, t_ref, t_float, trsf_type))
    seq_res_trsf_list.append(dest_folder + get_res_trsf_fname(float_img_fname, t_ref, t_float, "sequence_"+trsf_type))

# - Check if the result transformation files exists:
print ""
for f in seq_res_trsf_list:
    print "Existing tranformation file {}: {}\n".format(f, exists(f))


################################################################################
# - Computation:
################################################################################

# - Loading images:
list_img = []
print "\n# - Loading list of images to register:"
for n, img_fname in enumerate(list_img_fname):
    print "  - Time-point {}, reading image {}...".format(n, img_fname)
    im = read_image(img_fname)
    # print "  ---> Contrast stretching...".format(n, img_fname)
    # im = z_slice_contrast_stretch(im)
    list_img.append(im)


list_comp_trsf, list_res_img = [], []
# if not np.all([exists(f) for f in res_trsf_list]) or force:
if not np.all([exists(f) for f in seq_res_trsf_list]) or force:
    if not_sequence:
        list_comp_trsf, list_res_img = registration(list_img[0], list_img[1], method=trsf_type, pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll)
        list_comp_trsf = [list_comp_trsf]
    else:
        print "\n# - Computing SEQUENCE {} registration:".format(trsf_type.upper())
        list_comp_trsf, list_res_img = sequence_registration(list_img, method=trsf_type, return_images=True, pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll)
    # - Save estimated tranformations:
    for seq_trsf, seq_trsf_fname in zip(list_comp_trsf, seq_res_trsf_list):
        print "Saving computed SEQUENCE {} transformation file: {}".format(trsf_type.upper(), seq_trsf_fname)
        save_trsf(seq_trsf, seq_trsf_fname)
else:
    for seq_trsf_fname in seq_res_trsf_list:
        print "Loading existing SEQUENCE {} transformation file: {}".format(trsf_type.upper(), seq_trsf_fname)
        seq_trsf = read_trsf(seq_trsf_fname)
        list_comp_trsf.append(seq_trsf)
        list_res_img.append(apply_trsf(read_image(float_img_path + float_img_fname), seq_trsf))

# - Get the reference file name & path:
ref_im = list_img[-1]  # reference image is the last time-point
ref_img_path, ref_img_fname = split(list_img_fname[-1])
composed_trsf = zip(list_comp_trsf, t_float_list)
for n, (trsf, t) in enumerate(composed_trsf):  # 't' here refer to 't_float'
    # - Get the float file name & path:
    float_im = list_img[time2index[t]]
    float_img_path, float_img_fname = split(list_img_fname[time2index[t]])
    float_img_path += "/"
    # - Get the result image file name & path (output path):
    res_img_fname = get_res_img_fname(float_img_fname, t_ref, t, trsf_type)
    # - Get result trsf filename and write trsf:
    res_trsf_fname = get_res_trsf_fname(float_img_fname, t_ref, t, trsf_type)

    if not exists(dest_folder + res_trsf_fname) or force:
        if not_sequence or t == time_steps[-2]:
            # -- No need to "adjust" for time_steps[-2]/t_ref registration since it is NOT a composition:
            print "\n# - Saving {} t{}/t{} registration:".format(trsf_type.upper(), time2index[t], time2index[t_ref])
            res_trsf = trsf
        elif trsf_type == 'deformable':
            # -- One last round of vectorfield using composed transformation as init_trsf:
            print "\n# - Final {} registration adjustment for t{}/t{} composed transformation:".format(trsf_type.upper(), time2index[t], time2index[t_ref])
            print '  - t_{}h floating fname: {}'.format(t, float_img_fname)
            print '  - t_{}h reference fname: {}'.format(t_ref, ref_img_fname)
            # print '  - {} t_{}h/t_{}h composed-trsf as initialisation'.format(trsf_type, t, t_ref)
            print ""
            res_trsf, _ = registration(list_res_img[time2index[t]], ref_im, method=trsf_type, pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll)
            res_trsf = compose_trsf([trsf, res_trsf], template_img=ref_im)
            print ""
        else:
            res_trsf = trsf
            print "No need to performs supplementary round of registration for trsf_type: {}".format(trsf_type)

        # - Save result image and tranformation:
        print "Writing image file: {}".format(res_img_fname)
        imsave(dest_folder + res_img_fname, apply_trsf(read_image(float_img_path + float_img_fname), res_trsf))
        print "Writing trsf file: {}".format(res_trsf_fname)
        save_trsf(res_trsf, dest_folder + res_trsf_fname)
    else:
        print "Existing image file: {}".format(res_img_fname)
        print "Loading existing {} transformation file: {}".format(trsf_type.upper(), res_trsf_fname)
        res_trsf = read_trsf(dest_folder + res_trsf_fname)

    # -- Apply estimated transformation to other channels of the floating CZI:
    if extra_im:
        print "\nApplying estimated {} transformations to other channels...".format(trsf_type.upper())
        for n, x_ch_fname in enumerate(extra_im):
            # --- Defines output filename:
            x_ch_path, x_ch_fname = split(x_ch_fname)
            res_x_ch_fname = get_res_img_fname(x_ch_fname, t_ref, t, trsf_type)
            if not exists(dest_folder + res_x_ch_fname) or True:
                print "  - {}\n  --> {}".format(x_ch_fname, res_x_ch_fname)
                # --- Read the extra channel image file, apply trsf and save registered image:
                res_x_ch_img = apply_trsf(read_image(x_ch_path + sep + x_ch_fname), res_trsf)
                imsave(dest_folder + res_x_ch_fname, res_x_ch_img)
            else:
                print "  - existing file: {}".format(res_x_ch_fname)
    else:
        print "No supplementary channels to register."



# Work In Progress: make MIPs and a GIF out of them ?!
# from timagetk.algorithms.reconstruction import max_intensity_projection
# mips = []
# for  n, (trsf, t) in enumerate(composed_trsf):
#     # - Get the float file name & path:
#     float_im = list_img[time2index[t]]
#     float_img_path, float_img_fname = split(list_img_fname[time2index[t]])
#     float_img_path += "/"
#     # - Get the result image file name & path (output path):
#     res_img_fname = get_res_img_fname(float_img_fname, t_ref, t, trsf_type)
#     res_im = list_res_img[n]
#     mip = max_intensity_projection(res_im)
