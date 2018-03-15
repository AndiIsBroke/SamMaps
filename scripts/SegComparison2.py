# -*- coding: utf-8 -*-
import os
import sys
from os import mkdir
from os.path import exists, split

import numpy as np

from timagetk.components import imread, imsave
from timagetk.plugins import registration
from timagetk.wrapping import bal_trsf
from timagetk.algorithms import apply_trsf
from timagetk.algorithms import compose_trsf
from timagetk.algorithms import isometric_resampling

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
# ref_ch_name = 'PI'
ref_ch_name = sys.argv[3]
try:
    assert sys.argv[-1] == 'force'
except:
    force = False
else:
    force = True


"""
Performs label matching of segmentation after performing non-linear deformation estimation.

Examples
--------
$ python SamMaps/scripts/TissueLab/SegComparison2.py 'E35' '4' 'PI'
$ python SamMaps/scripts/TissueLab/SegComparison2.py 'E37' '5' 'PI' 'force'
"""

nom_file = SamMaps_dir + "nomenclature.csv"

# -1- CZI input infos:
base_fname = "qDII-CLV3-PIN1-PI-{}-LD-SAM{}".format(XP, SAM)
time_steps = [0, 5, 10, 14]
czi_base_fname = base_fname + "-T{}.czi"

# -3- OUTPUT directory:
image_dirname = dirname + "nuclei_images/"

microscope_orientation = -1  # inverted microscope!
back_id = 1

# By default we register all other channels:
extra_channels = list(set(channel_names) - set([ref_ch_name]))

time2index = {t: n for n, t in enumerate(time_steps)}
index2time = {t: n for n, t in time2index.items()}

print "\n# - Building list of images for which to apply registration process:"
list_img_fname = []
for n, t in enumerate(time_steps):
    # -- Get the INR file names:
    path_suffix, img_fname = get_nomenclature_channel_fname(czi_base_fname.format(t), nom_file, ref_ch_name)
    print "  - Time-point {}, adding image {}...".format(n, img_fname)
    img_fname = image_dirname + path_suffix + img_fname
    list_img_fname.append(img_fname)

# t_ref = time_steps[-1]
# t_float_list = time_steps[:-1]
# time_reg_list = [(t_ref, t) for t in t_float_list]

# trsf_type ='rigid'
#
# # - Build the list of result transformation filenames to check if they exist (if, not they will be computed):
# # res_trsf_list = []
# seq_res_trsf_list = []
# for t_ref, t_float in time_reg_list:  # 't' here refer to 't_float'
#     float_img_path, float_img_fname = split(list_img_fname[time2index[t_float]])
#     float_img_path += "/"
#     # - Get the result image file name & path (output path), and create it if necessary:
#     res_img_fname = get_res_img_fname(float_img_fname, t_ref, t_float, trsf_type)
#     res_path = float_img_path + '{}_registrations/'.format(trsf_type)
#     # - Get sequence registration result trsf filename and write trsf:
#     # res_trsf_list.append(res_path + get_res_trsf_fname(float_img_fname, t_ref, t_float, trsf_type))
#     seq_res_trsf_list.append(res_path + get_res_trsf_fname(float_img_fname, t_ref, t_float, "sequence_"+trsf_type))
#
# print [exists(f) for f in seq_res_trsf_list]

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


# from timagetk.plugins import sequence_registration
# list_comp_rig_trsf, list_res_rig_img = [], []
# # if not np.all([exists(f) for f in res_trsf_list]) or force:
# if not np.all([exists(f) for f in seq_res_trsf_list]) or force:
#     print "\n# - Computing sequence {} registration:".format(trsf_type.upper())
#     list_comp_rig_trsf, list_res_rig_img = sequence_registration(list_img, method='sequence_{}_registration'.format(trsf_type), try_plugin=False)
#     for seq_trsf, seq_trsf_fname in zip(list_comp_rig_trsf, seq_res_trsf_list):
#         print "Saving existing SEQUENCE {} transformation file: {}".format(trsf_type.upper(), seq_trsf_fname)
#         seq_trsf.write(seq_trsf_fname)
# else:
#     for seq_trsf_fname in seq_res_trsf_list:
#         print "Loading existing SEQUENCE {} transformation file: {}".format(trsf_type.upper(), seq_trsf_fname)
#         seq_trsf = bal_trsf.BalTransformation()
#         seq_trsf.read(seq_trsf_fname)
#         list_comp_rig_trsf.append(seq_trsf)
#         list_res_rig_img.append(apply_trsf(imread(float_img_path + float_img_fname), seq_trsf))
#
# # - Add global reference filename:
# list_res_rig_img.append(im[-1])

list_res_rig_img = list_img
# - Resample to isometric voxelsize to be able to apply it to isometric segmented image:
print "\nResample to isometric voxelsize to be able to apply it to isometric segmented image:"
list_res_rig_img = [isometric_resampling(im) for im in list_res_rig_img]


# from timagetk.plugins.consecutive_registration import consecutive_registration
# list_res_trsf, list_res_rig_img = consecutive_registration(list_res_rig_img, method='consecutive_deformable_registration', try_plugin=False)
from timagetk.algorithms import blockmatching
trsf_type = 'iso-deformable'
list_res_trsf, list_res_img = [], []
for ind, sp_img in enumerate(list_res_rig_img):
    if ind < len(list_res_rig_img) - 1:
        # --- filenames to save:
        img_path, img_fname = split(list_img_fname[ind])
        img_path += "/"
        # -- get the result image file name & path (output path), and create it if necessary:
        res_img_fname = get_res_img_fname(img_fname, index2time[ind+1], index2time[ind], trsf_type)
        res_path = img_path + '{}_registrations/'.format(trsf_type)
        # -- get DEFORMABLE registration result trsf filename and write trsf:
        res_trsf_fname = get_res_trsf_fname(img_fname, index2time[ind+1], index2time[ind], trsf_type)
        if exists(res_path + res_img_fname) and exists(res_path + res_trsf_fname):
            print "Found saved {} registered image and tranformation!".format(trsf_type)
            print "Loading BalTransformation:\n  {}".format(res_path + res_img_fname)
            res_trsf = bal_trsf.BalTransformation()
            res_trsf.read(res_path + res_trsf_fname)
            list_res_trsf.append(res_trsf)
            print "Loading SpatialImage:\n  {}".format(res_path + res_trsf_fname)
            res_img = imread(res_path + res_img_fname)
            list_res_img.append(res_img)
            continue
        if not exists(res_path):
            print "Creating folder: {}".format(res_path)
            mkdir(res_path)
        # --- rigid registration
        print "\nPerforming rigid registration of t{} on t{}:".format(ind,
                                                                      ind + 1)
        trsf_rig, res_rig = blockmatching(sp_img, list_res_rig_img[ind + 1],
                                          param_str_2='-trsf-type rigid -py-ll 1')
        # --- deformable registration, initialisation by a rigid transformation
        print "\nPerforming deformable registration of t{} on t{}:".format(ind,
                                                                           ind + 1)
        trsf_vf, res_vf = blockmatching(sp_img, list_res_rig_img[ind + 1],
                                        left_transformation=trsf_rig,
                                        param_str_2='-trsf-type vectorfield')
        # --- composition of transformations
        print "\nPerforming composition of rigid et deformable registration..."
        res_trsf = compose_trsf([trsf_rig, trsf_vf], template_img=list_img[ind+1])
        list_res_trsf.append(res_trsf)
        # -- application de la composition des tranformations sur l'image
        print "\nApplying composed registration on ISO-ORIGINAL intensity image..."
        res_img = apply_trsf(isometric_resampling(imread(list_img_fname[ind])), res_trsf)
        list_res_img.append(res_img)
        # -- save the DEFORMABLE consecutive registered intensity image:
        if not exists(res_path + res_img_fname):
            print "\nSaving the {} registered image: {}".format(trsf_type, res_img_fname)
            imsave(res_path + res_img_fname, res_img)

        # -- save the DEFORMABLE consecutive transformation:
        if not exists(res_path + res_trsf_fname):
            print "\nSaving {} transformation file: {}".format(trsf_type, res_trsf_fname)
            res_trsf.write(res_path + res_trsf_fname)

# add last reference image
list_res_img.append(list_img[-1])  # add last reference image


# # - Apply RIGID sequence_registration on segmented images:
# list_rig_seg_img = []
# for n, img in enumerate(list_res_rig_img):
#     seg_path_suffix, seg_img_fname = get_nomenclature_segmentation_name(czi_base_fname.format(index2time[n]), nom_file, ref_ch_name)
#     for seq_trsf in list_comp_rig_trsf:
#         list_rig_seg_img.append(apply_trsf(imread(image_dirname + seg_path_suffix + seg_img_fname), seq_trsf))
#
# list_rig_seg_img.append()


# - Apply DEFORMABLE consecutive_registration on segmented images:
list_res_seg_img_fname = []
for n, img in enumerate(list_res_rig_img[:-1]):
    # Apply DEFORMABLE consecutive registration to segmented image:
    print "\nApplying estimated {} transformation on '{}' to segmented image:".format('deformable', ref_ch_name)
    seg_path_suffix, seg_img_fname = get_nomenclature_segmentation_name(czi_base_fname.format(index2time[n]), nom_file, ref_ch_name)
    trsf = list_res_trsf[n]
    res_seg_img = apply_trsf(imread(image_dirname + seg_path_suffix + seg_img_fname), trsf, param_str_2='-nearest')
    res_seg_img[res_seg_img == 0] = back_id
    res_seg_img_fname = get_res_img_fname(seg_img_fname, index2time[n+1], index2time[n], 'iso-deformable')
    print "  - {}\n  --> {}".format(seg_img_fname, res_seg_img_fname)
    res_path = image_dirname + seg_path_suffix + '{}_registrations/'.format(trsf_type)
    if not exists(res_path + res_seg_img_fname):
        imsave(res_path + res_seg_img_fname, res_seg_img)
    list_res_seg_img_fname.append(res_path + res_seg_img_fname)


# - Create reference segmented image list:
list_ref_seg_img_fname = []
for n, img_fname in enumerate(list_img_fname[:-1]):
    seg_path_suffix, seg_img_fname = get_nomenclature_segmentation_name(czi_base_fname.format(index2time[n+1]), nom_file, ref_ch_name)
    list_ref_seg_img_fname.append(image_dirname + seg_path_suffix + seg_img_fname)

zip(list_res_seg_img_fname, list_ref_seg_img_fname)

# - Then compute segmentation overlapping:
for n, (seg_imgA, seg_imgB) in enumerate(zip(list_res_seg_img_fname, list_ref_seg_img_fname)):
    uf_seg_matching_cmd = "segmentationOverlapping {} {} -rv {} -probability -bckgrdA {} -bckgrdB {} | overlapPruning - -e 0 | overlapAnalysis - {} -complete -max"
    matching_txt = image_dirname + "SegMatching-{}-t{}_on_t{}.txt".format(base_fname, n, n+1)
    seg_matching_cmd = uf_seg_matching_cmd.format(seg_imgA, seg_imgB, 1, 1, 1, matching_txt)
    print seg_matching_cmd
    # os.system(seg_matching_cmd)
