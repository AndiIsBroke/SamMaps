# -*- coding: utf-8 -*-
import os
import sys
from os import mkdir
from os.path import exists

from timagetk.components import imread, imsave
from timagetk.plugins import registration
from timagetk.wrapping import bal_trsf
from timagetk.algorithms import apply_trsf
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

"""
Performs label matching of segmentation after performing non-linear deformation estimation.

Examples
--------
$ python SamMaps/scripts/TissueLab/rigid_registration_on_last_timepoint.py 'E35' '4' 'rigid'
$ python SamMaps/scripts/TissueLab/rigid_registration_on_last_timepoint.py 'E37' '5' 'vectorfield'
"""

nom_file = SamMaps_dir + "nomenclature.csv"


# -1- CZI input infos:
base_fname = "qDII-CLV3-PIN1-PI-{}-LD-SAM{}".format(XP, SAM)
time_steps = [0, 5, 10, 14]
czi_time_series = ['{}-T{}.czi'.format(base_fname, t) for t in time_steps]
# -3- OUTPUT directory:
image_dirname = dirname + "nuclei_images/"
# -4- Define CZI channel names, the microscope orientation, nuclei and membrane channel names and extra channels that should also be registered:
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
ref_ch_name = 'PI'
ref_ch_name += '_raw'

czi_base_fname = base_fname + "-T{}.czi"

# By default we register all other channels:
extra_channels = list(set(channel_names) - set([ref_ch_name]))
# By default do not recompute deformation when an associated file exist:
force = False


from timagetk.plugins import sequence_registration


time_reg_list = [(t, time_steps[n+1]) for n, t in enumerate(time_steps[:-1])]
time_reg_list.reverse()
time2index = {t: n for n, t in enumerate(time_steps)}
index2time = {t: n for n, t in time2index.items()}

for t_float, t_ref in time_reg_list:
    print "\n# - List images to register:"
    # - Load intensity images filenames:
    float_path_suffix, float_img_fname = get_nomenclature_channel_fname(czi_base_fname.format(t_float), nom_file, ref_ch_name)
    ref_path_suffix, ref_img_fname = get_nomenclature_channel_fname(czi_base_fname.format(t_ref), nom_file, ref_ch_name)

    # - Get RIGID registered images filenames:
    rig_float_path_suffix = float_path_suffix + 'rigid_registrations/'
    rig_float_img_fname = get_res_img_fname(float_img_fname, t_ref, t_float, 'rigid')
    if t_ref != time_steps[-1]:
        ref_path_suffix += 'rigid_registrations/'
        ref_img_fname = get_res_img_fname(ref_img_fname, index2time[time2index[t_ref]+1], t_ref, 'rigid')

    try:
        assert exists(image_dirname + rig_float_path_suffix + rig_float_img_fname)
    except:
        # - Get the result image file name & path (output path), and create it if necessary:
        res_path = image_dirname + rig_float_path_suffix
        if not exists(res_path):
            mkdir(res_path)
        # - Get result trsf filename:
        print "\n# - RIGID registration for t{}/t{}:".format(time2index[t_float], time2index[t_ref])
        py_hl = 3  # defines highest level of the blockmatching-pyramid
        py_ll = 1  # defines lowest level of the blockmatching-pyramid
        print '  - t_{}h floating fname: {}'.format(t_float, float_img_fname)
        im_float = imread(image_dirname + float_path_suffix + float_img_fname)
        # im_float = isometric_resampling(im_float)
        print '  - t_{}h reference fname: {}'.format(t_ref, ref_img_fname)
        im_ref = imread(image_dirname + ref_path_suffix + ref_img_fname)
        # im_ref = isometric_resampling(im_ref)
        print ""
        res_trsf, res_im = registration(im_float, im_ref, method='rigid_registration', pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll, try_plugin=False)
        print ""
        # - Save result image and tranformation:
        print "Writing image file: {}".format(rig_float_img_fname)
        imsave(res_path + rig_float_img_fname, res_im)
        print "Writing trsf file: {}".format(res_trsf_fname)
        res_trsf.write(res_path + res_trsf_fname)

    # - Get the result image file name & path (output path), and create it if necessary:
    res_img_fname = get_res_img_fname(float_img_fname, t_ref, t_float, 'deformable')
    res_path = image_dirname + float_path_suffix + 'deformable_registrations/'
    if not exists(res_path):
        mkdir(res_path)
    # - Get result trsf filename:
    res_trsf_fname = get_res_trsf_fname(float_img_fname, t_ref, t_float, 'deformable')

    if not exists(res_path + res_trsf_fname) or force:
        print "\n# - DEFORMABLE registration for t{}/t{}:".format(time2index[t_float], time2index[t_ref])
        py_hl = 3  # defines highest level of the blockmatching-pyramid
        py_ll = 0  # defines lowest level of the blockmatching-pyramid
        print '  - t_{}h floating fname: {}'.format(t_float, float_img_fname)
        im_float = imread(image_dirname + float_path_suffix + float_img_fname)
        # im_float = isometric_resampling(im_float)
        print '  - t_{}h reference fname: {}'.format(t_ref, ref_img_fname)
        im_ref = imread(image_dirname + ref_path_suffix + ref_img_fname)
        # im_ref = isometric_resampling(im_ref)
        print ""
        res_trsf, res_im = registration(im_float, im_ref, method='deformable_registration', pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll, try_plugin=False)
        print ""
        # - Save result image and tranformation:
        print "Writing image file: {}".format(res_img_fname)
        imsave(res_path + res_img_fname, res_im)
        print "Writing trsf file: {}".format(res_trsf_fname)
        res_trsf.write(res_path + res_trsf_fname)
    # else:
    #     print "Loading existing {} transformation file: {}".format('deformable', res_trsf_fname)
    #     res_trsf = bal_trsf.BalTransformation()
    #     res_trsf.read(res_path + res_trsf_fname)

    # -- Apply estimated transformation to segmented image:
    seg_path_suffix, seg_img_fname = get_nomenclature_segmentation_name(czi_base_fname.format(t_float), nom_file, ref_ch_name)
    # - Get RIGID registered segmentedimages filename:
    rig_seg_img_fname = get_res_img_fname(seg_img_fname, t_ref, t_float, 'rigid')
    fname = image_dirname + seg_path_suffix + 'rigid_registrations/' + rig_seg_img_fname
    if exists(fname):
        print "\nApplying estimated {} transformation on '{}' to segmented image:".format('deformable', ref_ch_name)
        res_seg_img_fname = get_res_img_fname(seg_img_fname, t_ref, t_float, 'deformable')
        if not exists(res_path + res_seg_img_fname):
            try:
                res_trsf
            except NameError:
                print "Loading existing {} transformation file: {}".format('deformable', res_trsf_fname)
                res_trsf = bal_trsf.BalTransformation()
                res_trsf.read(res_path + res_trsf_fname)
            print "  - {}\n  --> {}".format(seg_img_fname, res_seg_img_fname)
            # --- Read the segmented image file:
            seg_im = imread(fname)
            res_seg_im = apply_trsf(seg_im, res_trsf, tem, param_str_2=' -nearest -param')
            # --- Apply and save registered segmented image:
            imsave(res_path + res_seg_img_fname, res_seg_im)
        else:
            print "  - existing file: {}".format(res_seg_img_fname)
    else:
        print "Could not find segmented image:\n  '{}'".format(image_dirname + seg_path_suffix + seg_img_fname)

    if t_ref != time_steps[-1]:
        ref_seg_fname = get_res_img_fname(seg_img_fname, t_ref, t_float, 'deformable')
    else:
        ref_path_suffix, ref_seg_fname = get_nomenclature_segmentation_name(czi_base_fname.format(t_ref), nom_file, ref_ch_name)

    seg_imgA = image_dirname + ref_path_suffix + ref_seg_fname
    seg_imgB = res_path + res_seg_img_fname
    # -Then compute segmentation overlapping:
    uf_seg_matching_cmd = "segmentationOverlapping {} {} -rv {} -probability -bckgrdA {} -bckgrdB {} | overlapPruning - -e 0 | overlapAnalysis - {} -complete -max"

    matching_txt = "SegMatching--{}--{}.txt".format(ref_seg_fname, res_seg_img_fname)

    seg_matching_cmd = uf_seg_matching_cmd.format(seg_imgA, seg_imgB, 1, 1, 1, matching_txt)

    print seg_matching_cmd
    # os.system(seg_matching_cmd)
