# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from os import mkdir
from os.path import exists
from os.path import splitext
from os.path import split

from timagetk.algorithms import apply_trsf
from timagetk.io import imread
from timagetk.io import imsave
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
sys.path.append(SamMaps_dir+'/scripts/lib/')

from nomenclature import splitext_zip
from nomenclature import get_nomenclature_name
from nomenclature import get_nomenclature_channel_fname
from nomenclature import get_nomenclature_segmentation_name
from nomenclature import get_res_img_fname
from nomenclature import get_res_trsf_fname
from equalization import z_slice_equalize_adapthist

# XP = 'E37'
XP = sys.argv[1]
# SAM = '5'
SAM = sys.argv[2]
# trsf_type = 'deformable'
trsf_type = sys.argv[3]


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
    print "  - Time-point {}, reading image {}...".format(n, img_fname)
    img_fname = image_dirname + path_suffix + img_fname
    list_img_fname.append(img_fname)
    im = imread(img_fname)
    if membrane_ch_name.find('raw') != -1:
        im = z_slice_equalize_adapthist(im)
    else:
        pass
    list_img.append(im)


print "\n# - Computing sequence {} registration:".format(trsf_type.upper())
list_comp_tsrf, list_res_img = sequence_registration(list_img, method='sequence_{}_registration'.format(trsf_type))


force = True
ref_im = list_img[-1]  # reference image is the last time-point
time2index = {t: n for n, t in enumerate(time_steps)}
composed_trsf = zip(list_comp_tsrf, time_steps[:-1])
for trsf, t in composed_trsf:  # 't' here refer to 't_float'
    # - Get the reference file name & path:
    ref_img_path, ref_img_fname = split(list_img_fname[-1])
    # - Get the float file name & path:
    float_im = list_img[time2index[t]]
    float_img_path, float_img_fname = split(list_img_fname[time2index[t]])
    float_img_path += "/"
    # - Get the result image file name & path (output path), and create it if necessary:
    res_img_fname = get_res_img_fname(float_img_fname, time_steps[-1], t, trsf_type)
    res_path = float_img_path + '{}_registrations/'.format(trsf_type)
    if not exists(res_path):
        mkdir(res_path)
    # - Get result trsf filename and write trsf:
    res_trsf_fname = get_res_trsf_fname(float_img_fname, time_steps[-1], t, trsf_type)

    if not exists(res_path + res_trsf_fname) or force:
        if t == time_steps[-2]:
            # -- No need to "adjust" for time_steps[-2]/time_steps[-1] registration since it is NOT a composition:
            print "\n# - Saving {} t{}/t{} registration:".format(trsf_type.upper(), time2index[t], time2index[time_steps[-1]])
            res_trsf = trsf
            res_im = list_res_img[-1]
        else:
            # -- One last round of vectorfield using composed transformation as init_trsf:
            print "\n# - Final {} registration adjustment for t{}/t{} composed transformation:".format(trsf_type.upper(), time2index[t], time2index[time_steps[-1]])
            py_hl = 1  # defines highest level of the blockmatching-pyramid
            py_ll = 0  # defines lowest level of the blockmatching-pyramid
            print '  - t_{}h floating fname: {}'.format(t, float_img_fname)
            print '  - t_{}h reference fname: {}'.format(time_steps[-1], ref_img_fname)
            print '  - {} t_{}h/t_{}h composed-trsf as initialisation'.format(trsf_type, t, time_steps[-1])
            print ""
            res_trsf, res_im = registration(float_im, ref_im, method='{}_registration'.format(trsf_type), init_trsf=trsf, pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll)
            print ""

        # - Save result image and tranformation:
        print "Writing image file: {}".format(res_img_fname)
        imsave(res_path + res_img_fname, res_im)
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
            res_x_ch_fname = get_res_img_fname(x_ch_fname, time_steps[-1], t, trsf_type)
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
            res_seg_img_fname = get_res_img_fname(seg_img_fname, t_ref, t_float, trsf_type)
            if not exists(res_path + seg_img_fname) or force:
                print "  - {}\n  --> {}".format(seg_img_fname, res_seg_img_fname)
                # --- Read the segmented image file:
                seg_im = imread(image_dirname + seg_path_suffix + seg_img_fname)
                res_seg_im = apply_trsf(seg_im, res_trsf, param_str_2=' -nearest')
                # --- Apply and save registered segmented image:
                imsave(res_path + res_seg_img_fname, res_seg_im)
            else:
                print "  - existing file: {}".format(res_seg_img_fname)
        else:
            print "Could not find segmented image '{}'".format(seg_img_fname)



# - Initialise index dictionaries to keep track of images locations:
index2file = {}
index2time = {n: t for n, t in enumerate(time_steps)}
time2index = {v: k for k, v in index2time.items()}

# - Registration on PREVIOUS time-point (eg. to create lineage)
# temp_reg = [(14, 10), (10, 5), (5, 0)]
# - Registration on LAST time-point (eg. to have all image in same reference frame if trsf_type='rigid')
temp_reg = [(14, 10), (14, 5), (14, 0)]

for t_ref, t_float in temp_reg:
    # -- Get the reference CZI file names:
    ref_czi_fname = czi_base_fname.format(t_ref)
    # -- Get the reference INR file names:
    ref_path_suffix, ref_img_fname = get_nomenclature_channel_fname(ref_czi_fname, nomenclature_file, membrane_ch_name)

    # --- Get initial transformation matrix when time index difference is greater than 1:
    if time2index[t_ref] - time2index[t_float] > 1:
        print "\n\n# - Temporal registration of t{} on {}_t{}/{}...".format(t_float, trsf_type.upper(), index2time[time2index[t_float]+1], time_steps[-1])
        # --- Get t_float[tp+1] nomenclature image filename & path:
        ref_czi_fname = czi_base_fname.format(index2time[time2index[t_float]+1])
        ref_path_suffix, ref_img_fname = get_nomenclature_channel_fname(ref_czi_fname, nomenclature_file, membrane_ch_name)
        # --- Get t_ref / t_float[tp+1] estimated transformation to initiate registration:
        init_trsf_fname = get_res_trsf_fname(ref_img_fname, t_ref, index2time[time2index[t_float]+1], trsf_type)
        init_trsf_path = image_dirname + ref_path_suffix + trsf_type + "_registrations/"
        if exists(init_trsf_path + init_trsf_fname):
            print "Loading existing {} transformation file: {}".format(trsf_type.upper(), init_trsf_fname)
            init_trsf = bal_trsf.BalTransformation()
            init_trsf.read(init_trsf_path + init_trsf_fname)
            print "Done."
        else:
            raise IOError("Could not find {} transformation file: {}".format(trsf_type.upper(), init_trsf_path + init_trsf_fname))
        # --- Get t_ref / t_float[tp+1] registered image filename
        ref_path_suffix = ref_path_suffix + trsf_type + "_registrations/"
        ref_img_fname = get_res_img_fname(ref_img_fname, t_ref, index2time[time2index[t_float]+1], trsf_type)
    else:
        print "\n\n# - Temporal registration of t{} on t{}...".format(t_float, t_ref)
        init_trsf = None

    # -- Get the float CZI file names:
    float_czi_fname = czi_base_fname.format(t_float)
    # -- Get the float INR file names:
    float_path_suffix, float_img_fname = get_nomenclature_channel_fname(float_czi_fname, nomenclature_file, membrane_ch_name)

    # --- Get RESULT transformation matrix and image filename:
    res_path_suffix = float_path_suffix + trsf_type + "_registrations/"
    res_path = image_dirname + res_path_suffix
    if not exists(res_path):
        mkdir(res_path)
    res_trsf_fname = get_res_trsf_fname(float_img_fname, t_ref, t_float, trsf_type)
    res_img_fname = get_res_img_fname(float_img_fname, t_ref, t_float, trsf_type)

    if trsf_type == 'deformable':
        if not exists(image_dirname + float_path_suffix + 'rigid' + "_registrations/"):
            raise IOError("Could not find 'rigid_registration' folder, run with 'rigid' as trsf_type!")
        # --- Get 't_ref'-'t_float' rigid registered filenames for float images:
        float_path_suffix = float_path_suffix + 'rigid' + "_registrations/"
        float_img_fname = get_res_img_fname(float_img_fname, t_ref, t_float, 'rigid')
        # Remove rank-1 indication of rigid registration for shorted naming of res_trsf & res_img files:
        # vf_float_img_fname = float_img_fname.replace('-T{}_on_T{}'.format(t_float, index2time[time2index[t_float]+1]), '')

    # -- Add transformed filenames to the index dictionaries:
    # index2file[n+1] = res_path + res_img_fname

    # -- Skip estimation if associated *.trsf file exists and should not be overwritten:
    if not exists(res_path + res_trsf_fname) or force:
        # --- Performs REGISTRATION of t_float on t_ref:
        if time2index[t_ref] - time2index[t_float] > 1:
            print "\nEstimating {} transformation of t{} on {}_t{}/{}".format(trsf_type.upper(), t_float, trsf_type.upper(), index2time[time2index[t_float]+1], t_ref)
            print '  - t{} floating fname: {}'.format(t_float, float_img_fname)
            print '  - {}_t{}/{} reference fname: {}'.format(trsf_type, index2time[time2index[t_float]+1], t_ref, ref_img_fname)
            print '  - {}_t{}/{} init_trsf fname: {}'.format(trsf_type, index2time[time2index[t_float]+1], t_ref, init_trsf_fname)
        else:
            print "\nEstimating {} transformation of t{} on t{}".format(trsf_type.upper(), t_float, t_ref)
            print '  - t{} floating fname: {}'.format(t_float, float_img_fname)
            print '  - t{} reference fname: {}'.format(t_ref, ref_img_fname)
        py_hl = 3  # defines highest level of the blockmatching-pyramid
        py_ll = 0  # defines lowest level of the blockmatching-pyramid

        print ""
        print "Reading reference image filename: {}".format(ref_img_fname)
        ref_im = imread(image_dirname + ref_path_suffix + ref_img_fname)
        print "Reading floating image filename: {}".format(float_img_fname)
        float_im = imread(image_dirname + float_path_suffix + float_img_fname)
        print ""

        res_trsf, res_img = registration(float_im, ref_im, method=trsf_type+'_registration', init_trsf=init_trsf, pyramid_highest_level=py_hl, pyramid_lowest_level=py_ll)
        # --- Save estimated transformation matrix and registered image:
        print "Saving registered image: {}".format(res_img_fname)
        imsave(res_path + res_img_fname, res_img)
        print "Done."
        print "Saving transformation matrix: {}".format(res_trsf_fname)
        res_trsf.write(res_path + res_trsf_fname)
        print "Done."
    else:
        print "Loading existing {} transformation file: {}".format(trsf_type.upper(), res_trsf_fname)
        res_trsf = bal_trsf.BalTransformation()
        res_trsf.read(res_path + res_trsf_fname)
        print "Done."

    # -- Apply estimated transformation to other channels of the floating CZI:
    if extra_channels:
        print "\nApplying estimated {} transformation on '{}' to other channels: {}".format(trsf_type.upper(), membrane_ch_name, ', '.join(extra_channels))
        for x_ch_name in extra_channels:
            # --- Get the extra channel filenames:
            x_ch_path_suffix, x_ch_fname = get_nomenclature_channel_fname(float_czi_fname, nomenclature_file, x_ch_name)
            # --- Defines output filename & path:
            res_x_ch_fname = get_res_img_fname(x_ch_fname, t_ref, t_float, trsf_type)
            if not exists(res_path + res_x_ch_fname) or force:
                print "  - {}\n  --> {}".format(x_ch_fname, res_x_ch_fname)
                # --- Read the extra channel image file:
                x_ch_img = imread(image_dirname + x_ch_path_suffix + x_ch_fname)
                # --- Apply and save registered image:
                res_x_ch_img = apply_trsf(x_ch_img, res_trsf)
                imsave(res_path + res_x_ch_fname, res_x_ch_img)
            else:
                print "  - existing file: {}".format(res_x_ch_fname)

    # -- Apply estimated transformation to segmented image:
    f, e = splitext_zip(float_img_fname)
    seg_img_fname = f + "_segmented" + e
    if exists(image_dirname + float_path_suffix + seg_img_fname):
        print "\nApplying estimated {} transformation on '{}' to segmented image:".format(trsf_type.upper(), membrane_ch_name)
        f, e = splitext_zip(seg_img_fname)
        res_seg_img_fname = get_res_img_fname(f, t_ref, t_float, trsf_type)
        if not exists(res_path + seg_img_fname) or force:
            print "  - {}\n  --> {}".format(seg_img_fname, res_seg_img_fname)
            # --- Read the segmented image file:
            seg_img = imread(image_dirname + float_path_suffix + seg_img_fname)
            res_seg_im = apply_trsf(seg_img, res_trsf, param_str_2=' -linear')
            # --- Apply and save registered segmented image:
            imsave(res_path + res_seg_img_fname, res_seg_im)
        else:
            print "  - existing file: {}".format(res_seg_img_fname)
    else:
        print "Could not find segmented image '{}'".format(seg_img_fname)

# - Transform into generic name:
# open_file_index = {k: v.replace('_'+membrane_ch_name, '_{}') for k, v in index2file.items()}
# index = pd.DataFrame().from_dict({'time': index2time, 'trsf-type': {n: trsf_type for n in range(len(time_steps))}, 'filenames': open_file_index})
# index = index.sort_values('time')
# index.to_csv(image_dirname + base_fname + "-index.csv", index=False)

import numpy as np
from os.path import exists
from os.path import splitext
from os.path import split

from timagetk.io import imread
from timagetk.io import imsave

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
sys.path.append(SamMaps_dir+'/scripts/lib/')

from nomenclature import splitext_zip
from nomenclature import get_nomenclature_channel_fname
from nomenclature import get_nomenclature_segmentation_name
from nomenclature import get_res_img_fname
from nomenclature import get_res_trsf_fname
from equalization import z_slice_equalize_adapthist

XP = 'E35'
SAM = '6'
trsf_type = 'deformable'

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


def add_channel_regitration(czi_base_fname, nomenclature_file, ch_name, trsf_type, temp_reg, image_dirname, colormap='invert_grey'):
    """
    Add the registered image to tissuelab world.
    """
    for n, (t_ref, t_float) in enumerate(temp_reg):
        # -- Get the reference CZI file names:
        ref_czi_fname = czi_base_fname.format(t_ref)
        # -- Get the reference INR file names:
        ref_path_suffix, ref_img_fname = get_nomenclature_channel_fname(ref_czi_fname, nomenclature_file, ch_name)

        if n==0:
            print "Reading last time-point (t{}) image filename: {}".format(t_ref, ref_img_fname)
            ref_im = imread(image_dirname + ref_path_suffix + ref_img_fname)
            if ch_name.find('raw') != -1:
                ref_im = z_slice_equalize_adapthist(ref_im)
            world.add(ref_im, ref_img_fname, colormap='Reds')

        # -- Get the float CZI file names:
        float_czi_fname = czi_base_fname.format(t_float)
        # -- Get the float INR file names:
        float_path_suffix, float_img_fname = get_nomenclature_channel_fname(float_czi_fname, nomenclature_file, ch_name)
        # --- Get RESULT rigid transformation image filename:
        res_path_suffix = float_path_suffix + trsf_type + "_registrations/"
        res_img_fname = get_res_img_fname(float_img_fname, t_ref, t_float, trsf_type)

        # -- Read RESULT image file:
        res_im = imread(image_dirname + res_path_suffix + res_img_fname)
        print "Reading result floating (t{}) image filename: {}".format(t_float, res_img_fname)
        world.add(res_im, res_img_fname, colormap='Greens')
    return "Done."


temp_reg = [(14, 10), (14, 5), (14, 0)]
# add_channel_regitration(czi_base_fname, nomenclature_file, 'PI', 'rigid', temp_reg, image_dirname)
# add_channel_regitration(czi_base_fname, nomenclature_file, 'PI', 'deformable', temp_reg, image_dirname)
add_channel_regitration(czi_base_fname, nomenclature_file, 'PI_raw', 'deformable', temp_reg, image_dirname)

add_channel_regitration(czi_base_fname, nomenclature_file, 'PIN1', 'deformable', temp_reg, image_dirname)
