# -*- coding: utf-8 -*-
import argparse
import copy as cp
import numpy as np
import pandas as pd

from os.path import exists

from timagetk.components import imread
from timagetk.components import imsave
from timagetk.algorithms import isometric_resampling
from timagetk.plugins import morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling
from timagetk.plugins import linear_filtering
from timagetk.plugins import segmentation

from openalea.tissue_nukem_3d.microscopy_images import imread as read_czi

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

from equalization import z_slice_contrast_stretch
from nomenclature import splitext_zip
# Nomenclature file location:
nomenclature_file = SamMaps_dir + "nomenclature.csv"
# OUTPUT directory:
image_dirname = dirname + "nuclei_images/"

# - DEFAULT variables:
# Reference channel name used to compute tranformation matrix:
DEF_MEMB_CH = 'PI'
# CZI list of channel names:
DEF_CH_NAMES = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
# Microscope orientation:
DEF_ORIENT = -1  # '-1' == inverted microscope!
# Minimal volume threshold for cells, used to avoid too small cell from seed over-detection
DEF_MIN_VOL = 50.
# Background value: (not handled by parser)
back_id = 1

# PARAMETERS:
# -----------
parser = argparse.ArgumentParser(description='Consecutive backward registration.')
# positional arguments:
parser.add_argument('czi', type=str,
                    help="filename of the (multi-channel) CZI to segment.")
parser.add_argument('h_min', type=int,
                    help="value to use for minimal h-transform extraction.")
# optional arguments:
parser.add_argument('--seg_ch_name', type=str, default=DEF_MEMB_CH,
                    help="channel name containing intensity image to segment, '{}' by default".format(DEF_MEMB_CH))
parser.add_argument('--channel_names', type=str, nargs='+', default=DEF_CH_NAMES,
                    help="list of channel names found in the given CZI, '{}' by default".format(DEF_CH_NAMES))
parser.add_argument('--microscope_orientation', type=int, default=DEF_ORIENT,
                    help="orientation of the microscope (i.e. set '-1' when using an inverted microscope), '{}' by default".format(DEF_ORIENT))
parser.add_argument('--min_cell_volume', type=float, default=DEF_MIN_VOL,
                    help="minimal volume accepted for a cell, '{}' by default".format(DEF_ORIENT))
parser.add_argument('--substract_ch_name', type=str, default="",
                    help="if specified, substract this channel from the 'seg_ch_name' before segmentation, None by default")
parser.add_argument('--output_fname', type=str, default="",
                    help="if specified, the filename of the labbeled image, by default automatic naming contains some infos about the procedure")

parser.add_argument('--iso', action='store_true',
                    help="if given, performs resampling to isometric voxelsize before segmentation, 'False' by default")
parser.add_argument('--equalize', action='store_true',
                    help="if given, performs contrast strectching of the intensity image to segment, 'False' by default")
parser.add_argument('--force', action='store_true',
                    help="if given, force computation of values even if the *.CSV already exists, else skip it, 'False' by default")

args = parser.parse_args()

# - Variables definition from argument parsing:
czi_fname = args.czi
h_min = args.h_min
# - Variables definition from optional arguments:
seg_ch_name = args.seg_ch_name
print "Got '{}' as the reference channel name.".format(seg_ch_name)
channel_names = args.channel_names
print "Got '{}' as list of CZI channel names.".format(channel_names)
substract_ch_name = args.substract_ch_name
if substract_ch_name != "":
    print "Will performs channel substraction '{}'-'{}' before segmentation.".format(seg_ch_name, substract_ch_name)

min_cell_volume = args.min_cell_volume
try:
    assert min_cell_volume >= 0.
else:
    raise ValueError("Negative minimal volume!")

iso = args.iso
equalize = args.equalize

force =  args.force
if force:
    print "WARNING: any existing segmentation image will be overwritten!"
else:
    print "Existing segmentation will be kept!"

czi = read_czi(czi_fname)
im2seg = czi[seg_ch_name]
# try:
#     voxelsize = im2seg.voxelsize
# except AttributeError:
#     try:
#         voxelsize = im2seg.resolution
#     except AttributeError:
#         raise AttributeError("Could not find image resolution!")
# try:
#     ori = im2seg.get_origin()
# except:
#     ori = None

# TODO: add unused parameters, in the filename, to the SpatialImage metadata!
if output_fname:
    seg_img_fname = output_fname
else:
    suffix = '_seg'.format(h_min)
    suffix += '_iso' if iso else ''
    suffix += '_eq' if equalize else ''
    suffix += '_hmin{}'.format(h_min)
    seg_img_fname = splitext_zip(czi_fname)[0] + suffix + '.inr'

if exists(seg_img_fname) and not force:
    print "Found existing segmentation file: {}".format(seg_img_fname)
    print "ABORT!"
else:
    if iso:
        print "\n - Performing isometric resampling of the intensity image to segment..."
        img2seg = isometric_resampling(im2seg)

    if equalize:
        print "\n - Performing histogram contrast stretching of the intensity image to segment..."
        img2seg = z_slice_contrast_stretch(img2seg)

    if substract_ch_name != "":
        print "\n - Performing '{}'-'{}' signal substraction...".format(seg_ch_name, substract_ch_name)
        vxs = img2seg.get_voxelsize()
        im2sub = czi[substract_ch_name]
        if iso:
            im2sub = isometric_resampling(im2sub)
        # im2sub = morphology(im2sub, method='erosion', radius=3., iterations=3)
        tmp_im = img2seg - im2sub
        tmp_im[img2seg <= im2sub] = 0
        vxs = im2seg.get_voxelsize()
        ori = im2seg.origin()
        img2seg = SpatialImage(tmp_im, voxelsize=vxs, origin=ori)
        del tmp_im

    print "\n# - Automatic seed detection..."
    std_dev = 1.0
    # morpho_radius = 1.0
    # asf_img = morphology(img2seg, max_radius=morpho_radius, method='co_alternate_sequential_filter')
    # ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
    smooth_img = linear_filtering(img2seg, std_dev=std_dev, method='gaussian_smoothing')
    # ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
    ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')
    seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
    print "Detected {} seeds!".format(len(np.unique(seed_img)))

    print "\n - Performing seeded watershed segmentation..."
    seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed', try_plugin=False)
    seg_im[seg_im == 0] = back_id

    imsave(seg_img_fname, seg_im)
