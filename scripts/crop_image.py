# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon - INRIA
#
#       File author(s): Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
################################################################################
"""
Allows to crop (single channel) image(s) around a given bounding box (ie. start
& stop values along defined matrix dimensions).

Selection of output format is possible between 'inr' & 'tif'.
"""

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/lib')

from nomenclature import splitext_zip
from segmentation_pipeline import read_image
from timagetk.components import imsave
from timagetk.components import SpatialImage

from vplants.tissue_analysis.signal_quantification import get_infos
from vplants.tissue_analysis.signal_quantification import crop_image

import argparse
parser = argparse.ArgumentParser(description='Crop an image given boundaries.')
# positional arguments:
parser.add_argument('im2crop', type=str, nargs='+',
                    help="file or list of files containing image(s) to crop.")

POSS_FMT = ["inr", "tif"]
# optional arguments:
parser.add_argument('--x_bound', type=int, nargs=2, default=[0, -1],
                    help="lower and upper limit for the x-axis, starts at '0', ends at '-1'")
parser.add_argument('--y_bound', type=int, nargs=2, default=[0, -1],
                    help="lower and upper limit for the y-axis, starts at '0', ends at '-1'")
parser.add_argument('--z_bound', type=int, nargs=2, default=[0, -1],
                    help="lower and upper limit for the z-axis, starts at '0', ends at '-1'")
parser.add_argument('--out_fmt', type=str, default='inr',
                    help="format of the file to write, accepted formats are: {}.".format(POSS_FMT))

args = parser.parse_args()

# - Variables definition from mandatory arguments parsing:
# -- Image to crop:
im2crop_fnames = args.im2crop
if isinstance(im2crop_fnames, str):
    im2crop_fnames = [im2crop_fnames]

# - Variables definition from optional arguments parsing:
# -- Lower and upper boundaries on each axis:
x_min, x_max = args.x_bound
y_min, y_max = args.y_bound
z_min, z_max = args.z_bound
axis = ['x', 'y', 'z']
# -- output format:
ext = args.out_fmt
try:
    assert ext in POSS_FMT
except AssertionError:
    raise ValueError("Unknown format '{}', availables are: {}".format(ext, POSS_FMT))

# -- Check all images have same shape and voxelsize (otherwise you don't know what you are doing!)
shape_list = []
vxs_list = []
ndim_list = []
for im2crop_fname in im2crop_fnames:
    # TODO: would be nice to access ONLY the file header to get those info:
    im2crop = read_image(im2crop_fname)
    shape_list.append(im2crop.get_shape())
    vxs_list.append(im2crop.get_voxelsize())
    ndim_list.append(im2crop.get_dim())

# - Check dimensionality:
ndim = ndim_list[0]
try:
    assert all([nd == ndim for nd in ndim_list])
except AssertionError:
    raise ValueError("Images do not have the same dimensionality!")

axis = axis[:ndim]
if ndim == 2:
    lower_bounds = [x_min, y_min]
    upper_bounds = [x_max, y_max]
else:
    lower_bounds = [x_min, y_min, z_min]
    upper_bounds = [x_max, y_max, z_max]

# - Defining which directions are cropped and should be checked for compatible 'shape' and 'voxelsize'
axis2check = [False] * ndim
for n in range(ndim):
    if lower_bounds[n] != 0 or upper_bounds[n] != -1:
        axis2check[n] = True

# - Check the 'shape' and 'voxelsize' for cropped axes:
for n, ax in enumerate(axis2check):
    if ax:
        ref_sh = shape_list[0][n]
        try:
            assert all([sh[n] == ref_sh for sh in shape_list])
        except:
            raise ValueError("Shape missmatch along axis {} ({}) among list of images!".format(axis[n], n))
        ref_vxs = vxs_list[0][n]
        try:
            assert all([vxs[n] == ref_vxs for vxs in vxs_list])
        except:
            raise ValueError("Voxelsize missmatch along axis {} ({}) among list of images!".format(axis[n], n))

for im2crop_fname in im2crop_fnames:
    print "\n\n# - Reading image file {}...".format(im2crop_fname)
    im2crop = read_image(im2crop_fname)
    print "Done."
    # - Get original image infos:
    shape, ori, vxs, md = get_infos(im2crop)
    print "\nGot original shape: {}".format(shape)
    print "Got original voxelsize: {}".format(vxs)
    # - Crop the image:
    bounding_box = []
    for n in range(ndim):
        bounding_box.extend([lower_bounds[n], upper_bounds[n]])
    im = crop_image(im2crop, bounding_box)
    # - Create output filename:
    out_fname = splitext_zip(im2crop_fname)[0]
    for n, ax in enumerate(axis):
        if lower_bounds[n] != 0  or upper_bounds[n] != -1:
            out_fname += '-{}{}_{}'.format(ax, lower_bounds[n], upper_bounds[n])

    out_fname += '.' + ext
    print "\nSaving file: '{}'".format(out_fname)
    # - Save the cropped-image:
    imsave(out_fname, im)
