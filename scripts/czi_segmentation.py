# -*- coding: utf-8 -*-
import argparse
from os.path import exists

from timagetk.components import imsave


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

from segmentation_pipeline import seg_pipe
from segmentation_pipeline import read_image
from segmentation_pipeline import segmentation_fname

# - DEFAULT variables:
# Reference channel name used to compute tranformation matrix:
DEF_MEMB_CH = 'PI'
# CZI list of channel names:
DEF_CH_NAMES = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
# Microscope orientation:
DEF_ORIENT = -1  # '-1' == inverted microscope!
# Minimal volume threshold for cells, used to avoid too small cell from seed over-detection
DEF_MIN_VOL = 5.
# Background value: (not handled by parser)
back_id = 1
# Default smoothing factor for Gaussian smoothing (linear_filtering):
DEF_STD_DEV = 1.0

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
parser.add_argument('--std_dev', type=float, default=DEF_STD_DEV,
                    help="standard deviation used for Gaussian smoothing, '{}' by default".format(DEF_STD_DEV))
parser.add_argument('--min_cell_volume', type=float, default=DEF_MIN_VOL,
                    help="minimal volume accepted for a cell, '{}' by default".format(DEF_MIN_VOL))
parser.add_argument('--substract_ch_name', type=str, default="",
                    help="if specified, substract this channel from the 'seg_ch_name' before segmentation, None by default")
parser.add_argument('--output_fname', type=str, default="",
                    help="if specified, the filename of the labbeled image, by default automatic naming contains some infos about the procedure")

parser.add_argument('--iso', action='store_true',
                    help="if given, performs resampling to isometric voxelsize before segmentation, 'False' by default")
parser.add_argument('--equalize', action='store_true',
                    help="if given, performs contrast strectching of the intensity image to segment, 'False' by default")
parser.add_argument('--force', action='store_true',
                    help="if given, force computation of labelled image even if it already exists, 'False' by default")

args = parser.parse_args()

def exists_file(f):
    try:
        assert exists(f)
    except AssertionError:
        raise IOError("This file does not exixts: {}".format(f))
    else:
        print "Found file {}".format(f)
    return

# - Variables definition from argument parsing:
czi_fname = args.czi
exists_file(czi_fname)
h_min = args.h_min
# - Variables definition from optional arguments:
seg_ch_name = args.seg_ch_name
print "Got '{}' as the reference channel name.".format(seg_ch_name)
channel_names = args.channel_names
print "Got '{}' as list of CZI channel names.".format(channel_names)
substract_ch_name = args.substract_ch_name
if substract_ch_name != "":
    print "Will performs channel substraction '{}'-'{}' before segmentation.".format(seg_ch_name, substract_chs_name)

min_cell_volume = args.min_cell_volume
try:
    assert min_cell_volume >= 0.
except AssertionError:
    raise ValueError("Negative minimal volume!")

std_dev = args.std_dev
iso = args.iso
equalize = args.equalize
output_fname =  args.output_fname

force =  args.force
if force:
    print "WARNING: any existing segmentation image will be overwritten!"
else:
    print "Existing segmentation will be kept!"

if output_fname != "":
    seg_img_fname = output_fname
else:
    seg_img_fname = segmentation_fname(czi_fname, h_min, iso, equalize)
print seg_img_fname

if exists(seg_img_fname) and not force:
    print "Found existing segmentation file: {}".format(seg_img_fname)
    print "ABORT!"
else:
    czi_im = read_image(czi_fname, channel_names)
    im2seg = czi_im[seg_ch_name]
    if substract_ch_name != "":
        im2sub = czi_im[substract_ch_name]
    else:
        im2sub = None
    seg_im = seg_pipe(im2seg, h_min, im2sub, iso, equalize, std_dev, min_cell_volume, back_id)
    print "\n - Saving segmentation under '{}'".format(seg_img_fname)
    imsave(seg_img_fname, seg_im)
