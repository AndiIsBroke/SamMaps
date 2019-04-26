# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2019 CNRS - ENS Lyon - INRIA
#
#       File author(s): Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
################################################################################
"""
Allows to convert (single channel) LSM file.

Selection of output format is possible between 'inr', 'tif' & 'mha'.

Examples
--------
$ ipython lsm_convert "PIN1-CFP-Ler-E99-LD-SAM3.lsm" --out_fmt tif
"""

import argparse
from os.path import exists
from os.path import splitext
from timagetk.io import imread
from timagetk.io import imsave

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

from nomenclature import exists_file

SUPPORTED_FMT = ['inr', 'tif', 'mha']

# PARAMETERS:
# -----------
parser = argparse.ArgumentParser(description='Convert single channel LSM files to given format.')
# positional arguments:
parser.add_argument('lsm', type=str,
                    help="filename of the (single-channel) LSM to convert.")
# optional arguments:
parser.add_argument('--out_fmt', type=str, default='inr',
                    help="format of the file(s) to write.")
parser.add_argument('--output_fname', type=str, default="",
                    help="if specified, the base prefix filename of the exported image, (automatic naming by default)")
parser.add_argument('--force', action='store_true',
                    help="if given, force computation of labelled image even if it already exists, 'False' by default")


args = parser.parse_args()
# - Variables definition from argument parsing:
lsm_fname = args.lsm
exists_file(lsm_fname)

# - Variables definition from optional arguments:
# -- Output format:
out_fmt = args.out_fmt.split('.')[-1]
try:
    assert out_fmt in SUPPORTED_FMT
except AssertionError:
    raise TypeError("Unkown output file format '{}', supported formats are: '{}'.".format(out_fmt, SUPPORTED_FMT))

# -- Force overwrite existing file:
force =  args.force
if force:
    print "WARNING: any existing image will be overwritten!"
else:
    print "Existing images will be kept!"

# - Loading the LSM to convert:
lsm_im = imread(lsm_fname)

# -- Make the base filename to output (channel name & exrtension added later!)
output_fname =  args.output_fname
if output_fname == "":
    output_fname = splitext(lsm_fname)[0]

# - Save to given format:
img_fname = output_fname + '.{}'.format(out_fmt)
if exists(img_fname) and not force:
    print "Found existing file: {}".format(img_fname)
    print "Use '--force' argument to overwrite it!"
else:
    print "Saving to '{}' format under: '{}'".format(out_fmt.upper() ,img_fname)
    imsave(img_fname, lsm_im)
