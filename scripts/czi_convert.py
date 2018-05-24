# -*- coding: utf-8 -*-
import argparse
from os.path import exists, splitext

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

from segmentation_pipeline import read_image


SUPPORTED_FMT = ['inr', 'tif', 'tiff']

# PARAMETERS:
# -----------
parser = argparse.ArgumentParser(description='Consecutive backward registration.')
# positional arguments:
parser.add_argument('czi', type=str,
                    help="filename of the (multi-channel) CZI to convert.")
# optional arguments:
parser.add_argument('--out_fmt', type=str, default='inr',
                    help="format of the file(s) to write.")
parser.add_argument('--channel_names', type=str, nargs='+', default="",
                    help="list of channel names found in the given CZI, '{}' by default".format(DEF_CH_NAMES))
parser.add_argument('--out_channels', type=str, nargs='+', default="",
                    help="list of channel names to extract from the CZI, 'all' by default")
parser.add_argument('--output_fname', type=str, default="",
                    help="if specified, the base prefix filename of the exported image, (automatic naming by default)")


def exists_file(f):
    try:
        assert exists(f)
    except AssertionError:
        raise IOError("This file does not exixts: {}".format(f))
    else:
        print "Found file {}".format(f)
    return


args = parser.parse_args()
# - Variables definition from argument parsing:
czi_fname = args.czi
exists_file(czi_fname)

# - Variables definition from optional arguments:
out_fmt = args.out_fmt.split('.')[-1]
try:
    assert out_fmt in SUPPORTED_FMT
except AssertionError:
    raise TypeError("Unkown output file format '{}', supported formats are: '{}'.".format(out_fmt, SUPPORTED_FMT))

czi_im = read_image(czi_fname, channel_names)
nb_ch = len(czi_im)

channel_names = args.channel_names
if channel_names == '':
    print "Got no list of channel names."
    channel_names = map(str, range(0, nb_ch))
else:
    print "Got '{}' as list of channel names.".format(channel_names)

out_channels = args.out_channels
if out_channels == "":
    print "All {} channels will be extraxted".format(nb_ch)
    out_channels = channel_names
else:
    print "Only the following channels will be extracted: '{}'".format(out_channels)

force =  args.force
if force:
    print "WARNING: any existing image will be overwritten!"
else:
    print "Existing images will be kept!"

output_fname =  args.output_fname
if output_fname == "":
    output_fname = splitext(czi_fname)[0]
print "Base filename used to export channels is: '{}'".format(output_fname)

for n_ch in range(0, nb_ch):
    img_fname = output_fname + '_{}.{}'.format(output_fname, out_fmt)
    if exists(img_fname) and not force:
        print "Found existing file: {}".format(img_fname)
else:
    im = czi_im[n_ch]
    print "\n - Saving channel #{} ({}) under: '{}'".format(img_fname)
    imsave(img_fname, im)
