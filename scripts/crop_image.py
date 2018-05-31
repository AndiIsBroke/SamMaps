import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/TissueLab/')

from nomenclature import splitext_zip
from segmentation_pipeline import read_image
from timagetk.components import imsave
from timagetk.components import SpatialImage

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


################################################################################
#### EXTRA functions:
################################################################################
def get_infos(im):
    """
    Returns shape, origin, voxelsize and metadata dictionary from given SpatialImage
    """
    shape = im.get_shape()
    ori = im.get_origin()
    vxs = im.get_voxelsize()
    md = im.get_metadata()
    return shape, ori, vxs, md

################################################################################
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
for im2crop_fname in im2crop_fnames:
    im2crop = read_image(im2crop_fname)
    shape_list.append(im2crop.get_shape())
    vxs_list.append(im2crop.get_voxelsize())


for im2crop_fname in im2crop_fnames:
    print "\n\n# - Reading image file {}...".format(im2crop_fname)
    im2crop = read_image(im2crop_fname)
    print "Done."
    # - Get original image infos:
    shape, ori, vxs, md = get_infos(im2crop)
    print "\nGot original shape: {}".format(shape)
    print "Got original voxelsize: {}".format(vxs)
    # - Define lower and upper bounds:
    lower_bounds = [x_min, y_min, z_min]
    upper_bounds = [x_max, y_max, z_max]
    # -- Change '-1' into max shape value for given axis:
    for n, upper_bound in enumerate(upper_bounds):
        if upper_bound == -1:
            upper_bounds[n] = shape[n] - 1
    # -- Update values:
    x_max, y_max, z_max = upper_bounds

    # - Check lower and upper bound values are consistent with the shape of the image:
    for n, lower_bound in enumerate(lower_bounds):
        try:
            assert lower_bound >= 0
        except AssertionError:
            raise ValueError("Lower bound '{}' ({}) is strictly inferior to zero, please check!".format(axis[n], lower_bound))

    for n, upper_bound in enumerate(upper_bounds):
        try:
            assert upper_bound <= shape[n]
        except AssertionError:
            raise ValueError("Upper bound '{}' ({}) is greater than the max shape ({}), please check!".format(axis[n], upper_bound, shape[n]))

    # - Crop the image:
    print "\n# - Cropping image at [{}:{}, {}:{}, {}:{}]".format(x_min, x_max, y_min, y_max, z_min, z_max)
    im = im2crop[x_min:x_max, y_min:y_max, z_min:z_max]
    im = SpatialImage(im, voxelsize=vxs, origin=ori, metadata_dict=md)

    # - Create output filename:
    out_fname = splitext_zip(im2crop_fname)[0]
    for n, ax in enumerate(axis):
        if lower_bounds[n] != 0  or upper_bounds[n] != shape[n]-1:
            out_fname += '-{}{}_{}'.format(ax, lower_bounds[n], upper_bounds[n])

    out_fname += '.' + ext
    print "\nSaving file: '{}'".format(out_fname)
    # - Save the cropped-image:
    imsave(out_fname, im)
