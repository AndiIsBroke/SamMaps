import numpy as np
from os.path import splitext
from timagetk.io import imread, imsave
from timagetk.algorithms.exposure import global_contrast_stretch
from timagetk.algorithms.resample import isometric_resampling
from timagetk.plugins import morphology
from timagetk.plugins import linear_filtering
from timagetk.plugins import auto_seeded_watershed
from timagetk.visu.mplt import profile_hmin
from os.path import join
from timagetk.visu.mplt import gif_slice_blending

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    base_dir = '/data/Yoan/'
else:
    raise ValueError("Unknown custom path to 'base_dir' for this system...")

fname = 'B09-myr-YFP-4-C1.tif'
connect = None
control = 'most'

base_fname, ext = splitext(fname)
image = imread(base_dir + fname)
image.voxelsize

iso_image = isometric_resampling(image, method='min', option='cspline')
iso_image.shape

iso_image = global_contrast_stretch(iso_image)
# iso_image = z_slice_contrast_stretch(iso_image, pc_min=1.5)
smooth_image = linear_filtering(iso_image, method="gaussian_smoothing", sigma=1., real=True)

asf_image = morphology(iso_image, method="oc_alternate_sequential_filter", max_radius=1)


xsh, ysh, zsh = iso_image.shape
mid_x, mid_y, mid_z = int(xsh/2.), int(ysh/2.), int(zsh/2.)

from timagetk.visu.mplt import grayscale_imshow
from timagetk.visu.stack import stack_browser
grayscale_imshow([iso_image, smooth_image, asf_image], slice_id=mid_z)

stack_browser(asf_image)

selected_hmin = profile_hmin(smooth_image, x=mid_x, z=mid_z, plane='x', zone=mid_z)
selected_hmin = profile_hmin(asf_image, x=mid_x, z=mid_z, plane='x', zone=mid_z)


from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling
from timagetk.plugins import seeded_watershed

ext_img = h_transform(asf_image, h=selected_hmin, method='min', connectivity=connect)
if ext_img is None:
    raise ValueError(
        "Function 'h_transform' returned a NULL image, aborting automatic segmentation!")

seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=selected_hmin,
                           method='connected_components',
                           connectivity=connect)
if seed_img is None:
    raise ValueError(
        "Function 'region_labeling' returned a NULL image, aborting automatic segmentation!")

n_seeds = len(np.unique(seed_img)) - 1  # '0' is in the list!
print("Detected {} seeds!".format(n_seeds))

from timagetk.util import _method_check
from timagetk.plugins.segmentation import WATERSHED_CONTROLS, DEFAULT_CONTROL
control = _method_check(control, WATERSHED_CONTROLS, DEFAULT_CONTROL)
params = '-labelchoice ' + str(control)

seg_img = seeded_watershed(asf_image, seed_img, param_str_2=params)
if seg_img is None:
    raise ValueError(
        "Function 'watershed' returned a NULL image, aborting automatic segmentation!")

from timagetk.visu.util import label_blending
from timagetk.visu.stack import rgb_stack_browser
vx, vy, vz = iso_image.voxelsize
blend = label_blending(seg_img, iso_image)
rgb_stack_browser(blend, "Intensity & segmentation blending", xy_ratio=vx/vy)


gif_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.gif')
gif_slice_blending(seg_im, iso_image, base_dir + gif_fname, duration=10.)

seg_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.tif')
imsave(base_dir + seg_fname, seg_im)



for hmin in [20, 30, 40]:
    ext_img = h_transform(image, h=hmin, method='min', connectivity=connect)
    if ext_img is None:
        raise ValueError(
            "Function 'h_transform' returned a NULL image, aborting automatic segmentation!")

    seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=hmin,
                               method='connected_components',
                               connectivity=connect)
    if seed_img is None:
        raise ValueError(
            "Function 'region_labeling' returned a NULL image, aborting automatic segmentation!")

    n_seeds = len(np.unique(seed_img)) - 1  # '0' is in the list!
    print("Detected {} seeds!".format(n_seeds))

    seg_img = watershed(image, seed_img, param_str_2=params)
    if seg_img is None:
        raise ValueError(
            "Function 'watershed' returned a NULL image, aborting automatic segmentation!")

    seg_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.tif')
    imsave(base_dir + seg_fname, seg_im)
    gif_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.gif')
    gif_slice_blending(seg_im, iso_image, base_dir+gif_fname, duration=10., out_shape=[400, 400])
