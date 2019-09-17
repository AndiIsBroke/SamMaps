import numpy as np
from os.path import splitext
from timagetk.io import imread, imsave
from timagetk.algorithms.exposure import global_contrast_stretch
from timagetk.algorithms.exposure import z_slice_equalize_adapthist
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

fname = 'B09-myr-YFP-7-C1.tif'
connect = None
control = 'most'

base_fname, ext = splitext(fname)
image = imread(base_dir + fname)
image.voxelsize

iso_image = isometric_resampling(image, method='min', option='cspline')
iso_image.shape

st_img1 = global_contrast_stretch(iso_image)
st_img2 = z_slice_equalize_adapthist(iso_image)

from timagetk.visu.mplt import grayscale_imshow
grayscale_imshow([iso_image, st_img1, st_img2], 100, title=["Original", "global_contrast_stretch", "z_slice_equalize_adapthist"])


seed_detect_img = morphology(st_img2, "erosion", radius=1)

seed_detect_img = morphology(seed_detect_img, "coc_alternate_sequential_filter", max_radius=1, iterations=2)
seed_detect_img = linear_filtering(seed_detect_img, method="gaussian_smoothing", sigma=1., real=True)

xsh, ysh, zsh = iso_image.shape
mid_x, mid_y, mid_z = int(xsh/2.), int(ysh/2.), int(zsh/2.)
selected_hmin = profile_hmin(seed_detect_img, x=mid_x, z=mid_z, plane='z', zone=200)

for hmin in [20, 30, 40]:
    seg_im = auto_seeded_watershed(seed_detect_img, hmin, control='most')
    # Save the segmented image:
    seg_fname = base_fname+"-seg_hmin{}{}".format(hmin, ext)
    imsave(base_dir + seg_fname, seg_im)
    # Segmentation & intensity blending animation:
    from timagetk.visu.mplt import gif_slice_blending
    gif_fname = base_fname+"-seg_hmin{}{}".format(hmin, '.gif')
    gif_slice_blending(seg_im, seed_detect_img, base_dir+gif_fname, duration=20., out_shape=[512, 512])
