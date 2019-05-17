import numpy as np
from os.path import splitext
from timagetk.io import imread, imsave
from timagetk.algorithms.exposure import z_slice_contrast_stretch
from timagetk.algorithms.exposure import z_slice_equalize_adapthist
from timagetk.algorithms.resample import isometric_resampling
from timagetk.plugins import linear_filtering
from timagetk.plugins import auto_seeded_watershed
from timagetk.visu.mplt import profile_hmin
from os.path import join

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    base_dir = '/data/Meristems/Bihai/20180616_3-global_della_Ler_LD/'
elif platform.uname()[1] == "calculus":
    base_dir = '/projects/SamGrowth/microscopy/20180616_global_della_Ler_LD'
else:
    raise ValueError("Unknown custom path to 'base_dir' for this system...")

fname = '20180616 3-global della_Ler_LD +++.lsm'

base_fname, ext = splitext(fname)
image = imread(base_dir + fname)
image.voxelsize

iso_image = isometric_resampling(image, method='min', option='cspline')
iso_image.shape

iso_image = z_slice_equalize_adapthist(iso_image)
# iso_image = z_slice_contrast_stretch(iso_image, pc_min=1.5)
iso_image = linear_filtering(iso_image, method="gaussian_smoothing", sigma=0.25, real=True)

xsh, ysh, zsh = iso_image.shape
mid_x, mid_y, mid_z = int(xsh/2.), int(ysh/2.), int(zsh/2.)

selected_hmin = profile_hmin(iso_image, x=mid_x, z=mid_z, plane='x', zone=mid_z)

seg_im = auto_seeded_watershed(iso_image, selected_hmin, control='most')
np.unique(seg_im)

seg_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.tif')
imsave(base_dir + seg_fname, seg_im)

from timagetk.visu.mplt import gif_slice_blending
gif_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.gif')
gif_slice_blending(seg_im, iso_image, base_dir + gif_fname, fps=40)


for selected_hmin in [4000, 5000, 6000]:
    seg_im = auto_seeded_watershed(iso_image, selected_hmin, control='most')
    seg_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.tif')
    imsave(base_dir + seg_fname, seg_im)

import timagetk.visu.mplt
reload(timagetk.visu.mplt)
from timagetk.visu.mplt import gif_slice_blending
for selected_hmin in [4000, 5000, 6000]:
    seg_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.tif')
    seg_im = imread(base_dir + seg_fname)
    gif_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.gif')
    gif_slice_blending(seg_im, iso_image, base_dir+gif_fname, duration=10., resize=[300, 300], invert=True)
