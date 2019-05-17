import numpy as np
from timagetk.util import data_path
from timagetk.io import imread
from timagetk.algorithms.exposure import z_slice_contrast_stretch
from timagetk.algorithms.resample import isometric_resampling
from timagetk.plugins import linear_filtering
from timagetk.plugins.segmentation import auto_seeded_watershed
from timagetk.visu.mplt import profile_hmin

from skimage import img_as_float
from skimage.color import label2rgb, gray2rgb

int_img = imread(data_path('p58-t0-a0.lsm'))
print int_img.shape
print int_img.voxelsize

int_img = isometric_resampling(int_img, method='min', option='cspline')
int_img = z_slice_contrast_stretch(int_img, pc_min=1)
int_img = linear_filtering(int_img, method="gaussian_smoothing", sigma=0.25, real=True)

xsh, ysh, zsh = int_img.shape
mid_x, mid_y, mid_z = int(xsh/2.), int(ysh/2.), int(zsh/2.)

h_min = profile_hmin(int_img, x=mid_x, z=mid_z, plane='x', zone=mid_z)

seg_img = auto_seeded_watershed(int_img, hmin=h_min)
print seg_img.shape

label_rgb = label2rgb(seg_img, alpha=1, bg_label=1, bg_color=(0,0,0))
print label_rgb.shape

image_rgb = gray2rgb(img_as_float(int_img))
print image_rgb.shape

# alpha = 0.3
# blend = label_rgb * alpha + image_rgb * (1 - alpha)
#
# from timagetk.visu.mplt import xyz_array_browser
# xyz_array_browser(blend)

import timagetk.visu.mplt
reload(timagetk.visu.mplt)
from timagetk.visu.mplt import gif_slice_blending
gif_fname = "p58-t0-a0-seg_hmin{}{}".format(h_min, '.gif')
gif_slice_blending(seg_img, int_img, data_path(gif_fname), duration=2.)
