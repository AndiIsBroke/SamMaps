import numpy as np
from os.path import splitext
from timagetk.io import imread, imsave
from timagetk.algorithms.exposure import z_slice_contrast_stretch
from timagetk.algorithms.resample import isometric_resampling
from timagetk.plugins import linear_filtering
from timagetk.plugins import auto_seeded_watershed
from timagetk.visu.mplt import profile_hmin

import platform
if platform.uname()[1] == "RDP-M7520-JL":
    base_dir = '/data/Meristems/Carlos/SuperResolution/'
elif platform.uname()[1] == "calculus":
    base_dir = '/projects/SamMaps/SuperResolution/LSM Airyscan/'
else:
    raise ValueError("Unknown custom path to 'base_dir' for this system...")

fname = 'SAM1-gfp-pi-stack-LSM800-Airyscan Processing-high_PI_conf_z-stack_reg.tif'
base_fname, ext = splitext(fname)
image = imread(base_dir + fname)

iso_image = isometric_resampling(image, method='min', option='cspline')
iso_image.shape

iso_image = z_slice_contrast_stretch(iso_image, pc_min=1.5)
iso_image = linear_filtering(iso_image, method="gaussian_smoothing", sigma=0.1, real=True)

# xsh, ysh, zsh = iso_image.shape
# mid_x, mid_y, mid_z = int(xsh/2.), int(ysh/2.), int(zsh/2.)
#
# selected_hmin = profile_hmin(iso_image, x=mid_x, z=mid_z, plane='x', zone=mid_z)

for selected_hmin in np.arange(4000, 6000, 500):
    seg_im = auto_seeded_watershed(iso_image, selected_hmin, control='most')
    # np.unique(seg_im)

    seg_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, ext)
    imsave(base_dir + seg_fname, seg_im)
    # seg_im = imread(base_dir + seg_fname)

    # blend = label_blending(seg_im, iso_image)
    # from timagetk.visu.mplt import rgb_stack_browser
    # rgb_stack_browser(blend)

    from timagetk.visu.mplt import gif_slice_blending
    gif_fname = base_fname+"-seg_hmin{}{}".format(selected_hmin, '.gif')
    gif_slice_blending(seg_im, iso_image, base_dir+gif_fname, duration=5., resize=[256, 256])
