import numpy as np
from timagetk.components import imread, SpatialImage

from matplotlib import gridspec
import matplotlib.pyplot as plt

from timagetk.plugins import linear_filtering
from timagetk.algorithms import isometric_resampling

import sys
sys.path.append('/home/marie/SamMaps/scripts/TissueLab/')
from equalization import z_slice_contrast_stretch, x_slice_contrast_stretch, y_slice_contrast_stretch
from equalization import z_slice_equalize_adapthist
from slice_view import slice_view
from slice_view import slice_n_hist

dirname = '/home/marie/Carlos/qDII-CLV3-PIN1-PI-E35-LD/SAM4/'
fname = '/qDII-CLV3-PIN1-PI-E35-LD-SAM4-T0_CH2_iso.inr'
img = imread(dirname + fname)
x_sh, y_sh, z_sh = img.get_shape()

z_slice = 107
# Display slice and histograms:
slice_n_hist(img[:,:,z_slice], 'Original image', 'z-slice {}/{}'.format(z_slice, z_sh))

# Display orthogonal view of ORIGINAL image:
slice_view(img, x_sh/2, y_sh/2, z_sh/2, 'original_image', dirname + fname[:-4] + ".png")


# Display and save orthogonal view of EQUALIZE ADAPTHIST image for various clip_limit:

for clip_limit in [0.005,0.007,0.01,0.02,0.05,0.1]:
    im_eq = z_slice_equalize_adapthist(img,clip_limit=float(clip_limit))
    x_sh, y_sh, z_sh = im_eq.shape
    title = "Equalize adapthist (z-slice): clip limit = {}".format(clip_limit)
    filename = fname[:-4]+"_z_equalize_adapthist-clip_limit_{}.png".format(clip_limit)
    slice_view(im_eq, x_sh/2, y_sh/2, z_sh/2, title, dirname + filename)

# Display and save orthogonal view of EQUALIZE ADAPTHIST image for nbins=2**16:
n_bins=2**16
clip_limit=0.005
im_eq = z_slice_equalize_adapthist(img,clip_limit=clip_limit,n_bins=n_bins)
x_sh, y_sh, z_sh = im_eq.shape
title = "Equalize adapthist (z-slice):clip limit = {}".format(clip_limit)
filename = fname[:-4]+"_z_equalize_adapthist_clip_limit_{}.png".format(clip_limit)
slice_view(im_eq, x_sh/2, y_sh/2, z_sh/2, title, dirname + filename)

slice_n_hist(im_eq[:,:,z_slice], "Equalize adapthist (z-slice):clip limit = {}.format(clip_limit), 'z-slice {}/{}'.format(z_slice, z_sh))

# Display and save orthogonal view of CONTRAST STRETCHED image for various upper percentiles (pc_max):
pc_min = 2
pc_max = 90
for pc_max in [90]:
    im_eq_z = z_slice_contrast_stretch(img, pc_min=pc_min, pc_max=pc_max)
    im_eq_zy = y_slice_contrast_stretch(im_eq_z, pc_min=pc_min, pc_max=pc_max)
    im_eq_zyx = z_slice_contrast_stretch(im_eq_zy, pc_min=pc_min, pc_max=pc_max)

    x_sh, y_sh, z_sh = im_eq_zyx.shape
    title = "Contrast stretched (zyx slices): {}pc.-{}pc.".format(pc_min, pc_max)
    filename = fname[:-4]+"_zyx_contrast_stretch_{}-{}pc.png".format(pc_min, pc_max)
    slice_view(im_eq_zyx, x_sh/2, y_sh/2, z_sh/2, title, dirname + filename)

slice_n_hist(im_eq[:,:,z_slice], "Contrast stretched (z-slice): {}pc.-{}pc.".format(pc_min, pc_max), 'z-slice {}/{}'.format(z_slice, z_sh))
#
# # gaussian_smoothing
# im_eq = isometric_resampling(im_eq)
# vxs = im_eq.voxelsize
# im_eq=SpatialImage(im_eq,voxelsize=vxs)
# std_dev=2.0
# smooth_img = linear_filtering(im_eq, std_dev=std_dev, method='gaussian_smoothing')
# x_sh, y_sh, z_sh = im_eq.shape
# title = "Contrast stretched (z-slice): {}pc.-{}pc + gaussian smoothing: std_dev:{}.".format(pc_min, pc_max, std_dev)
# filename = fname[:-4]+"_z_contrast_stretch_{}-{}pc, gaussian smoothing_std_dev_{}.png".format(pc_min, pc_max,std_dev)
# slice_view(smooth, x_sh/2, y_sh/2, z_sh/2, title, dirname + filename)
#
#
