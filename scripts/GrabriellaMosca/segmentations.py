import numpy as np

from timagetk.components import SpatialImage

from timagetk.io import imread
from timagetk.io import imsave

from timagetk.plugins import morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling
from timagetk.plugins import linear_filtering
from timagetk.plugins import segmentation
from timagetk.algorithms.resample import resample
from timagetk.algorithms.resample import isometric_resampling

from timagetk.algorithms.exposure import z_slice_contrast_stretch
from timagetk.algorithms.exposure import z_slice_equalize_adapthist


raw_img = imread("/data/GabriellaMosca/EM_C_140/EM_C_140- C=0.tif")
h_min = 2
sigma = 1.5

# raw_img = imread("/data/GabriellaMosca/EM_C_214/EM_C_214 C=0.tif")

print "\n - Performing z-slices adaptative histogram equalisation on the intensity image to segment..."
eq_img1 = z_slice_equalize_adapthist(raw_img)
print "\n - Performing z-slices histogram contrast stretching on the intensity image to segment..."
eq_img2 = z_slice_contrast_stretch(raw_img)

from timagetk.visu.mplt import grayscale_imshow
grayscale_imshow([raw_img.get_z_slice(85), eq_img1.get_z_slice(85), eq_img2.get_z_slice(85)], img_title=['Original', "AHE", 'CS'] )

img2seg = eq_img2

print "\n - Automatic seed detection...".format(h_min)
print " -- Gaussian smoothing with std_dev={}...".format(sigma)
smooth_img = linear_filtering(img2seg, sigma=sigma, method='gaussian_smoothing')
grayscale_imshow([raw_img.get_z_slice(85), eq_img2.get_z_slice(85), smooth_img.get_z_slice(85)], img_title=['Original', "CS", 'CS+GS'] )


ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')
seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components', param=True)
print "Detected {} seeds!".format(len(np.unique(seed_img))-1)  # '0' is in the list!
del ext_img  # no need to keep this image after this step!

print "\n - Performing seeded watershed segmentation..."
seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed')
# seg_im[seg_im == 0] = back_id
print "Detected {} labels!".format(len(np.unique(seg_im)))
