import numpy as np

from timagetk.components import imread
from timagetk.algorithms import isometric_resampling
from timagetk.plugins import morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling
from timagetk.plugins import linear_filtering
from timagetk.plugins import segmentation

from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
from openalea.tissue_nukem_3d.microscopy_images import imread as read_czi

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/TissueLab/')

from nomenclature import splitext_zip
from equalization import z_slice_contrast_stretch


def segmentation_fname(im2seg_fname, h_min, iso, equalize):
    """
    Generate the segmentation filename using some of the pipeline steps.

    Parameters
    ----------
    im2seg_fname : str
        filename of the image to segment.
    h_min : int
        h-minima used with the h-transform function
    iso : bool
        indicate if isometric resampling was performed by the pipeline
    equalize : bool
        indicate if intensity equalization was performed by the pipeline
    """
    suffix = '_seg'.format(h_min)
    suffix += '_iso' if iso else ''
    suffix += '_eq' if equalize else ''
    suffix += '_hmin{}'.format(h_min)
    seg_img_fname = splitext_zip(inr_fname)[0] + suffix + '.inr'


def signal_subtraction(im2seg, im2sub):
    """
    Performs SpatialImage subtraction.

    Parameters
    ----------
    im2seg : str
        image to segment.
    im2sub : str, optional
        image to subtract to the image to segment.
    """
    vxs = im2seg.get_voxelsize()
    ori = im2seg.origin()
    try:
        assert np.allclose(im2seg.get_shape(), im2sub.get_shape())
    except AssertionError:
        raise ValueError("Input images does not have the same shape!")
    im2sub = read_image(substract_inr)
    # im2sub = morphology(im2sub, method='erosion', radius=3.)
    tmp_im = img2seg - im2sub
    tmp_im[img2seg <= im2sub] = 0
    img2seg = SpatialImage(tmp_im, voxelsize=vxs, origin=ori)

    return im2seg

def read_image(im_fname):
    """
    Read CZI or INR images based on the 'im_fname' extension.

    Parameters
    ----------
    im_fname : str
        filename of the image to read.
    """
    if im_fname.endswith(".inr"):
        im = imread(im_fname)
    elif im_fname.endswith(".czi"):
        im = read_czi(im_fname)
    else:
        raise TypeError("Unknown reader for file '{}'".format(im_fname))
    return im

def seg_pipe(im2seg, h_min, im2sub=None, iso=True, equalize=True, std_dev=1.0, min_cell_volume=50.):
    """
    Define the sementation pipeline

    Parameters
    ----------
    im2seg : str
        image to segment.
    h_min : int
        h-minima used with the h-transform function
    im2sub : str, optional
        image to subtract to the image to segment.
    iso : bool, optional
        if True (default), isometric resampling is performed before segmentation
    equalize : bool, optional
        if True (default), intensity equalization is performed before segmentation
    std_dev : float, optional
        standard deviation used for Gaussian smoothing of the image to segment
    min_cell_volume : float, optional
        minimal volume accepted in the segmented image
    """
    if iso:
        print "\n - Performing isometric resampling of the intensity image to segment..."
        img2seg = isometric_resampling(im2seg)

    if equalize:
        print "\n - Performing histogram contrast stretching of the intensity image to segment..."
        img2seg = z_slice_contrast_stretch(img2seg)

    if im2sub is not None:
        print "\n - Performing signal substraction..."
        if iso:
            print "\n - Performing isometric resampling of the intensity image to subtract..."
            im2sub = isometric_resampling(im2sub)
        im2seg = signal_subtraction(im2seg, im2sub)

    print "\n# - Automatic seed detection..."
    # morpho_radius = 1.0
    # asf_img = morphology(img2seg, max_radius=morpho_radius, method='co_alternate_sequential_filter')
    # ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
    smooth_img = linear_filtering(img2seg, std_dev=std_dev, method='gaussian_smoothing')
    # ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
    ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')
    seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
    print "Detected {} seeds!".format(len(np.unique(seed_img)))

    print "\n - Performing seeded watershed segmentation..."
    seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed', try_plugin=False)
    # seg_im[seg_im == 0] = back_id
    print "Detected {} labels!".format(len(np.unique(seg_im)))

    if min_cell_volume > 0.:
        print "\n - Performing cell volume filtering..."
        spia = SpatialImageAnalysis(seg_im, background=back_id)
        vol = spia.volume()
        too_small_labels = [k for k, v in vol.items() if v < min_cell_volume]
        if too_small_labels != []:
            print "Detected {} labels with a volume < {}µm2".format(len(too_small_labels), min_cell_volume)
            print " -- Removing seeds leading to small cells..."
            for l in too_small_labels:
                seed_img[seed_img == l] = 0
            print " -- Performing seeded watershed segmentation..."
            seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed', try_plugin=False)
            # seg_im[seg_im == 0] = back_id
            print "Detected {} labels!".format(len(np.unique(seg_im)))

    return seg_im
