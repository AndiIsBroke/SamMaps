# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon - INRIA
#
#       File author(s): Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
################################################################################
"""
Library associated to the segmentation process.
"""
import numpy as np

from timagetk.components import imread
from timagetk.components import SpatialImage
from timagetk.plugins import morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling
from timagetk.plugins import linear_filtering
from timagetk.plugins import segmentation
from timagetk.algorithms.resample import resample
from timagetk.algorithms.resample import isometric_resampling

from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
from openalea.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image as read_czi
from openalea.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_lsm_image as read_lsm

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/lib/')

from nomenclature import splitext_zip
from equalization import z_slice_contrast_stretch
from equalization import z_slice_equalize_adapthist


def segmentation_fname(img2seg_fname, h_min, iso, equalize, stretch):
    """
    Generate the segmentation filename using some of the pipeline steps.

    Parameters
    ----------
    img2seg_fname : str
        filename of the image to segment.
    h_min : int
        h-minima used with the h-transform function
    iso : bool
        indicate if isometric resampling was performed by the pipeline
    equalize : bool
        indicate if adaptative equalization of intensity was performed
    stretch : bool
        indicate if intensity histogram stretching was performed
    """
    suffix = '-seg'
    suffix += '-iso' if iso else ''
    suffix += '-adpat_eq' if equalize else ''
    suffix += '-hist_stretch' if stretch else ''
    suffix += '-h_min{}'.format(h_min)
    seg_img_fname = splitext_zip(img2seg_fname)[0] + suffix + '.inr'
    return seg_img_fname


def signal_subtraction(img2seg, img2sub):
    """
    Performs SpatialImage subtraction.

    Parameters
    ----------
    img2seg : str
        image to segment.
    img2sub : str, optional
        image to subtract to the image to segment.
    """
    vxs = img2seg.get_voxelsize()
    ori = img2seg.origin()
    try:
        assert np.allclose(img2seg.get_shape(), img2sub.get_shape())
    except AssertionError:
        raise ValueError("Input images does not have the same shape!")
    img2sub = read_image(substract_inr)
    # img2sub = morphology(img2sub, method='erosion', radius=3.)
    tmp_im = img2seg - img2sub
    tmp_im[img2seg <= img2sub] = 0
    img2seg = SpatialImage(tmp_im, voxelsize=vxs, origin=ori)

    return img2seg


def read_image(im_fname, channel_names=None):
    """
    Read CZI or INR images based on the 'im_fname' extension.

    Parameters
    ----------
    im_fname : str
        filename of the image to read.
    """
    if im_fname.endswith(".inr") or im_fname.endswith(".inr.gz"):
        im = imread(im_fname)
    elif im_fname.endswith(".tif"):
        im = imread(im_fname)
    elif im_fname.endswith(".lsm"):
        im = read_lsm(im_fname)
        if isinstance(im, dict):
            print im.keys()
            im = {k: SpatialImage(ch, voxelsize=ch.voxelsize) for k, ch in im.items()}
            if channel_names is not None:
                im = replace_channel_names(im, channel_names)
        else:
            im = SpatialImage(im, voxelsize=im.voxelsize)
    elif im_fname.endswith(".czi"):
        im = read_czi(im_fname)
        try:
            im2 = read_czi(im_fname, pattern="..CZXY.")
            assert isinstance(im2, dict)
        except:
            del im2
        else:
            im = im2
        for k, ch in im.items():
            if not isinstance(ch, SpatialImage):
                im[k] = SpatialImage(ch, voxelsize=ch.voxelsize)
        if channel_names is not None:
            im = replace_channel_names(im, channel_names)
    else:
        raise TypeError("Unknown reader for file '{}'".format(im_fname))
    return im


def replace_channel_names(img_dict, channel_names):
    """
    Replace the
    """
    try:
        assert len(channel_names) == len(img_dict)
    except AssertionError:
        raise ValueError("Not enought channel names ({}) for image channels ({})!".format(len(channel_names), len(img_dict)))

    for n, k in enumerate(img_dict.keys()):
        img_dict[channel_names[n]] = img_dict.pop(k)

    return img_dict


def seg_pipe(img2seg, h_min, img2sub=None, iso=True, equalize=True, stretch=False, std_dev=1.0, min_cell_volume=20., back_id=1):
    """
    Define the sementation pipeline

    Parameters
    ----------
    img2seg : str
        image to segment.
    h_min : int
        h-minima used with the h-transform function
    img2sub : str, optional
        image to subtract to the image to segment.
    iso : bool, optional
        if True (default), isometric resampling is performed after h-minima
        detection and before watershed segmentation
    equalize : bool, optional
        if True (default), intensity adaptative equalization is performed before
        h-minima detection
    stretch : bool, optional
        if True (default, False), intensity histogram stretching is performed
        before h-minima detection
    std_dev : float, optional
        standard deviation used for Gaussian smoothing of the image to segment
    min_cell_volume : float, optional
        minimal volume accepted in the segmented image


    Returns
    -------
    seg_im : SpatialImage
        the labelled image obtained by seeded-watershed

    Notes
    -----
    Both 'equalize' & 'stretch' can not be True at the same time since they work
    on the intensity of the pixels.
    Linear filtering (Gaussian smoothing) is performed before h-minima transform
    for local minima detection.
    Gaussian smoothing is performed
    """
    try:
        assert equalize + stretch < 2
    except AssertionError:
        raise ValueError("Both 'equalize' & 'stretch' can not be True at once!")

    ori_vxs = img2seg.get_voxelsize()
    ori_shape = img2seg.get_shape()
    if equalize:
        print "\n - Performing z-slices adaptative histogram equalisation on the intensity image to segment..."
        img2seg = z_slice_equalize_adapthist(img2seg)
    if stretch:
        print "\n - Performing z-slices histogram contrast stretching on the intensity image to segment..."
        img2seg = z_slice_contrast_stretch(img2seg)
    if img2sub is not None:
        print "\n - Performing signal substraction..."
        img2seg = signal_subtraction(img2seg, img2sub)

    print "\n - Automatic seed detection...".format(h_min)
    # morpho_radius = 1.0
    # asf_img = morphology(img2seg, max_radius=morpho_radius, method='co_alternate_sequential_filter')
    # ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
    print " -- Isometric resampling prior to Gaussian smoothing...".format(std_dev)
    iso_img = isometric_resampling(img2seg)
    print " -- Gaussian smoothing with std_dev={}...".format(std_dev)
    iso_smooth_img = linear_filtering(iso_img, std_dev=std_dev, method='gaussian_smoothing')
    del iso_img  # no need to keep this image after this step!
    print " -- Down-sampling back to original voxelsize..."
    smooth_img = resample(iso_smooth_img, ori_vxs)
    if not np.allclose(ori_shape, smooth_img.get_shape()):
        print "WARNING: shape missmatch after down-sampling from isometric image:"
        print " -- original image shape: {}".format(ori_shape)
        print " -- down-sampled image shape: {}".format(smooth_img.get_shape())
    if not iso:
        del iso_smooth_img  # no need to keep this image after this step!
    # ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
    print " -- H-minima transform with h-min={}...".format(h_min)
    ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')
    if iso:
        smooth_img = iso_smooth_img  # no need to keep both images after this step!
    print " -- Region labelling: connexe components detection..."
    seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
    print "Detected {} seeds!".format(len(np.unique(seed_img))-1)  # '0' is in the list!
    del ext_img  # no need to keep this image after this step!

    print "\n - Performing seeded watershed segmentation..."
    if iso:
        seed_img = isometric_resampling(seed_img, option='label')
    seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed', try_plugin=False)
    # seg_im[seg_im == 0] = back_id
    print "Detected {} labels!".format(len(np.unique(seg_im)))

    if min_cell_volume > 0.:
        print "\n - Performing cell volume filtering..."
        spia = SpatialImageAnalysis(seg_im, background=None)
        vol = spia.volume()
        too_small_labels = [k for k, v in vol.items() if v < min_cell_volume and k != 0]
        if too_small_labels != []:
            print "Detected {} labels with a volume < {}µm2".format(len(too_small_labels), min_cell_volume)
            print " -- Removing seeds leading to small cells..."
            spia = SpatialImageAnalysis(seed_img, background=None)
            seed_img = spia.get_image_without_labels(too_small_labels)
            print " -- Performing seeded watershed segmentation..."
            seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed', try_plugin=False)
            # seg_im[seg_im == 0] = back_id
            print "Detected {} labels!".format(len(np.unique(seg_im)))

    return seg_im
