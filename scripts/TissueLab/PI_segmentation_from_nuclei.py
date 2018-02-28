# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scipy.ndimage as nd

from openalea.tissue_nukem_3d.nuclei_segmentation import seed_image_from_points

from timagetk.algorithms import isometric_resampling
from timagetk.components import SpatialImage
from timagetk.components import imread
from timagetk.components import imsave
from timagetk.plugins import linear_filtering, morphology, segmentation

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
    dirname = "/data/Meristems/Carlos/PIN_maps/"
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
    dirname = "/projects/SamMaps/"
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/TissueLab/')

from equalization import z_slice_equalize_adapthist
from nomenclature import get_nomenclature_name
from nomenclature import get_nomenclature_channel_fname
from nomenclature import get_nomenclature_segmentation_name

import time
start = time.time()

# XP = 'E35'
XP = sys.argv[1]
# SAM = '5'
SAM = sys.argv[2]

# Examples
# --------
# python SamMaps/scripts/TissueLab/PI_segmentation_from_nuclei.py 'E35' '4'
# python SamMaps/scripts/TissueLab/PI_segmentation_from_nuclei.py 'E37' '5'

nomenclature_file = SamMaps_dir + "nomenclature.csv"
force = True

# PARAMETERS:
# -----------
# -1- CZI input infos:
base_fname = "qDII-CLV3-PIN1-PI-{}-LD-SAM{}".format(XP, SAM)
czi_base_fname = base_fname + "-T{}.czi"
time_steps = [0, 5, 10, 14]

# -3- OUTPUT directory:
image_dirname = dirname + "nuclei_images/"

# -4- Define CZI channel names, the microscope orientation, nuclei and membrane channel names and extra channels that should also be registered:
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
membrane_ch_name = 'PI'
membrane_ch_name += '_raw'
background = 1

for tp, t in enumerate(time_steps):
    raw_czi_fname = czi_base_fname.format(t)
    print "\n\n# - Entering segmentation process for {}".format(raw_czi_fname)

    # - Defines segmented file name and path:
    seg_path_suffix, seg_img_fname = get_nomenclature_segmentation_name(raw_czi_fname, nomenclature_file, membrane_ch_name)
    if os.path.exists(image_dirname + seg_path_suffix + seg_img_fname) and not force:
        print "A segmentation file '{}' already exists, aborting now.".format(seg_img_fname)
        sys.exit(0)

    # - Get the image to segment:
    # -- Get the file name and path of the image to segment:
    path_suffix, img2seg_fname = get_nomenclature_channel_fname(raw_czi_fname, nomenclature_file, membrane_ch_name)
    print "\n - Loading image to segment: {}".format(img2seg_fname)
    img2seg = imread(image_dirname + path_suffix + img2seg_fname)
    vxs = np.array(img2seg.get_voxelsize())
    ori = np.array(img2seg.get_origin())
    # -- Get the file name and path of the channel to substract to the image to segment:
    # used to clear-out the cells for better segmentation
    path_suffix, substract_img_fname = get_nomenclature_channel_fname(raw_czi_fname, nomenclature_file, 'CLV3')
    print "\n - Loading image to substract: {}".format(substract_img_fname)
    substract_img = imread(image_dirname + path_suffix + substract_img_fname)
    # substract the 'CLV3' signal from the 'PI' since it might have leaked:
    print "\n - Performing images substraction..."
    img2seg = img2seg - substract_img
    img2seg[img2seg <= substract_img] = 0
    img2seg = SpatialImage(img2seg, voxelsize=vxs, origin=ori)
    # -- Display the image to segment:
    # world.add(img2seg,'{}_channel'.format(membrane_ch_name), colormap='invert_grey', voxelsize=microscope_orientation*vxs)
    # world['{}_channel'.format(membrane_ch_name)]['intensity_range'] = (-1, 2**16)
    # -- Adaptative histogram equalization of the image to segment:
    print "\n - Performing adaptative histogram equalization of the image to segment..."
    img2seg = z_slice_equalize_adapthist(img2seg)
    # -- Performs isometric resampling of the image to segment:
    print "\n - Performing isometric resampling of the image to segment..."
    img2seg = isometric_resampling(img2seg)
    iso_vxs = np.array(img2seg.get_voxelsize())
    iso_shape = img2seg.get_shape()
    # -- Display the isometric version of the "equalized" image to segment:
    # world.add(img2seg,'{}_channel_equalized_isometric'.format(membrane_ch_name), colormap='invert_grey', voxelsize=microscope_orientation*iso_vxs)
    # world['{}_channel_equalized_isometric'.format(membrane_ch_name)]['intensity_range'] = (-1, 2**16)

    # - Create a seed image from the list of NUCLEI barycenters:
    print "\n - Creating a a seed image from the list of NUCLEI barycenters..."
    # -- Read CSV file containing barycenter positions:
    nom_names = get_nomenclature_name(nomenclature_file)
    signal_file = image_dirname + nom_names[raw_czi_fname] + '/' + nom_names[raw_czi_fname] + "_signal_data.csv"
    signal_data = pd.read_csv(signal_file, sep=',')[:-1]
    x, y, z = signal_data.center_x, signal_data.center_y, signal_data.center_z
    bary_dict = {k: np.array([x[k], y[k], z[k]])*microscope_orientation for k in x.keys()}
    # -- If 0 or background (usually 1) is a key in the list of barycenters, we change
    # them since they are "special values":
    unauthorized_labels = [0, background]
    for l in unauthorized_labels:
        if l in bary_dict.keys():
            bary_dict[np.max(bary_dict.keys())+1] = bary_dict.pop(l)
    # -- Construct the seed image with a 'zero' background, since "true background" will be added later:
    seed_img = seed_image_from_points(iso_shape, iso_vxs, bary_dict, 1.0, 0)
    # -- Add background position (use membrane intensity):
    background_threshold = 2000.
    smooth_img_bck = linear_filtering(img2seg, std_dev=3.0, method='gaussian_smoothing')
    background_img = (smooth_img_bck < background_threshold).astype(np.uint16)
    for it in xrange(15):
        background_img = morphology(background_img, param_str_2 = '-operation erosion -iterations 10')
    del smooth_img_bck
    # -- Detect small regions defined as background and remove them:
    connected_background_components, n_components = nd.label(background_img)
    components_area = nd.sum(np.ones_like(connected_background_components), connected_background_components, index=np.arange(n_components)+1)
    largest_component = (np.arange(n_components)+1)[np.argmax(components_area)]
    background_img = (connected_background_components == largest_component).astype(np.uint16)
    # -- Finaly add the background and make a SpatialImage:
    seed_img[background_img==background] = background
    seed_img = SpatialImage(seed_img, voxelsize=iso_vxs)
    del background_img
    # world.add(seed_img, "seed_image", colormap="glasbey", alphamap="constant", voxelsize=microscope_orientation*iso_vxs, bg_id=background)

    # - Performs automatic seeded watershed using previously created seed image:
    print "\n - Performing seeded watershed segmentation using seed image from nuclei..."
    # -- Performs Gaussian smoothing:
    std_dev = 1.0
    smooth_img = linear_filtering(img2seg, std_dev=std_dev, method='gaussian_smoothing')
    # -- Performs the seeded watershed segmentation:
    seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed')
    # -- Display the segmented image:
    # world.add(seg_im, "seg_image", colormap="glasbey", alphamap="constant",voxelsize=microscope_orientation*iso_vxs, bg_id=background)
    # -- Save the segmented image:
    print "Saving the segmented image: {}".format(seg_img_fname)
    imsave(image_dirname + seg_path_suffix + seg_img_fname, seg_im)
    # -- Print some info about how it went:
    labels = np.unique(seg_im)
    nb_labels = len(labels)
    seeds = np.unique(seed_img)
    nb_seeds = len(seeds)
    print "Found {} labels out of {} seeds!".format(nb_labels, nb_seeds)
    end = int(np.ceil(time.time() - start))
    print "It took {}min {}s to performs all operations on {}!".format(end/60, end%60, raw_czi_fname)
