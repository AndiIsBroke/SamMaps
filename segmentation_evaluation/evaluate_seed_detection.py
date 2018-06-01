import os
import numpy as np
import pandas as pd

from copy import deepcopy
from os.path import exists
from scipy.cluster.vq import vq

from openalea.container import array_dict
# from openalea.image.serial.all import imread
from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.mesh.property_topomesh_io import read_ply_property_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh
from openalea.oalab.colormap.colormap_def import load_colormaps
from openalea.tissue_nukem_3d.nuclei_mesh_tools import nuclei_layer
from timagetk.algorithms import isometric_resampling
from timagetk.components import imread
from timagetk.components import SpatialImage
from timagetk.plugins import linear_filtering, morphology, h_transform, region_labeling, segmentation
from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image

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

from equalization import z_slice_contrast_stretch
from equalization import z_slice_equalize_adapthist
from slice_view import slice_view
from slice_view import slice_n_hist
from detection_evaluation import evaluate_positions_detection
from detection_evaluation import filter_topomesh_vertices
from nomenclature import splitext_zip
from nomenclature import get_nomenclature_channel_fname

XP = 'E35'
SAM = 4
tp = 0
force = True

image_dirname = dirname + "nuclei_images/"
nomenclature_file = SamMaps_dir + "nomenclature.csv"

# -1- CZI input infos:
base_fname = "qDII-CLV3-PIN1-PI-{}-LD-SAM{}".format(XP, SAM)
czi_fname = base_fname + "-T{}.czi".format(tp)

# -4- Define CZI channel names, the microscope orientation, nuclei and membrane channel names and extra channels that should also be registered:
time_steps = [0, 5, 10, 14]
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
ref_ch_name = 'PI'
nuc_ch_name = "TagBFP"

nuc_path_suffix, nuc_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, nuc_ch_name)
memb_path_suffix, memb_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, ref_ch_name)

# Original MEMBRANE image
#------------------------------
img = imread(image_dirname + memb_path_suffix + memb_signal_fname)
size = np.array(img.get_shape())
vxs = np.array(img.get_voxelsize())
ori = np.array(img.get_voxelsize())

# Mask
#------------------------------
## mask image obtein by maximum intensity projection :
# mask_filename = image_dirname+"/"+filename+"/"+filename+"_projection_mask.inr.gz"
## 3D mask image obtein by piling a mask for each slice :
mask_filename = image_dirname + memb_path_suffix + memb_path_suffix[:-1] + "_mask.inr.gz"
if os.path.exists(mask_filename) and 'raw' in ref_ch_name:
    print "Found mask image: '{}'".format(mask_filename)
    mask_img = imread(mask_filename)
    img[mask_img == 0] = 0

# world.add(mask_img, "mask", voxelsize=microscope_orientation*np.array(mask_img.voxelsize), colormap='grey', alphamap='constant', bg_id=255)
# world.add(img,"reference_image",colormap="invert_grey",voxelsize=microscope_orientation*vxs)


# EXPERT topomesh = ground truth
#---------------------------------------------------
# - Load the EXPERT topomesh:
expert_topomesh_fname = image_dirname + memb_path_suffix + memb_path_suffix[:-1] + "_EXPERT_seed.ply"
expert_topomesh = read_ply_property_topomesh(expert_topomesh_fname)

# - Remove masked cells from the 'expert_topomesh':
try:
    mask_img
except NameError:
    pass
else:
    # - Get a dictionary of barycenters:
    expert_barycenters = expert_topomesh.wisp_property('barycenter', 0)
    # -- Convert coordinates into voxel units:
    expert_coords = expert_barycenters.values()/(microscope_orientation*vxs)
    # -- ???
    expert_coords = np.maximum(0,np.minimum(size-1,expert_coords)).astype(np.uint16)
    expert_coords = tuple(np.transpose(expert_coords))
    # -- Detect coordinates outside the mask and remove them:
    expert_mask_value = mask_img[expert_coords]
    expert_cells_to_remove = expert_barycenters.keys()[expert_mask_value==0]
    expert_topomesh = filter_topomesh_vertices(expert_topomesh, expert_cells_to_remove.tolist())
# world.add(expert_topomesh,"masked_expert_seed")
# world["masked_expert_seed"]["property_name_0"] = 'layer'
# world["masked_expert_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# - Create a 'L1_expert_topomesh' (L1 ground truth) out of L1 non-masked cells only:
L1_expert_topomesh = filter_topomesh_vertices(expert_topomesh, "L1")
# world.add(L1_expert_topomesh,"L1_masked_expert_seed")
# world["L1_masked_expert_seed"]["property_name_0"] = 'layer'
# world["L1_masked_expert_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']


#-------------------------------------------------------------------------------
# EVALUATION of seed detection by 'h_transform' and 'region_labeling':
#-------------------------------------------------------------------------------
evaluations = {}
L1_evaluations={}

## Detection parameters:
morpho_radius = 1
h_min = 1500

## Evaluation parameters:
max_matching_distance=2.
outlying_distance=4

std_dev_range = np.concatenate((np.array([0]), np.arange(1.0, 2.0, 0.2)))
rescale_type = ['Original', 'AdaptHistEq', 'ContrastStretch']
for rescaling in rescale_type:
    for iso_resampling in [False, True]:
        for std_dev in std_dev_range:
            for no_nuc in [False, True]:
                # - build a string collecting list of algorithm applied to the nuclei image
                suffix = '-' + rescaling + ('-iso' if iso_resampling else '') + ('-gauss_smooth_{}'.format(std_dev) if std_dev != 0 else '') + ('-{}'.format(nuc_ch_name) if no_nuc else '') + '-hmin_{}'.format(h_min)
                print "\n\nPerforming detection case: {}".format(suffix)
                # - Get the name of the topomesh file:
                topomesh_file = image_dirname + nuc_path_suffix + nuc_path_suffix[:-1] + "{}_seed_detection.ply".format(suffix)
                # - Read MEMBRANE image filename:
                print "\n# Reading membrane signal image file: {}".format(memb_signal_fname)
                img = imread(image_dirname + nuc_path_suffix + memb_signal_fname)
                # world.add(img, 'img', colormap='invert_grey')
                # - If topomesh exist use it, else create it
                if exists(topomesh_file) and not force:
                    print "Found PLY topomesh: '{}'".format(topomesh_file)
                    detected_topomesh = read_ply_property_topomesh(topomesh_file)
                else:
                    print "No topomesh file detected, running seed detection script..."
                    # - Performs masked subtraction of signal:
                    try:
                        img[mask_img == 0] = 0
                    except:
                        pass
                    else:
                        print "Applied mask to membrane signal...",
                    if iso_resampling:
                        print "\nPerforming 'isometric_resampling'..."
                        img = isometric_resampling(img)
                    # - Performs NUCLEI image subtraction from membrane image:
                    if no_nuc:
                        # -- Reading NUCLEI intensity image:
                        print " --> Reading nuclei image file {}".format(nuc_signal_fname)
                        nuc_im = imread(image_dirname + nuc_path_suffix + nuc_signal_fname)
                        if iso_resampling:
                            print " --> Performing 'isometric_resampling' of nuclei image..."
                            nuc_im = isometric_resampling(nuc_im)
                        # -- Gaussian Smoothing:
                        print " --> Smoothing..."
                        nuc_im = linear_filtering(nuc_im, std_dev=1.0,
                                                          method='gaussian_smoothing')
                        # -- Remove the NUCLEI signal from the MEMBRANE signal image:
                        print " --> Substracting NUCLEI signal from the MEMBRANE signal image..."
                        zero_mask = np.greater(nuc_im.get_array(), img.get_array())
                        img = img - nuc_im
                        img[zero_mask] = 0
                        # world.add(nuc_im, 'nuc_im', colormap='invert_grey')
                        # world.add(img, 'img'+suffix, colormap='invert_grey')
                        print "Done!"
                        del nuc_im, zero_mask
                    # -- Performs intensity rescaling for membrane image:
                    if rescaling == 'AdaptHistEq':
                        print " --> Adaptative Histogram Equalization..."
                        img = z_slice_equalize_adapthist(img)
                    if rescaling == 'ContrastStretch':
                        print " --> Contrast Stretching..."
                        img = z_slice_contrast_stretch(img)
                    if std_dev != 0:
                        print " --> 'gaussian_smoothing' with std_dev={}...".format(std_dev)
                        img = linear_filtering(img, 'gaussian_smoothing', std_dev=std_dev)

                    # - Performs seed detection:
                    print " --> Closing-Opening Alternate Sequencial Filter..."
                    asf_img = morphology(img, max_radius=morpho_radius, method='co_alternate_sequential_filter')
                    print " --> H-transform: local minima detection..."
                    ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
                    print " --> Components labelling..."
                    con_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
                    nb_connectec_comp = len(np.unique(con_img.get_array()))
                    print nb_connectec_comp, "seeds detected!"
                    if nb_connectec_comp <= 3:
                        print "Not enough seeds, aborting!"
                        continue
                    # world.add(con_img, 'labelled_seeds', voxelsize=vxs)
                    del ext_img
                    print " --> Seeded watershed..."
                    seg_im = segmentation(img, con_img, method='seeded_watershed', try_plugin=False)
                    # - Performs topomesh creation from detected seeds:
                    print " --> Analysing segmented image...",
                    img_graph = graph_from_image(seg_im, background=1, spatio_temporal_properties=['barycenter', 'L1'], ignore_cells_at_stack_margins=False)
                    print img_graph.nb_vertices(), "seeds extracted by 'graph_from_image'!"
                    # -- Create a topomesh from 'seed_positions':
                    print " --> Topomesh creation..."
                    seed_positions = {v: img_graph.vertex_property('barycenter')[v]*microscope_orientation for v in img_graph.vertices()}
                    detected_topomesh = vertex_topomesh(seed_positions)
                    # -- Detect L1 using 'nuclei_layer':
                    oriented_seeds = {k: np.array([1.,1.,-1.])*v for k, v in seed_positions.items()}
                    cell_layer = img_graph.vertex_property('L1')
                    # -- Update the topomesh with the 'layer' property:
                    detected_topomesh.update_wisp_property('layer', 0, cell_layer)
                    # -- Save the detected topomesh:
                    ppty2ply = dict([(0, ['layer']), (1,[]),(2,[]),(3,[])])
                    save_ply_property_topomesh(detected_topomesh, topomesh_file, properties_to_save=ppty2ply, color_faces=False)

                # - Create a 'detected_topomesh' out of L1 cells only:
                L1_detected_topomesh = filter_topomesh_vertices(detected_topomesh, "L1")
                # world.add(L1_detected_topomesh, L1_detected_seed"+suffix)
                # world[L1_detected_seed"+suffix]["property_name_0"] = 'layer'
                # world["L1_detected_seed{}_vertices".format(suffix)]["polydata_colormap"] = load_colormaps()['Reds']

                # - Performs evaluation of detected seeds agains expert:
                print "\n# Evaluation: comparison to EXPERT seeds..."
                evaluation = evaluate_positions_detection(detected_topomesh, expert_topomesh, max_distance=np.linalg.norm(size*vxs))
                evaluations[suffix] = evaluation
                L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_topomesh, max_distance=np.linalg.norm(size*vxs))
                L1_evaluations[suffix] = L1_evaluation


eval_fname = image_dirname + memb_path_suffix + memb_path_suffix[:-1] + "_nuclei_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(evaluations)
evaluation_df.to_csv(eval_fname)

L1_eval_fname = image_dirname + memb_path_suffix + memb_path_suffix[:-1] + "_L1_nuclei_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(L1_evaluations)
evaluation_df.to_csv(L1_eval_fname)
