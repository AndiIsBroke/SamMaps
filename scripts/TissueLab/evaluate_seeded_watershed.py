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
from openalea.tissue_nukem_3d.nuclei_segmentation import seed_image_from_points
from timagetk.algorithms import isometric_resampling
from timagetk.components import imread
from timagetk.components import SpatialImage
from timagetk.plugins import linear_filtering, morphology, h_transform, region_labeling, segmentation
from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image


import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    sys.path.append('/data/Meristems/Carlos/SamMaps/scripts/TissueLab/')
elif platform.uname()[1] == "RDP-T3600-AL":
    sys.path.append('/home/marie/SamMaps/scripts/TissueLab/')
else:
    raise ValueError("Unknown custom path for this system...")

from equalization import z_slice_contrast_stretch
from equalization import z_slice_equalize_adapthist
from slice_view import slice_view
from slice_view import slice_n_hist
from detection_evaluation import evaluate_positions_detection

# Files's directories
#-----------------------
if platform.uname()[1] == "RDP-M7520-JL":
    dirname = "/data/Meristems/"
elif platform.uname()[1] == "RDP-T3600-AL":
    dirname = "/home/marie/"
else:
    raise ValueError("Unknown custom path for this system...")


# image_dirname = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/seed_ground_truth_images/"
# image_dirname = "/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images"
image_dirname = dirname+"Carlos/nuclei_images"
#image_dirname = dirname+"Marie/Lti6b/2017-12-01/"

# filename = 'DR5N_6.1_151124_sam01_z0.50_t00'
# filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t05'
filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t00'
microscope_orientation = -1

# reference_name = "tdT"
reference_name = "PI"

image_filename = image_dirname+"/"+filename+"/"+filename+"_"+reference_name+".inr.gz"

# Original image
#------------------------------
img = imread(image_filename)
size = np.array(img.shape)
voxelsize = np.array(img.voxelsize)

# Mask
#------------------------------
## mask image obtein by maximum intensity projection :
# mask_filename = image_dirname+"/"+filename+"/"+filename+"_projection_mask.inr.gz"
## 3D mask image obtein by piling a mask for each slice :
mask_filename = image_dirname+"/"+filename+"/"+filename+"_mask.inr.gz"
if exists(mask_filename):
    mask_img = imread(mask_filename)
else:
    mask_img = np.ones_like(img)

img[mask_img == 0] = 0
img = SpatialImage(img, voxelsize=voxelsize)

# world.add(mask_img,"mask",voxelsize=microscope_orientation*np.array(mask_img.voxelsize),colormap='grey',alphamap='constant',bg_id=255)
# world.add(img,"reference_image",colormap="invert_grey",voxelsize=microscope_orientation*voxelsize)

# Corrected image of detected seed = ground truth
#---------------------------------------------------
corrected_filename = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_topomesh_corrected.ply"
# corrected_filename = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_topomesh_corrected_AdaptHistEq.ply"

corrected_topomesh = read_ply_property_topomesh(corrected_filename)
corrected_positions = corrected_topomesh.wisp_property('barycenter',0)
# Convert coordinates into voxel units:
corrected_coords = corrected_positions.values()/(microscope_orientation*voxelsize)
# ???
corrected_coords = np.maximum(0,np.minimum(size-1,corrected_coords)).astype(np.uint16)
corrected_coords = tuple(np.transpose(corrected_coords))

## Mask application :
corrected_mask_value = mask_img[corrected_coords]
corrected_cells_to_remove = corrected_positions.keys()[corrected_mask_value==0]
for c in corrected_cells_to_remove:
    corrected_topomesh.remove_wisp(0,c)
for property_name in corrected_topomesh.wisp_property_names(0):
    corrected_topomesh.update_wisp_property(property_name,0,array_dict(corrected_topomesh.wisp_property(property_name,0).values(list(corrected_topomesh.wisps(0))),keys=list(corrected_topomesh.wisps(0))))

# world.add(corrected_topomesh,"corrected_seed")
# world["corrected_seed"]["property_name_0"] = 'layer'
# world["corrected_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# - Filter L1-corrected seed (ground truth):
L1_corrected_topomesh = deepcopy(corrected_topomesh)
L1_corrected_cells = np.array(list(L1_corrected_topomesh.wisps(0)))[L1_corrected_topomesh.wisp_property('layer',0).values()==1]
non_L1_corrected_cells = [c for c in L1_corrected_topomesh.wisps(0) if not c in L1_corrected_cells]
for c in non_L1_corrected_cells:
    L1_corrected_topomesh.remove_wisp(0,c)
for property_name in L1_corrected_topomesh.wisp_property_names(0):
    L1_corrected_topomesh.update_wisp_property(property_name,0,array_dict(L1_corrected_topomesh.wisp_property(property_name,0).values(list(L1_corrected_topomesh.wisps(0))),keys=list(L1_corrected_topomesh.wisps(0))))
# world.add(L1_corrected_topomesh,"L1_corrected_seed"+suffix)
# world["L1_corrected_seed"]["property_name_0"] = 'layer'
# world["L1_corrected_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']


# EVALUATION
#---------------------------------------------------
evaluations = {}
L1_evaluations={}
## Parameters
std_dev = 2.0
morpho_radius = 3
h_min = 170

# - Starts by comparing cell barycenters (obtained by segmentation using expert seeds) to expert seed position:
# -- Create a seed image from expertised seed positions:
xp_seed_pos = corrected_topomesh.wisp_property('barycenter', 0)
xp_seed_pos = {k:v*microscope_orientation for k, v in xp_seed_pos.items()}
# --- Change following values, as required by watershed algorithm:
#  - '0': watershed will fill these with other label
#  - '1': background value (outside the biological object)
for label in [0, 1]:
    if xp_seed_pos.has_key(label):
        mk = max(xp_seed_pos.keys())
        xp_seed_pos[mk + 1] = xp_seed_pos[label]
        xp_seed_pos.delete(label)

# --- Create the seed image:
con_img = seed_image_from_points(size, voxelsize, xp_seed_pos, 2., 0)
# --- Add background position:
background_threshold = 2000.
smooth_img_bck = linear_filtering(img, std_dev=3.0, method='gaussian_smoothing')
background_img = (smooth_img_bck < background_threshold).astype(np.uint16)
for it in xrange(10):
    background_img = morphology(background_img, param_str_2 = '-operation erosion -iterations 10')
con_img += background_img
del smooth_img_bck, background_img
con_img = SpatialImage(con_img, voxelsize=voxelsize)
# -- Performs automatic seeded watershed using previously created seed image:
smooth_img = linear_filtering(img, std_dev=std_dev, method='gaussian_smoothing')
seg_im = segmentation(smooth_img, con_img)
# -- Create a vertex_topomesh from detected cell positions:
# --- Get cell barycenters positions:
img_graph = graph_from_image(seg_im, background=1, spatio_temporal_properties=['L1', 'barycenter'], ignore_cells_at_stack_margins=False)
print img_graph.nb_vertices()," cells detected"
vtx = list(img_graph.vertices())
in_L1 = img_graph.vertex_property('L1')
L1_labels = [l for l in vtx if in_L1[l]]
bary = img_graph.vertex_property('barycenter')
cell_layer = {l: in_L1[l] for l in vtx}
L1_cell_layer = {l: 1 for l in L1_labels}

# --- Create a topomesh out of them:
cell_positions = {v: bary[v]*microscope_orientation for v in vtx}
detected_topomesh = vertex_topomesh(cell_positions)
detected_topomesh.update_wisp_property('layer', 0, cell_layer)
# --- Create a topomesh out of them:
L1_cell_positions = {v: bary[v]*microscope_orientation for v in L1_labels}
L1_detected_topomesh = vertex_topomesh(L1_cell_positions)
L1_detected_topomesh.update_wisp_property('layer', 0, L1_cell_layer)
# -- Performs evaluation:
evaluation = evaluate_positions_detection(detected_topomesh, corrected_topomesh, max_distance=np.linalg.norm(size*voxelsize))
evaluations['Expert'] = evaluation
L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_corrected_topomesh, max_distance=np.linalg.norm(size*voxelsize))
L1_evaluations['Expert'] = L1_evaluation


# - Now evaluate the effect of contrast stretching techniques on automatic seeded watershed algorithm
rescale_type = ['Original', 'AdaptHistEq', 'ContrastStretch']
for rescaling in rescale_type:
    print "rescale_type : " + rescaling
    evaluations[rescaling] = []
    L1_evaluations[rescaling] = []
    suffix = "_" + rescaling
    topomesh_file = image_dirname+"/"+filename+"/"+filename+"_{}_nuclei_detection_topomesh.ply".format(rescaling)
    if exists(topomesh_file):
        detected_topomesh = read_ply_property_topomesh(topomesh_file)
    else:
        if rescaling == 'AdaptHistEq':
            # Need to relaod the orignial image, we don't want to apply histogram equalization technique on masked images
            img = imread(image_filename)
            img = z_slice_equalize_adapthist(img)
            img[mask_img == 0] = 0
        if rescaling == 'ContrastStretch':
            # Need to relaod the orignial image, we don't want to apply histogram equalization technique on masked images
            img = imread(image_filename)
            img = z_slice_contrast_stretch(img)
            img[mask_img == 0] = 0

        img = SpatialImage(img, voxelsize=voxelsize)
        # world.add(img,"ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
        img = isometric_resampling(img)
        # world.add(img,"iso_ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
        size = np.array(img.shape)
        voxelsize = np.array(img.voxelsize)
        print "Shape: ", img.get_shape(), "; Size: ", img.get_voxelsize()

        # - Performs seed detection:
        smooth_img = linear_filtering(img, std_dev=std_dev, method='gaussian_smoothing')
        asf_img = morphology(img, max_radius=morpho_radius, method='co_alternate_sequential_filter')
        ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
        con_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
        # world.add(con_img, 'labelled_seeds', voxelsize=voxelsize)
        seg_im = segmentation(smooth_img, con_img)
        img_graph = graph_from_image(seg_im, background=1, spatio_temporal_properties=['L1', 'barycenter'], ignore_cells_at_stack_margins=False)
        print img_graph.nb_vertices()," cells detected"

        vtx = list(img_graph.vertices())
        L1 = img_graph.vertex_property('L1')
        bary = img_graph.vertex_property('barycenter')
        cell_layer = {l: L1[l] for l in vtx}
        cell_positions = {v: bary[v] for v in vtx}

        detected_topomesh = vertex_topomesh(cell_positions)
        detected_topomesh.update_wisp_property('layer', 0, cell_layer)

        ppty2ply = dict([(0, [reference_name]+['layer']), (1,[]),(2,[]),(3,[])])
        save_ply_property_topomesh(detected_topomesh, topomesh_file, properties_to_save=ppty2ply, color_faces=False)

    # - Filter L1-detected seed:
    L1_detected_topomesh = deepcopy(detected_topomesh)
    L1_detected_cells = np.array(list(L1_detected_topomesh.wisps(0)))[L1_detected_topomesh.wisp_property('layer',0).values()==1]
    non_L1_detected_cells = [c for c in L1_detected_topomesh.wisps(0) if not c in L1_detected_cells]
    for c in non_L1_detected_cells:
        L1_detected_topomesh.remove_wisp(0,c)
    for property_name in L1_detected_topomesh.wisp_property_names(0):
        L1_detected_topomesh.update_wisp_property(property_name,0,array_dict(L1_detected_topomesh.wisp_property(property_name,0).values(list(L1_detected_topomesh.wisps(0))),keys=list(L1_detected_topomesh.wisps(0))))
    # world.add(L1_detected_topomesh,"L1_detected_seed"+suffix)
    # world["L1_detected_seed"+ suffix]["property_name_0"] = 'layer'
    # world["L1_detected_seed{}_vertices".format(suffix)]["polydata_colormap"] = load_colormaps()['Reds']

    # - Evaluate seed detection for all cells:
    evaluation = evaluate_positions_detection(detected_topomesh, corrected_topomesh, max_distance=np.linalg.norm(size*voxelsize))
    evaluations[rescaling] = evaluation

    # -- Evaluate seed detection for L1 filtered seed:
    L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_corrected_topomesh, max_distance=np.linalg.norm(size*voxelsize))
    L1_evaluations[rescaling] = L1_evaluation


eval_fname = image_dirname+"/"+filename+"/"+filename+"_seed_wat_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(evaluations)
evaluation_df.to_csv(eval_fname)

L1_eval_fname = image_dirname+"/"+filename+"/"+filename+"_L1_seed_wat_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(L1_evaluations)
evaluation_df.to_csv(L1_eval_fname)
