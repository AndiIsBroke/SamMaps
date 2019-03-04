import numpy as np
import pandas as pd
import scipy.ndimage as nd

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
from timagetk.io import imread
from timagetk.components import SpatialImage
from timagetk.plugins import linear_filtering
from timagetk.plugins import morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling
from timagetk.plugins import segmentation
from timagetk.plugins import registration

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
from detection_evaluation import get_biggest_bounding_box
from detection_evaluation import get_background_value
from detection_evaluation import apply_trsf2pts
from detection_evaluation import filter_topomesh_vertices

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
# filenames = ['Lti6b_xy0.156_z0.8_CH0_iso.inr',
# 'Lti6b_xy0.156_z0.32_CH0_iso.inr',
# 'Lti6b_xy0.156_z0.156_CH0_iso.inr',
# 'Lti6b_xy0.156_z0.32_pinH0.34_CH0_iso.inr',
# 'Lti6b_xy0.156_z0.80_pinH0.34_CH0_iso.inr']

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
xp_topomesh_fname = image_dirname+"/"+filename+"/"+filename+"_EXPERT_seed.ply"
# xp_topomesh_fname = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_topomesh_corrected_AdaptHistEq.ply"

expert_topomesh = read_ply_property_topomesh(xp_topomesh_fname)
expert_positions = expert_topomesh.wisp_property('barycenter',0)
# Convert coordinates into voxel units:
expert_coords = expert_positions.values()/(microscope_orientation*voxelsize)
# ???
expert_coords = np.maximum(0,np.minimum(size-1,expert_coords)).astype(np.uint16)
expert_coords = tuple(np.transpose(expert_coords))

## Mask application :
expert_mask_value = mask_img[expert_coords]
expert_cells_to_remove = expert_positions.keys()[expert_mask_value==0]
for c in expert_cells_to_remove:
    expert_topomesh.remove_wisp(0,c)
for property_name in expert_topomesh.wisp_property_names(0):
    expert_topomesh.update_wisp_property(property_name,0,array_dict(expert_topomesh.wisp_property(property_name,0).values(list(expert_topomesh.wisps(0))),keys=list(expert_topomesh.wisps(0))))

# world.add(expert_topomesh,"expert_seed")
# world["expert_seed"]["property_name_0"] = 'layer'
# world["expert_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# - Filter L1-corrected seed (ground truth):
L1_expert_topomesh = deepcopy(expert_topomesh)
L1_expert_cells = np.array(list(L1_expert_topomesh.wisps(0)))[L1_expert_topomesh.wisp_property('layer',0).values()==1]
non_L1_expert_cells = [c for c in L1_expert_topomesh.wisps(0) if not c in L1_expert_cells]
for c in non_L1_expert_cells:
    L1_expert_topomesh.remove_wisp(0,c)
for property_name in L1_expert_topomesh.wisp_property_names(0):
    L1_expert_topomesh.update_wisp_property(property_name,0,array_dict(L1_expert_topomesh.wisp_property(property_name,0).values(list(L1_expert_topomesh.wisps(0))),keys=list(L1_expert_topomesh.wisps(0))))
world.add(L1_expert_topomesh,"L1_expert_seed")
world["L1_expert_seed"]["property_name_0"] = 'layer'
world["L1_expert_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']


# EVALUATION
#---------------------------------------------------
evaluations = {}
L1_evaluations={}
## Parameters
std_dev = 2.0
morpho_radius = 3
h_min = 230

# - Starts by comparing cell barycenters (obtained by segmentation using expert seeds) to expert seed position:
img = isometric_resampling(img)
# world.add(img,"iso_ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
size = np.array(img.shape)
voxelsize = np.array(img.voxelsize)
print "Shape: ", img.get_shape(), "; Size: ", img.get_voxelsize()

# -- Create a seed image from expertised seed positions:
xp_seed_pos = expert_topomesh.wisp_property('barycenter', 0)
xp_seed_pos = {k:v*microscope_orientation for k, v in xp_seed_pos.items()}
# --- Change following values, as required by watershed algorithm:
#  - '0': watershed will fill these with other label
#  - '1': background value (outside the biological object)
for label in [0, 1]:
    if xp_seed_pos.has_key(label):
        mk = max(xp_seed_pos.keys())
        xp_seed_pos[mk + 1] = xp_seed_pos[label]
        xp_seed_pos.pop(label)

# --- Create the seed image:
con_img = seed_image_from_points(size, voxelsize, xp_seed_pos, 2., 0)
# --- Add background position:
background_threshold = 2000.
smooth_img_bck = linear_filtering(img, std_dev=3.0, method='gaussian_smoothing')
background_img = (smooth_img_bck < background_threshold).astype(np.uint16)
for it in xrange(15):
    background_img = morphology(background_img, param_str_2 = '-operation erosion -iterations 10')

connected_background_components, n_components = nd.label(background_img)
components_area = nd.sum(np.ones_like(connected_background_components),connected_background_components,index=np.arange(n_components)+1)
largest_component = (np.arange(n_components)+1)[np.argmax(components_area)]
background_img = (connected_background_components == largest_component).astype(np.uint16)

con_img[background_img==1] = 1
del smooth_img_bck, background_img
con_img = SpatialImage(con_img, voxelsize=voxelsize)
# world.add(con_img,"seed_image", colormap="glasbey", alphamap="constant",voxelsize=microscope_orientation*voxelsize, bg_id=0)
# -- Performs automatic seeded watershed using previously created seed image:
smooth_img = linear_filtering(img, std_dev=std_dev, method='gaussian_smoothing')
seg_im = segmentation(smooth_img, con_img)
# world.add(seg_im,"seg_image", colormap="glasbey", alphamap="constant",voxelsize=microscope_orientation*voxelsize)
# -- Create a vertex_topomesh from detected cell positions:
# --- Get cell barycenters positions:
img_graph = graph_from_image(seg_im, background=1, spatio_temporal_properties=['L1', 'barycenter'], ignore_cells_at_stack_margins=True)
print img_graph.nb_vertices()," cells detected"
vtx = list(img_graph.vertices())
labels = img_graph.vertex_property('labels')

margin_expert_cells = [c for c in expert_topomesh.wisps(0) if not c in [labels[v] for v in vtx]]
for c in margin_expert_cells:
    expert_topomesh.remove_wisp(0,c)
for property_name in expert_topomesh.wisp_property_names(0):
    expert_topomesh.update_wisp_property(property_name,0,array_dict(expert_topomesh.wisp_property(property_name,0).values(list(expert_topomesh.wisps(0))),keys=list(expert_topomesh.wisps(0))))

L1_margin_expert_cells = [c for c in L1_expert_topomesh.wisps(0) if not c in [labels[v] for v in vtx]]
for c in L1_margin_expert_cells:
    L1_expert_topomesh.remove_wisp(0,c)
for property_name in L1_expert_topomesh.wisp_property_names(0):
    L1_expert_topomesh.update_wisp_property(property_name,0,array_dict(L1_expert_topomesh.wisp_property(property_name,0).values(list(L1_expert_topomesh.wisps(0))),keys=list(L1_expert_topomesh.wisps(0))))
world.add(L1_expert_topomesh,"L1_expert_seed")
world["L1_expert_seed"]["property_name_0"] = 'layer'
world["L1_expert_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

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
suffix = "_expert"
world.add(L1_detected_topomesh,"L1_detected_seed"+suffix)
world["L1_detected_seed"+ suffix]["property_name_0"] = 'layer'
world["L1_detected_seed{}_vertices".format(suffix)]["polydata_colormap"] = load_colormaps()['Reds']


# -- Performs evaluation:
evaluation = evaluate_positions_detection(detected_topomesh, expert_topomesh, max_distance=np.linalg.norm(size*voxelsize))
evaluations['Expert'] = evaluation
L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_topomesh, max_distance=np.linalg.norm(size*voxelsize))
L1_evaluations['Expert'] = L1_evaluation


# - Now evaluate the effect of contrast stretching techniques on automatic seeded watershed algorithm
rescale_type = ['Original', 'AdaptHistEq', 'ContrastStretch']
for rescaling in rescale_type:
    print "rescale_type : " + rescaling
    evaluations[rescaling] = []
    L1_evaluations[rescaling] = []
    suffix = "_" + rescaling
    topomesh_file = image_dirname+"/"+filename+"/"+filename+"_{}_hmin{}_seed_wat_detection_topomesh.ply".format(rescaling,h_min)
    if exists(topomesh_file):
        detected_topomesh = read_ply_property_topomesh(topomesh_file)
    else:
        # Need to relaod the orignial image, we don't want to apply histogram equalization technique on masked images
        img = imread(image_filename)
        voxelsize = np.array(img.voxelsize)
        if rescaling == 'AdaptHistEq':
            img = z_slice_equalize_adapthist(img)
        elif rescaling == 'ContrastStretch':
            img = z_slice_contrast_stretch(img)
        else:
            pass
        img[mask_img == 0] = 0
        img = SpatialImage(img, voxelsize=voxelsize)
        # world.add(img,"ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
        img = isometric_resampling(img)
        # world.add(img,"iso_ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
        size = np.array(img.get_shape())
        voxelsize = np.array(img.get_voxelsize())
        print "Shape:", size, "; Voxelsize:", voxelsize

        # - Performs seed detection:
        print "\n# - Automatic seed detection..."
        smooth_img = linear_filtering(img, std_dev=std_dev, method='gaussian_smoothing')
        asf_img = morphology(img, max_radius=morpho_radius, method='co_alternate_sequential_filter')
        ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
        con_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
        # world.add(con_img, 'labelled_seeds', voxelsize=voxelsize)
        print "\n# - Seeded watershed from automatic seed detection..."
        seg_im = segmentation(smooth_img, con_img)
        #world.add(seg_im,"seg_image"+suffix, colormap="glasbey", voxelsize=microscope_orientation*voxelsize)
        # Use bounding box to determine background value:
        background = get_background_value(seg_im, microscope_orientation)
        print "Detected background value:", background
        # -- Create a vertex_topomesh from detected cell positions:
        print "\n# - Extracting 'barycenter' & 'L1' properties from segmented image..."
        # --- Compute 'L1' and 'barycenter' properties using 'graph_from_image'
        img_graph = graph_from_image(seg_im, background=1, spatio_temporal_properties=['L1', 'barycenter'], ignore_cells_at_stack_margins=True)
        print img_graph.nb_vertices()," cells detected"
        print "\n# - Creating a vertex_topomesh..."
        vtx = list(img_graph.vertices())
        vtx2labels = img_graph.vertex_property('labels')
        # --- Get cell barycenters positions and L1 cells:
        bary = img_graph.vertex_property('barycenter')
        in_L1 = img_graph.vertex_property('L1')
        # --- Create a topomesh using detected cell barycenters:
        label_positions = {l: bary[v]*microscope_orientation for v,l in vtx2labels.items()}
        detected_topomesh = vertex_topomesh(label_positions)
        # --- Add the 'layer' property to the topomesh:
        label_layer = {l: in_L1[v] for v,l in vtx2labels.items()}
        detected_topomesh.add_wisp_property('layer', 0, label_layer)
        # --- Save the detected topomesh:
        ppty2ply = dict([(0, ['layer']), (1,[]),(2,[]),(3,[])])
        save_ply_property_topomesh(detected_topomesh, topomesh_file, properties_to_save=ppty2ply, color_faces=False)

    # - Filter L1-detected seed:
    L1_detected_topomesh = filter_topomesh_vertices(detected_topomesh, "L1")
    # world.add(L1_detected_topomesh,"L1_detected_seed"+suffix)
    # world["L1_detected_seed"+ suffix]["property_name_0"] = 'layer'
    # world["L1_detected_seed{}_vertices".format(suffix)]["polydata_colormap"] = load_colormaps()['Reds']

    # - Evaluate seed detection for all cells:
    evaluation = evaluate_positions_detection(detected_topomesh, expert_topomesh, max_distance=np.linalg.norm(size*voxelsize))
    evaluations[rescaling] = evaluation

    # -- Evaluate seed detection for L1 filtered seed:
    L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_topomesh, max_distance=np.linalg.norm(size*voxelsize))
    L1_evaluations[rescaling] = L1_evaluation


eval_fname = image_dirname+"/"+filename+"/"+filename+"_seed_wat_detection_eval_hmin{}.csv".format(h_min)
evaluation_df = pd.DataFrame().from_dict(evaluations)
evaluation_df.to_csv(eval_fname)

L1_eval_fname = image_dirname+"/"+filename+"/"+filename+"_L1_seed_wat_detection_eval_hmin{}.csv".format(h_min)
evaluation_df = pd.DataFrame().from_dict(L1_evaluations)
evaluation_df.to_csv(L1_eval_fname)
