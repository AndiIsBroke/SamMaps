import numpy as np
import pandas as pd

from copy import deepcopy
from os.path import exists
from scipy.cluster.vq import vq

from openalea.container import array_dict
# from openalea.image.serial.all import imread
from openalea.mesh.property_topomesh_io import read_ply_property_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh
from openalea.tissue_nukem_3d.nuclei_image_topomesh import nuclei_image_topomesh
from openalea.oalab.colormap.colormap_def import load_colormaps
from timagetk.components import imread
from timagetk.components import SpatialImage

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
from nomenclature import splitext_zip
from nomenclature import get_nomenclature_channel_fname

XP = 'E35'
SAM = 4
tp = 0

image_dirname = dirname + "nuclei_images/"
nomenclature_file = SamMaps_dir + "nomenclature.csv"

# -1- CZI input infos:
base_fname = "qDII-CLV3-PIN1-PI-{}-LD-SAM{}".format(XP, SAM)
czi_fname = base_fname + "-T{}.czi".format(tp)

# -4- Define CZI channel names, the microscope orientation, nuclei and membrane channel names and extra channels that should also be registered:
time_steps = [0, 5, 10, 14]
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
membrane_ch_name = 'PI'
nuclei_ch_name = "TagBFP"

nuc_path_suffix, nuc_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, nuclei_ch_name)

# Original image
#------------------------------
img = imread(image_dirname + nuc_path_suffix + nuc_signal_fname)
size = np.array(img.shape)
voxelsize = np.array(img.voxelsize)

# Mask
#------------------------------
## mask image obtein by maximum intensity projection :
mask_filename = image_dirname + nuc_path_suffix + nuc_path_suffix[:-1] + "_mask.inr.gz"
## 3D mask image obtein by piling a mask for each slice :
if exists(mask_filename):
    mask_img = imread(mask_filename)
else:
    mask_img = np.ones_like(img)

img[mask_img == 0] = 0

# world.add(mask_img,"mask",voxelsize=microscope_orientation*np.array(mask_img.voxelsize),colormap='grey',alphamap='constant',bg_id=255)
# world.add(img,"reference_image",colormap="invert_grey",voxelsize=microscope_orientation*voxelsize)

# Corrected image of detected nuclei = ground truth
#---------------------------------------------------
expert_seed_filename = image_dirname + nuc_path_suffix + nuc_path_suffix[:-1] + "_EXPERT_seed.ply"
expert_seed_topomesh = read_ply_property_topomesh(expert_seed_filename)

# - Get the dictionary of positions:
expert_seed_positions = expert_seed_topomesh.wisp_property('barycenter', 0)

## Mask application :
expert_seed_coords = expert_seed_positions.values()/(microscope_orientation*voxelsize)
expert_seed_coords = np.maximum(0,np.minimum(size-1,expert_seed_coords)).astype(np.uint16)
expert_seed_coords = tuple(np.transpose(expert_seed_coords))
expert_seed_mask_value = mask_img[expert_seed_coords]
expert_seed_cells_to_remove = expert_seed_positions.keys()[expert_seed_mask_value==0]
for c in expert_seed_cells_to_remove:
    expert_seed_topomesh.remove_wisp(0,c)
for property_name in expert_seed_topomesh.wisp_property_names(0):
    expert_seed_topomesh.update_wisp_property(property_name,0,array_dict(expert_seed_topomesh.wisp_property(property_name,0).values(list(expert_seed_topomesh.wisps(0))),keys=list(expert_seed_topomesh.wisps(0))))

# world.add(expert_seed_topomesh,"expert_seeds")
# world["expert_seeds"]["property_name_0"] = 'layer'
# world["expert_seeds_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# - Filter L1-corrected nuclei (ground truth):
L1_expert_seed_topomesh = deepcopy(expert_seed_topomesh)
L1_expert_seed_cells = np.array(list(L1_expert_seed_topomesh.wisps(0)))[L1_expert_seed_topomesh.wisp_property('layer',0).values()==1]
non_L1_expert_seed_cells = [c for c in L1_expert_seed_topomesh.wisps(0) if not c in L1_expert_seed_cells]
for c in non_L1_expert_seed_cells:
    L1_expert_seed_topomesh.remove_wisp(0,c)
for property_name in L1_expert_seed_topomesh.wisp_property_names(0):
    L1_expert_seed_topomesh.update_wisp_property(property_name,0,array_dict(L1_expert_seed_topomesh.wisp_property(property_name,0).values(list(L1_expert_seed_topomesh.wisps(0))),keys=list(L1_expert_seed_topomesh.wisps(0))))

world.add(L1_expert_seed_topomesh,"L1_expert_seeds")
world["L1_expert_seeds"]["property_name_0"] = 'layer'
world["L1_expert_seeds_vertices"]["polydata_colormap"] = load_colormaps()['Greens']


# EVALUATION
#---------------------------------------------------

## Parameters
radius_min = 0.8
radius_max = 1.2
threshold = 2000
max_matching_distance=2.
outlying_distance=4

rescale_type = ['Original', 'AdaptHistEq', 'ContrastStretch']
evaluations = {}
L1_evaluations={}
for rescaling in rescale_type:
    evaluations[rescaling] = []
    L1_evaluations[rescaling] = []
    suffix = "_" + rescaling
    topomesh_file = image_dirname+"/"+filename+"/"+filename+"_{}_nuclei_detection.ply".format(rescaling)
    if exists(topomesh_file):
        detected_topomesh = read_ply_property_topomesh(topomesh_file)
    else:
        if rescaling == 'AdaptHistEq':
            # Need to relaod the orignial image, we don't want to apply histogram equalization technique on masked images
            img = imread(image_filename)
            try:
                vxs = img.voxelsize
            except:
                vxs = img.resolution
            img = z_slice_equalize_adapthist(img)
            img[mask_img == 0] = 0
            img = SpatialImage(img, voxelsize=vxs)
            # world.add(img,"reference_image"+suffix,colormap="invert_grey",voxelsize=microscope_orientation*voxelsize)
        if rescaling == 'ContrastStretch':
            # Need to relaod the orignial image, we don't want to apply histogram equalization technique on masked images
            img = imread(image_filename)
            try:
                vxs = img.voxelsize
            except:
                vxs = img.resolution
            img = z_slice_contrast_stretch(img)
            img[mask_img == 0] = 0
            img = SpatialImage(img, voxelsize=vxs)
            # world.add(img,"reference_image"+suffix,colormap="invert_grey",voxelsize=microscope_orientation*voxelsize)

        # - Performs nuclei detection:
        detected_topomesh = nuclei_image_topomesh(dict([(reference_name,img)]), reference_name=reference_name, signal_names=[], compute_ratios=[], microscope_orientation=microscope_orientation, radius_range=(radius_min,radius_max), threshold=threshold)
        # detected_positions = detected_topomesh.wisp_property('barycenter',0)
        save_ply_property_topomesh(detected_topomesh, topomesh_file, properties_to_save=dict([(0,[reference_name]+['layer']),(1,[]),(2,[]),(3,[])]), color_faces=False)

    world.add(detected_topomesh, "detected_nuclei"+suffix)
    world["detected_nuclei"+suffix]["property_name_0"] = 'layer'
    world["detected_nuclei"+suffix+"_vertices"]["polydata_colormap"] = load_colormaps()['Reds']

    # - Filter L1-detected nuclei:
    L1_detected_topomesh = deepcopy(detected_topomesh)
    L1_detected_cells = np.array(list(L1_detected_topomesh.wisps(0)))[L1_detected_topomesh.wisp_property('layer',0).values()==1]
    non_L1_detected_cells = [c for c in L1_detected_topomesh.wisps(0) if not c in L1_detected_cells]
    for c in non_L1_detected_cells:
        L1_detected_topomesh.remove_wisp(0,c)
    for property_name in L1_detected_topomesh.wisp_property_names(0):
        L1_detected_topomesh.update_wisp_property(property_name,0,array_dict(L1_detected_topomesh.wisp_property(property_name,0).values(list(L1_detected_topomesh.wisps(0))),keys=list(L1_detected_topomesh.wisps(0))))
    # world.add(L1_detected_topomesh,"L1_detected_nuclei"+suffix)
    # world["L1_detected_nuclei"+suffix]["property_name_0"] = 'layer'
    # world["L1_detected_nuclei"+suffix+"_vertices"]["polydata_colormap"] = load_colormaps()['Reds']

    # - Evaluate nuclei detection for all cells:
    evaluation = evaluate_positions_detection(detected_topomesh, expert_seed_topomesh, max_matching_distance=max_matching_distance, outlying_distance=outlying_distance, max_distance=np.linalg.norm(size*voxelsize))
    evaluations[rescaling] = evaluation

    # -- Evaluate nuclei detection for L1 filtered nuclei:
    L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_seed_topomesh, max_matching_distance=max_matching_distance, outlying_distance=outlying_distance, max_distance=np.linalg.norm(size*voxelsize))
    L1_evaluations[rescaling] = L1_evaluation

eval_fname = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(evaluations)
evaluation_df.to_csv(eval_fname)

L1_eval_fname = image_dirname+"/"+filename+"/"+filename+"_L1_nuclei_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(L1_evaluations)
evaluation_df.to_csv(L1_eval_fname)


evaluation_data = {}
for field in ['filename','radius_min','radius_max','threshold']:
    evaluation_data[field] = []
evaluation_fields = ['Precision','Recall','Jaccard']
for layer in ['','L1_']:
    for field in evaluation_fields:
        evaluation_data[layer+field] = []

for radius_min in np.linspace(0.3,1.0,8):
# for radius_min in [0.8]:
    min_max = np.maximum(radius_min+0.1,0.8)
    for radius_max in np.linspace(min_max,min_max+0.7,8):
    # for radius_max in [1.4]:
        # for threshold in np.linspace(500,5000,10):
        for threshold in [2000,3000,4000]:

            evaluation_data['filename'] += [filename]
            evaluation_data['radius_min'] += [radius_min]
            evaluation_data['radius_max'] += [radius_max]
            evaluation_data['threshold'] += [threshold]

            detected_topomesh = nuclei_image_topomesh(dict([(reference_name,img)]), reference_name=reference_name, signal_names=[], compute_ratios=[], microscope_orientation=microscope_orientation, radius_range=(radius_min,radius_max), threshold=threshold)
            detected_positions = detected_topomesh.wisp_property('barycenter',0)

            world.add(detected_topomesh,"detected_nuclei")
            world["detected_nuclei"]["property_name_0"] = 'layer'
            world["detected_nuclei_vertices"]["polydata_colormap"] = load_colormaps()['Reds']

            evaluation = evaluate_positions_detection(detected_topomesh, expert_seed_topomesh, max_matching_distance=2.0, outlying_distance=4.0, max_distance=np.linalg.norm(size*voxelsize))

            for field in evaluation_fields:
                evaluation_data[field] += [evaluation[field]]

            L1_detected_topomesh = deepcopy(detected_topomesh)
            L1_detected_cells = np.array(list(L1_detected_topomesh.wisps(0)))[L1_detected_topomesh.wisp_property('layer',0).values()==1]
            non_L1_detected_cells = [c for c in L1_detected_topomesh.wisps(0) if not c in L1_detected_cells]
            for c in non_L1_detected_cells:
                L1_detected_topomesh.remove_wisp(0,c)
            for property_name in L1_detected_topomesh.wisp_property_names(0):
                L1_detected_topomesh.update_wisp_property(property_name,0,array_dict(L1_detected_topomesh.wisp_property(property_name,0).values(list(L1_detected_topomesh.wisps(0))),keys=list(L1_detected_topomesh.wisps(0))))

            L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_seed_topomesh, max_matching_distance=2.0, outlying_distance=4.0, max_distance=np.linalg.norm(size*voxelsize))

            for field in evaluation_fields:
                evaluation_data['L1_'+field] += [L1_evaluation[field]]

n_points = np.max(map(len,evaluation_data.values()))
for k in evaluation_df.keys():
    if len(evaluation_data[k]) == n_points:
        evaluation_data[k] = evaluation_data[k][:-1]
evaluation_df = pd.DataFrame().from_dict(evaluation_data)
evaluation_df.to_csv(image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_evaluation.csv")
