import os
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
sys.path.append(SamMaps_dir+'/scripts/lib/')

from equalization import z_slice_contrast_stretch
from equalization import z_slice_equalize_adapthist
from slice_view import slice_view
from slice_view import slice_n_hist
from detection_evaluation import filter_topomesh_vertices
from detection_evaluation import evaluate_positions_detection
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
nuclei_ch_name = "TagBFP"

nuc_path_suffix, nuc_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, nuclei_ch_name)


# EXPERT topomesh = ground truth
#---------------------------------------------------
# - Load the EXPERT topomesh:
expert_topomesh_fname = image_dirname + nuc_path_suffix + nuc_path_suffix[:-1] + "_EXPERT_seed.ply"
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
# EVALUATION of nuclei detection by 'nuclei_image_topomesh'
#-------------------------------------------------------------------------------
evaluations = {}
L1_evaluations = {}

## Detection parameters:
radius_min = 0.8
radius_max = 1.2
threshold = 2000

## Evaluation parameters:
max_matching_distance=2.
outlying_distance=4

std_dev_range = np.concatenate((np.array([0]), np.arange(1.0, 2.0, 0.2)))
rescale_type = ['Original', 'AdaptHistEq', 'ContrastStretch']
for rescaling in rescale_type:
    for iso_resampling in [False, True]:
        for std_dev in std_dev_range:
            # - build a string collecting list of algorithm applied to the nuclei image
            suffix = '-' + rescaling + ('-iso' if iso_resampling else '') + ('-gauss_smooth_{}'.format(std_dev) if std_dev != 0 else '')
            print "\n\nPerforming detection case: {}".format(suffix)
            # - Get the name of the topomesh file:
            topomesh_file = image_dirname + nuc_path_suffix + nuc_path_suffix[:-1] + "{}_nuclei_detection.ply".format(suffix)
            if exists(topomesh_file) and not force:
                print "Found topomesh file:\n{}".format(topomesh_file)
                detected_topomesh = read_ply_property_topomesh(topomesh_file)
            else:
                print "No topomesh file detected, running detection script..."
                img = imread(image_dirname + nuc_path_suffix + nuc_signal_fname)
                if rescaling == 'AdaptHistEq':
                    print "\nPerforming 'z_slice_equalize_adapthist'..."
                    img = z_slice_equalize_adapthist(img)
                    # world.add(img, "reference_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
                if rescaling == 'ContrastStretch':
                    print "\nPerforming 'z_slice_contrast_stretch'..."
                    img = z_slice_contrast_stretch(img)
                    # world.add(img, "reference_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
                # - Performs masked subtraction of signal:
                try:
                    img[mask_img == 0] = 0
                except:
                    pass
                else:
                    print "Applied mask to nuclei signal...",
                if iso_resampling:
                    print "\nPerforming 'isometric_resampling'..."
                    img = isometric_resampling(img)
                if std_dev != 0:
                    print "\nPerforming 'gaussian_smoothing' with std_dev={}...".format(std_dev)
                    img = linear_filtering(img, 'gaussian_smoothing', std_dev=std_dev)

                # - Performs nuclei detection:
                detected_topomesh = nuclei_image_topomesh(dict([(nuclei_ch_name, img)]), reference_name=nuclei_ch_name, signal_names=[], compute_ratios=[], microscope_orientation=microscope_orientation, radius_range=(radius_min,radius_max), threshold=threshold, subsampling=10)
                # detected_positions = detected_topomesh.wisp_property('barycenter',0)
                print "Done"
                save_ply_property_topomesh(detected_topomesh, topomesh_file, properties_to_save=dict([(0,[nuclei_ch_name]+['layer']),(1,[]),(2,[]),(3,[])]), color_faces=False)

            # - Display detected_topomesh:
            # world.add(detected_topomesh, "detected_nuclei"+suffix)
            # world["detected_nuclei"+suffix]["property_name_0"] = 'layer'
            # world["detected_nuclei"+suffix+"_vertices"]["polydata_colormap"] = load_colormaps()['Reds']

            # - Filter L1-detected nuclei:
            L1_detected_topomesh = filter_topomesh_vertices(detected_topomesh, "L1")
            # - Display L1_detected_topomesh:
            # world.add(L1_detected_topomesh,"L1_detected_nuclei"+suffix)
            # world["L1_detected_nuclei"+suffix]["property_name_0"] = 'layer'
            # world["L1_detected_nuclei"+suffix+"_vertices"]["polydata_colormap"] = load_colormaps()['Reds']

            # - Evaluate nuclei detection for all cells:
            print "Evaluate nuclei detection for all cells:"
            evaluation = evaluate_positions_detection(detected_topomesh, expert_topomesh, max_matching_distance=max_matching_distance, outlying_distance=outlying_distance, max_distance=np.linalg.norm(size*img.get_voxelsize()))
            evaluations[suffix] = evaluation

            # - Evaluate nuclei detection for L1 filtered nuclei:
            print "Evaluate nuclei detection for L1 filtered nuclei:"
            L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_topomesh, max_matching_distance=max_matching_distance, outlying_distance=outlying_distance, max_distance=np.linalg.norm(size*img.get_voxelsize()))
            L1_evaluations[suffix] = L1_evaluation


eval_fname = image_dirname + nuc_path_suffix + nuc_path_suffix[:-1] + "_nuclei_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(evaluations)
evaluation_df.to_csv(eval_fname)

L1_eval_fname = image_dirname + nuc_path_suffix + nuc_path_suffix[:-1] + "_L1_nuclei_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(L1_evaluations)
evaluation_df.to_csv(L1_eval_fname)


# evaluation_data = {}
# for field in ['filename','radius_min','radius_max','threshold']:
#     evaluation_data[field] = []
# evaluation_fields = ['Precision','Recall','Jaccard']
# for layer in ['','L1_']:
#     for field in evaluation_fields:
#         evaluation_data[layer+field] = []
#
#
# for radius_min in np.linspace(0.3,1.0,8):
# # for radius_min in [0.8]:
#     min_max = np.maximum(radius_min+0.1,0.8)
#     for radius_max in np.linspace(min_max,min_max+0.7,8):
#     # for radius_max in [1.4]:
#         # for threshold in np.linspace(500,5000,10):
#         for threshold in [2000,3000,4000]:
#
#             evaluation_data['filename'] += [filename]
#             evaluation_data['radius_min'] += [radius_min]
#             evaluation_data['radius_max'] += [radius_max]
#             evaluation_data['threshold'] += [threshold]
#
#             detected_topomesh = nuclei_image_topomesh(dict([(nuclei_ch_name,img)]), nuclei_ch_name=nuclei_ch_name, signal_names=[], compute_ratios=[], microscope_orientation=microscope_orientation, radius_range=(radius_min,radius_max), threshold=threshold)
#             detected_positions = detected_topomesh.wisp_property('barycenter',0)
#
#             world.add(detected_topomesh,"detected_nuclei")
#             world["detected_nuclei"]["property_name_0"] = 'layer'
#             world["detected_nuclei_vertices"]["polydata_colormap"] = load_colormaps()['Reds']
#
#             evaluation = evaluate_positions_detection(detected_topomesh, expert_topomesh, max_matching_distance=2.0, outlying_distance=4.0, max_distance=np.linalg.norm(size*voxelsize))
#
#             for field in evaluation_fields:
#                 evaluation_data[field] += [evaluation[field]]
#
#             L1_detected_topomesh = deepcopy(detected_topomesh)
#             L1_detected_cells = np.array(list(L1_detected_topomesh.wisps(0)))[L1_detected_topomesh.wisp_property('layer',0).values()==1]
#             non_L1_detected_cells = [c for c in L1_detected_topomesh.wisps(0) if not c in L1_detected_cells]
#             for c in non_L1_detected_cells:
#                 L1_detected_topomesh.remove_wisp(0,c)
#             for property_name in L1_detected_topomesh.wisp_property_names(0):
#                 L1_detected_topomesh.update_wisp_property(property_name,0,array_dict(L1_detected_topomesh.wisp_property(property_name,0).values(list(L1_detected_topomesh.wisps(0))),keys=list(L1_detected_topomesh.wisps(0))))
#
#             L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_topomesh, max_matching_distance=2.0, outlying_distance=4.0, max_distance=np.linalg.norm(size*voxelsize))
#
#             for field in evaluation_fields:
#                 evaluation_data['L1_'+field] += [L1_evaluation[field]]
#
# n_points = np.max(map(len,evaluation_data.values()))
# for k in evaluation_df.keys():
#     if len(evaluation_data[k]) == n_points:
#         evaluation_data[k] = evaluation_data[k][:-1]
# evaluation_df = pd.DataFrame().from_dict(evaluation_data)
# evaluation_df.to_csv(image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_evaluation.csv")
