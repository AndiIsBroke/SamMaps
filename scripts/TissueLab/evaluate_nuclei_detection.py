import numpy as np
from scipy.cluster.vq import vq

from openalea.container import array_dict
from openalea.image.serial.all import imread
from openalea.mesh.property_topomesh_io import read_ply_property_topomesh
from openalea.tissue_nukem_3d.nuclei_image_topomesh import nuclei_image_topomesh, nuclei_detection
from openalea.oalab.colormap.colormap_def import load_colormaps

from equalization import sl_equalize_adapthist

from timagetk.components import SpatialImage

from copy import deepcopy
from time import time

#from equalization import *

def evaluate_nuclei_detection(nuclei_topomesh, ground_truth_topomesh, max_matching_distance=3.0, outlying_distance=5.0, max_distance=100.):
    """
    Requires the Hungarian library : https://github.com/hrldcpr/hungarian
    """

    from hungarian import lap

    segmentation_ground_truth_matching = vq(nuclei_topomesh.wisp_property('barycenter',0).values(),ground_truth_topomesh.wisp_property('barycenter',0).values())
    ground_truth_segmentation_complete_matching = np.array([vq(nuclei_topomesh.wisp_property('barycenter',0).values(),np.array([p]))[1] for p in ground_truth_topomesh.wisp_property('barycenter',0).values()])

    segmentation_outliers = array_dict(segmentation_ground_truth_matching[1]>outlying_distance+1,nuclei_topomesh.wisp_property('barycenter',0).keys())

    cost_matrix = deepcopy(ground_truth_segmentation_complete_matching)
    if cost_matrix.shape[0]<cost_matrix.shape[1]:
        cost_matrix = np.concatenate([cost_matrix,np.ones((cost_matrix.shape[1]-cost_matrix.shape[0],cost_matrix.shape[1]))*max_distance])
    elif cost_matrix.shape[1]<cost_matrix.shape[0]:
        cost_matrix = np.concatenate([cost_matrix,np.ones((cost_matrix.shape[0],cost_matrix.shape[0]-cost_matrix.shape[1]))*max_distance],axis=1)

    cost_matrix[cost_matrix > outlying_distance] = max_distance

    initial_cost_matrix = deepcopy(cost_matrix)

    start_time = time()
    print "--> Hungarian assignment..."
    assignment = lap(cost_matrix)
    end_time = time()
    print "<-- Hungarian assignment     [",end_time-start_time,"s]"

    ground_truth_assignment = np.arange(ground_truth_topomesh.nb_wisps(0))
    segmentation_assignment = assignment[0][:ground_truth_topomesh.nb_wisps(0)]
    assignment_distances = initial_cost_matrix[(ground_truth_assignment,segmentation_assignment)]
    #print "Assignment : ",assignment_distances.mean()

    evaluation = {}

    evaluation['True Positive'] = (assignment_distances < max_matching_distance).sum()
    evaluation['False Negative'] = (assignment_distances >= max_matching_distance).sum()
    evaluation['False Positive'] = nuclei_topomesh.nb_wisps(0) - segmentation_outliers.values().sum() - evaluation['True Positive']

    evaluation['Precision'] = evaluation['True Positive']/float(evaluation['True Positive']+evaluation['False Positive']) if evaluation['True Positive']+evaluation['False Positive']>0 else 100.
    evaluation['Recall'] = evaluation['True Positive']/float(evaluation['True Positive']+evaluation['False Negative'])
    evaluation['Jaccard'] = evaluation['True Positive']/float(evaluation['True Positive']+evaluation['False Positive']+evaluation['False Negative'])
    evaluation['Dice'] = 2.*evaluation['True Positive']/float(2.*evaluation['True Positive']+evaluation['False Positive']+evaluation['False Negative'])

    print "Precision ",np.round(100.*evaluation['Precision'],2),"%, Recall ",np.round(100.*evaluation['Recall'],2),"%"

    return evaluation


# Files's directories
#-----------------------

dirname = "/home/marie/"

# image_dirname = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/nuclei_ground_truth_images/"
# image_dirname = "/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images"
image_dirname = dirname+"Carlos/nuclei_images"

# filename = 'DR5N_6.1_151124_sam01_z0.50_t00'
# filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t05'
filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t00'

# reference_name = "tdT"
reference_name = "TagBFP"

microscope_orientation = -1

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

mask_img = imread(mask_filename)
img[mask_img == 0] = 0
world.add(mask_img,"mask",voxelsize=microscope_orientation*np.array(mask_img.voxelsize),colormap='grey',alphamap='constant',bg_id=255)
world.add(img,"reference_image",colormap="invert_grey",voxelsize=microscope_orientation*voxelsize)

# Corrected image of detected nuclei = ground truth
#---------------------------------------------------

#corrected_filename = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_topomesh_corrected.ply"
corrected_filename = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_topomesh_corrected_AdaptHistEq.ply"

corrected_topomesh = read_ply_property_topomesh(corrected_filename)
corrected_positions = corrected_topomesh.wisp_property('barycenter',0)

corrected_coords = corrected_positions.values()/(microscope_orientation*voxelsize)
corrected_coords = np.maximum(0,np.minimum(size-1,corrected_coords)).astype(np.uint16)
corrected_coords = tuple(np.transpose(corrected_coords))

## Mask application :

corrected_mask_value = mask_img[corrected_coords]
corrected_cells_to_remove = corrected_positions.keys()[corrected_mask_value==0]
for c in corrected_cells_to_remove:
    corrected_topomesh.remove_wisp(0,c)
for property_name in corrected_topomesh.wisp_property_names(0):
    corrected_topomesh.update_wisp_property(property_name,0,array_dict(corrected_topomesh.wisp_property(property_name,0).values(list(corrected_topomesh.wisps(0))),keys=list(corrected_topomesh.wisps(0))))

# world.add(corrected_topomesh,"corrected_nuclei")
# world["corrected_nuclei"]["property_name_0"] = 'layer'
# world["corrected_nuclei_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# EVALUATION
#---------------------------------------------------

## Parameters
radius_min = 0.8
radius_max = 1.4
threshold = 1000

evaluations=[]
L1_evaluations=[]
for rescaling in [False, True]:
    suffix = ""
    if rescaling:
        suffix = "_AdaptHistEq"
        img = imread(image_filename)
        img = sl_equalize_adapthist(img)
        img[mask_img == 0] = 0
        try:
            vxs = img.voxelsize
        except:
            vxs = img.resolution
        img = SpatialImage(img, voxelsize=vxs)
        world.add(img,"reference_image"+suffix,colormap="invert_grey",voxelsize=microscope_orientation*voxelsize)

    detected_topomesh = nuclei_image_topomesh(dict([(reference_name,img)]), reference_name=reference_name, signal_names=[], compute_ratios=[], microscope_orientation=microscope_orientation, radius_range=(radius_min,radius_max), threshold=threshold)
    detected_positions = detected_topomesh.wisp_property('barycenter',0)

    # world.add(detected_topomesh,"detected_nuclei"+suffix)
    # world["detected_nuclei"]["property_name_0"] = 'layer'
    # world["detected_nuclei_vertices"]["polydata_colormap"] = load_colormaps()['Reds']

    evaluation = evaluate_nuclei_detection(detected_topomesh, corrected_topomesh, max_matching_distance=2.0, outlying_distance=4.0, max_distance=np.linalg.norm(size*voxelsize))
    evaluations.append(evaluation)

    ## Evaluation of the L1 cells only
    ## L1-corrected (ground truth):
    L1_corrected_topomesh = deepcopy(corrected_topomesh)
    L1_corrected_cells = np.array(list(L1_corrected_topomesh.wisps(0)))[L1_corrected_topomesh.wisp_property('layer',0).values()==1]
    non_L1_corrected_cells = [c for c in L1_corrected_topomesh.wisps(0) if not c in L1_corrected_cells]
    for c in non_L1_corrected_cells:
        L1_corrected_topomesh.remove_wisp(0,c)
    for property_name in L1_corrected_topomesh.wisp_property_names(0):
        L1_corrected_topomesh.update_wisp_property(property_name,0,array_dict(L1_corrected_topomesh.wisp_property(property_name,0).values(list(L1_corrected_topomesh.wisps(0))),keys=list(L1_corrected_topomesh.wisps(0))))

    world.add(L1_corrected_topomesh,"L1_corrected_nuclei"+suffix)
    world["L1_corrected_nuclei"]["property_name_0"] = 'layer'
    world["L1_corrected_nuclei_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

    ## L1-detected:
    L1_detected_topomesh = deepcopy(detected_topomesh)
    L1_detected_cells = np.array(list(L1_detected_topomesh.wisps(0)))[L1_detected_topomesh.wisp_property('layer',0).values()==1]
    non_L1_detected_cells = [c for c in L1_detected_topomesh.wisps(0) if not c in L1_detected_cells]
    for c in non_L1_detected_cells:
        L1_detected_topomesh.remove_wisp(0,c)
    for property_name in L1_detected_topomesh.wisp_property_names(0):
        L1_detected_topomesh.update_wisp_property(property_name,0,array_dict(L1_detected_topomesh.wisp_property(property_name,0).values(list(L1_detected_topomesh.wisps(0))),keys=list(L1_detected_topomesh.wisps(0))))

    world.add(L1_detected_topomesh,"L1_detected_nuclei"+suffix)
    world["L1_detected_nuclei"]["property_name_0"] = 'layer'
    world["L1_detected_nuclei_vertices"]["polydata_colormap"] = load_colormaps()['Reds']

    L1_evaluation = evaluate_nuclei_detection(L1_detected_topomesh, L1_corrected_topomesh, max_matching_distance=2.0, outlying_distance=4.0, max_distance=np.linalg.norm(size*voxelsize))
    L1_evaluations.append(L1_evaluation)
