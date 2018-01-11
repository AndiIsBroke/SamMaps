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
from timagetk.components import imread
from timagetk.components import SpatialImage
from timagetk.plugins import linear_filtering, morphology, h_transform, region_labeling, segmentation, registration
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
image_dirname = dirname+"Marie/Lti6b/2017-12-01/"
#image_dirname = dirname+"Marie/Lti6b/2017-12-01/"

# filename = 'DR5N_6.1_151124_sam01_z0.50_t00'
# filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t05'
# filenames = ['Lti6b_xy0.156_z0.32_CH0_iso.inr',
# 'Lti6b_xy0.156_z0.8_CH0_iso.inr',
# 'Lti6b_xy0.156_z0.32_pinH0.34_CH0_iso.inr',
# 'Lti6b_xy0.156_z0.80_pinH0.34_CH0_iso.inr']
filenames = ['Lti6b_xy0.156_z0.8_CH0_iso.inr']

xp_filename = 'Lti6b_xy0.156_z0.156_CH0_iso.inr'
microscope_orientation = 1

image_registration = True

# Corrected image of detected seed = ground truth
#---------------------------------------------------
corrected_filename = image_dirname+"Lti6b_xy0.156_z0.156_CH0_iso_eq_seeds_CORRECTED_topomesh.ply"
# corrected_filename = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_topomesh_corrected_AdaptHistEq.ply"

corrected_topomesh = read_ply_property_topomesh(corrected_filename)
corrected_positions = corrected_topomesh.wisp_property('barycenter',0)
# Convert coordinates into voxel units:
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
# world.add(L1_corrected_topomesh,"L1_corrected_seed")
# world["L1_corrected_seed"]["property_name_0"] = 'layer'
# world["L1_corrected_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

def get_biggest_bounding_box(bboxes):
    """
    Compute the bounding box "size" and return the label for the largest.

    Parameters
    ----------
    bboxes : dict
        dictionary of bounding box (values) with labels as keys

    Returns
    -------
    label : int
        the labelwith the largest bounding box
    """
    label_biggest_bbox = None
    bbox_size = 0
    is2D = len(bboxes.values()[0])==2
    for label, bbox in bboxes.items():
        if is2D:
            x_sl, y_sl = bbox
            size = (x_sl.stop - x_sl.start) * (y_sl.stop - y_sl.start)
        else:
            x_sl, y_sl, z_sl = bbox
            size = (x_sl.stop - x_sl.start) * (y_sl.stop - y_sl.start) * (z_sl.stop - z_sl.start)
        if bbox_size < size:
            bbox_size = size
            label_biggest_bbox = label

    return label_biggest_bbox

def get_background_value(seg_im, microscope_orientation=1):
    """
    Determine the background value using the largewt bounding box.

    Parameters
    ----------
    seg_im : SpatialImage
        SpatialImage for which to determine the background value
    microscope_orientation : int
        For upright microscope use '1' for inverted use (-1)

    Returns
    -------
    background : int
        the labelwith the largest bounding box
    """
    if microscope_orientation == -1:
        top_slice = seg_im[:,:,0]
    else:
        top_slice = seg_im[:,:,-1]
    top_slice_labels = sorted(np.unique(top_slice))
    top_bboxes = nd.find_objects(top_slice, max_label = top_slice_labels[-1])
    top_bboxes = {n+1: top_bbox for n, top_bbox in enumerate(top_bboxes) if top_bbox is not None}

    return get_biggest_bounding_box(top_bboxes)


# EVALUATION
#---------------------------------------------------
evaluations = {}
L1_evaluations={}
## Parameters
std_dev = 2.0
morpho_radius = 3
h_min = 230

# - EXPERT evaluation:
topomesh_file = image_dirname+xp_filename+"_seed_wat_EXPERT_detection_topomesh.ply"
img = imread(image_dirname + xp_filename)
# - Starts by comparing cell barycenters (obtained by segmentation using expert seeds) to expert seed position:
img = isometric_resampling(img)
# world.add(img,"iso_ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
size = np.array(img.shape)
voxelsize = np.array(img.voxelsize)

if exists(topomesh_file):
    detected_topomesh = read_ply_property_topomesh(topomesh_file)
    # - Filter L1-corrected seed (ground truth):
    L1_detected_topomesh = deepcopy(corrected_topomesh)
    L1_corrected_cells = np.array(list(L1_detected_topomesh.wisps(0)))[L1_detected_topomesh.wisp_property('layer',0).values()==1]
    non_L1_corrected_cells = [c for c in L1_detected_topomesh.wisps(0) if not c in L1_corrected_cells]
    for c in non_L1_corrected_cells:
        L1_detected_topomesh.remove_wisp(0,c)
    for property_name in L1_detected_topomesh.wisp_property_names(0):
        L1_detected_topomesh.update_wisp_property(property_name,0,array_dict(L1_detected_topomesh.wisp_property(property_name,0).values(list(L1_detected_topomesh.wisps(0))),keys=list(L1_detected_topomesh.wisps(0))))
else :
    print "Shape: ", img.get_shape(), "; Size: ", img.get_voxelsize()
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
    # Use bounding box to determine background value:
    background = get_background_value(seg_im, microscope_orientation)
    print "Detected background value:", background
    # world.add(seg_im,"seg_image", colormap="glasbey", alphamap="constant",voxelsize=microscope_orientation*voxelsize, bg_id=background)

    # -- Create a vertex_topomesh from detected cell positions:
    # --- Get cell barycenters positions:
    img_graph = graph_from_image(seg_im, background=background, spatio_temporal_properties=['L1', 'barycenter'], ignore_cells_at_stack_margins=True)
    print img_graph.nb_vertices()," cells detected"
    vtx = list(img_graph.vertices())
    labels = img_graph.vertex_property('labels')

    margin_corrected_cells = [c for c in corrected_topomesh.wisps(0) if not c in [labels[v] for v in vtx]]
    for c in margin_corrected_cells:
        corrected_topomesh.remove_wisp(0,c)
    for property_name in corrected_topomesh.wisp_property_names(0):
        corrected_topomesh.update_wisp_property(property_name,0,array_dict(corrected_topomesh.wisp_property(property_name,0).values(list(corrected_topomesh.wisps(0))),keys=list(corrected_topomesh.wisps(0))))
    if margin_corrected_cells:
        ppty2ply = dict([(0,['layer']), (1,[]),(2,[]),(3,[])])
        save_ply_property_topomesh(corrected_topomesh, corrected_filename, properties_to_save=ppty2ply, color_faces=False)

    L1_margin_corrected_cells = [c for c in L1_corrected_topomesh.wisps(0) if not c in [labels[v] for v in vtx]]
    for c in L1_margin_corrected_cells:
        L1_corrected_topomesh.remove_wisp(0,c)
    for property_name in L1_corrected_topomesh.wisp_property_names(0):
        L1_corrected_topomesh.update_wisp_property(property_name,0,array_dict(L1_corrected_topomesh.wisp_property(property_name,0).values(list(L1_corrected_topomesh.wisps(0))),keys=list(L1_corrected_topomesh.wisps(0))))
    # world.add(L1_corrected_topomesh,"L1_corrected_seed")
    # world["L1_corrected_seed"]["property_name_0"] = 'layer'
    # world["L1_corrected_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

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
    # world.add(L1_detected_topomesh,"L1_detected_seed"+suffix)
    # world["L1_detected_seed"+ suffix]["property_name_0"] = 'layer'
    # world["L1_detected_seed{}_vertices".format(suffix)]["polydata_colormap"] = load_colormaps()['Reds']

    ppty2ply = dict([(0,['layer']), (1,[]),(2,[]),(3,[])])
    save_ply_property_topomesh(detected_topomesh, topomesh_file, properties_to_save=ppty2ply, color_faces=False)

# -- Performs evaluation:
evaluation = evaluate_positions_detection(detected_topomesh, corrected_topomesh, max_distance=np.linalg.norm(size*voxelsize))
evaluations['Expert'] = evaluation
L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_corrected_topomesh, max_distance=np.linalg.norm(size*voxelsize))
L1_evaluations['Expert'] = L1_evaluation


# RIGID Registration:
###########################
trsfs = {}
if image_registration:
    for filename in filenames:
        ref_image = imread(image_dirname + xp_filename)
        float_image = imread(image_dirname + filename)
        trsfs[filename], res_img = registration(float_image, ref_image, method="rigid_registration")
        del res_img


def apply_trsf2pts(rigid_trsf, points):
    """
    Function applying a RIGID transformation to a set of points.

    Parameters
    ----------
    rigid_trsf: np.array | BalTransformation
        a quaternion obtained by rigid registration
    points: np.array
        a Nxd list of points to tranform, with d the dimensionality and N the
        number of points
    """
    if isinstance(rigid_trsf, BalTransformation):
        try:
            assert rigid_trsf.isLinear()
        except:
            raise TypeError("The provided transformation is not linear!")
        rigid_trsf = rigid_trsf.mat.to_np_array()
    X, Y, Z = points.T
    homogeneous_points = np.concatenate([np.transpose([X,Y,Z]), np.ones((len(X),1))], axis=1)
    transformed_points = np.einsum("...ij,...j->...i", rigid_trsf, homogeneous_points)

    return transformed_points[:,:3]


for filename in filenames:
    evaluations[filename] = []
    L1_evaluations[filename] = []
    suffix = "_" + filename
    topomesh_file = image_dirname+filename+"_seed_wat_detection_topomesh.ply"
    # reload image, might be ,eccesary for size and voxelsize variables:
    image_filename = image_dirname+filename
    img = imread(image_filename)
    img = isometric_resampling(img)
    # world.add(img,"iso_ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
    size = np.array(img.shape)
    voxelsize = np.array(img.voxelsize)
    if exists(topomesh_file):
        detected_topomesh = read_ply_property_topomesh(topomesh_file)
    else:
        print "Shape: ", img.get_shape(), "; Size: ", img.get_voxelsize()
        # - Performs seed detection:
        smooth_img = linear_filtering(img, std_dev=std_dev, method='gaussian_smoothing')
        asf_img = morphology(img, max_radius=morpho_radius, method='co_alternate_sequential_filter')
        ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
        con_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
        # world.add(con_img, 'labelled_seeds', voxelsize=voxelsize)
        seg_im = segmentation(smooth_img, con_img)
        # world.add(seg_im,"seg"+suffix, colormap="glasbey", voxelsize=microscope_orientation*voxelsize)
        # Use bounding box to determine background value:
        background = get_background_value(seg_im, microscope_orientation)
        print "Detected background value:", background
        img_graph = graph_from_image(seg_im, background=background, spatio_temporal_properties=['L1', 'barycenter'], ignore_cells_at_stack_margins=True)
        print img_graph.nb_vertices()," cells detected"

        vtx = list(img_graph.vertices())
        L1 = img_graph.vertex_property('L1')
        L1_labels = [l for l in vtx if L1[l]]
        bary = img_graph.vertex_property('barycenter')
        cell_layer = {l: L1[l] for l in vtx}
        cell_positions = {v: bary[v]*microscope_orientation for v in vtx}

        detected_topomesh = vertex_topomesh(cell_positions)
        detected_topomesh.update_wisp_property('layer', 0, cell_layer)

        ppty2ply = dict([(0,['layer']), (1,[]),(2,[]),(3,[])])
        save_ply_property_topomesh(detected_topomesh, topomesh_file, properties_to_save=ppty2ply, color_faces=False)

    if image_registration:
        corrected_coords = corrected_topomesh.wisp_property('barycenter', 0).values()
        corrected_coords = apply_trsf2pts(trsfs[filename], corrected_coords)
        corrected_topomesh = vertex_topomesh(corrected_coords)

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
    evaluations[filename] = evaluation

    # -- Evaluate seed detection for L1 filtered seed:
    L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_corrected_topomesh, max_distance=np.linalg.norm(size*voxelsize))
    L1_evaluations[filename] = L1_evaluation


eval_fname = image_dirname+filename+"_seed_wat_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(evaluations)
evaluation_df.to_csv(eval_fname)

L1_eval_fname = image_dirname+filename+"_L1_seed_wat_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(L1_evaluations)
evaluation_df.to_csv(L1_eval_fname)
