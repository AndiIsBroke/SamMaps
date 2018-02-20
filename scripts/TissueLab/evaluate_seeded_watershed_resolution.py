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
from detection_evaluation import filter_topomesh_vertices
from detection_evaluation import apply_trsf2pts
from detection_evaluation import get_background_value
from detection_evaluation import get_biggest_bounding_box


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
filenames = ['Lti6b_xy0.156_z0.32_CH0_iso.inr',
'Lti6b_xy0.156_z0.8_CH0_iso.inr',
'Lti6b_xy0.156_z0.32_pinH0.34_CH0_iso.inr',
'Lti6b_xy0.156_z0.80_pinH0.34_CH0_iso.inr']
xp_filename = 'Lti6b_xy0.156_z0.156_CH0_iso.inr'
microscope_orientation = 1
image_registration = True


# Corrected image of detected seed = ground truth
#---------------------------------------------------
xp_topomesh_fname = image_dirname+"Lti6b_xy0.156_z0.156_CH0_iso_eq_seeds_CORRECTED_topomesh.ply"


expert_topomesh = read_ply_property_topomesh(xp_topomesh_fname)
# world.add(expert_topomesh,"corrected_seed")
# world["corrected_seed"]["property_name_0"] = 'layer'
# world["corrected_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# - Filter L1 expert seed (ground truth):
L1_expert_topomesh = filter_topomesh_vertices(expert_topomesh, "L1")
# world.add(L1_expert_topomesh,"L1_expert_seeds")
# world["L1_expert_seeds"]["property_name_0"] = 'layer'
# world["L1_expert_seeds_vertices"]["polydata_colormap"] = load_colormaps()['Greens']


# EVALUATION
#---------------------------------------------------
evaluations = {}
L1_evaluations={}
## Parameters
std_dev = 2.0
morpho_radius = 3
h_min = 230

# - EXPERT evaluation:
topomesh_file = image_dirname + xp_filename[:-4] + "_seed_wat_EXPERT_detection_topomesh.ply"
img = imread(image_dirname + xp_filename)
img = isometric_resampling(img)
# world.add(img,"iso_ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
size = np.array(img.shape)
voxelsize = np.array(img.voxelsize)

if exists(topomesh_file):
    detected_topomesh = read_ply_property_topomesh(topomesh_file)
else :
    print "Shape: ", img.get_shape(), "; Size: ", img.get_voxelsize()
    # -- Change following values, as required by watershed algorithm:
    #  - '0': watershed will fill these with other label
    #  - '1': background value (outside the biological object)
    vtx = list(expert_topomesh.wisps(0))
    if 0 in vtx or 1 in vtx:
        # --- Initialise relabelling dictionary:
        relabel = {v: v for v in vtx}
        # --- Change label values for 0 & 1:
        for label in [0, 1]:
            mk = max(relabel.values())
            relabel[label] = mk+1
        # --- Create a temporary expert topomesh for label edition:
        expert_positions = expert_topomesh.wisp_property('barycenter',0)
        expert_positions = {relabel[k]: v for k, v in expert_positions.items()}
        tmp_expert_topomesh = vertex_topomesh(expert_positions)
        # --- Relabel all existing properties:
        for ppty in expert_topomesh.wisp_property_names(0):
            ppty_dict = array_dict(expert_topomesh.wisp_property(ppty, 0).values(vtx), keys=vtx)
            ppty_dict = {relabel[k]: v for k, v in ppty_dict.items()}
            tmp_expert_topomesh.update_wisp_property(ppty, 0, ppty_dict)
        try:
            assert tmp_expert_topomesh.has_wisp_property('layer', 0, True)
        except AssertionError:
            raise ValueError("Error during relabelling, please check!")
        else:
            expert_topomesh = tmp_expert_topomesh
    # -- Create a seed image from expertised seed positions:
    print "\n# - Creating seed image from EXPERT seed positions..."
    xp_seed_pos = expert_topomesh.wisp_property('barycenter', 0)
    xp_seed_pos = {k: v*microscope_orientation for k, v in xp_seed_pos.items()}
    # --- Create the seed image:
    seed_img = seed_image_from_points(size, voxelsize, xp_seed_pos, 2., 0)
    # --- Add background position:
    background_threshold = 2000.
    smooth_img_bck = linear_filtering(img, std_dev=3.0, method='gaussian_smoothing')
    background_img = (smooth_img_bck < background_threshold).astype(np.uint16)
    for it in xrange(15):
        background_img = morphology(background_img, param_str_2 = '-operation erosion -iterations 10')
    # ---- Detect small regions defined as background and remove them:
    connected_background_components, n_components = nd.label(background_img)
    components_area = nd.sum(np.ones_like(connected_background_components), connected_background_components, index=np.arange(n_components)+1)
    largest_component = (np.arange(n_components)+1)[np.argmax(components_area)]
    background_img = (connected_background_components == largest_component).astype(np.uint16)
    # ---- Finaly add the background and make a SpatialImage:
    seed_img[background_img==1] = 1
    del smooth_img_bck, background_img
    seed_img = SpatialImage(seed_img, voxelsize=voxelsize)
    # world.add(seed_img,"seed_image", colormap="glasbey", alphamap="constant",voxelsize=microscope_orientation*voxelsize, bg_id=0)

    # -- Performs automatic seeded watershed using previously created seed image:
    print "\n# - Seeded watershed using seed EXPERT seed positions..."
    smooth_img = linear_filtering(img, std_dev=std_dev, method='gaussian_smoothing')
    seg_im = segmentation(smooth_img, seed_img)
    # Use largest bounding box to determine background value:
    background = get_background_value(seg_im, microscope_orientation)
    print "Detected background value:", background
    # world.add(seg_im,"seg_image", colormap="glasbey", alphamap="constant",voxelsize=microscope_orientation*voxelsize, bg_id=background)

    # -- Create a vertex_topomesh from detected cell positions:
    print "\n# - Extracting 'barycenter' & 'L1' properties from segmented image..."
    # --- Compute 'L1' and 'barycenter' properties using 'graph_from_image'
    img_graph = graph_from_image(seg_im, background=background, spatio_temporal_properties=['L1', 'barycenter'], ignore_cells_at_stack_margins=True)
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
    # -- Add the 'marginal' property to the EXPERT topomesh ('expert_topomesh'):
    margin_cells = list(set(xp_seed_pos) - set(vtx2labels.values()))
    if margin_cells:
        print "Found {} marginal cells...".format(len(margin_cells))
        try:
            expert_topomesh.add_wisp_property('marginal', 0, {l: l in margin_cells for l in expert_topomesh.wisps(0)})
        except:
            expert_topomesh.update_wisp_property('marginal', 0, {l: l in margin_cells for l in expert_topomesh.wisps(0)})
        ppty2ply = dict([(0, ['layer', 'marginal']), (1,[]),(2,[]),(3,[])])
        save_ply_property_topomesh(expert_topomesh, xp_topomesh_fname, properties_to_save=ppty2ply, color_faces=False)

# -- Edit 'expert_topomesh' (ground truth) for potential labels at the stack margins:
margin_cells = [k for k, v in expert_topomesh.wisp_property('marginal', 0).items() if v]
non_margin_cells = list(set(expert_topomesh.wisps(0)) - set(margin_cells))
expert_topomesh = filter_topomesh_vertices(expert_topomesh, non_margin_cells)
# --- Update EXPERT topomesh display:
world.add(expert_topomesh,"expert_seeds")
world["expert_seeds"]["property_name_0"] = 'layer'
world["expert_seeds_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# -- Create a 'detected_topomesh' out of L1 cells only:
L1_detected_topomesh = filter_topomesh_vertices(detected_topomesh, "L1")
suffix = "_expert"
# world.add(L1_detected_topomesh,"L1_detected_seed"+suffix)
# world["L1_detected_seed"+ suffix]["property_name_0"] = 'layer'
# world["L1_detected_seed{}_vertices".format(suffix)]["polydata_colormap"] = load_colormaps()['Reds']

# -- Create a 'L1_expert_topomesh' (L1 ground truth) out of L1 cells only:
L1_expert_topomesh = filter_topomesh_vertices(expert_topomesh, "L1")
world.add(L1_expert_topomesh,"L1_expert_seeds")
world["L1_expert_seeds"]["property_name_0"] = 'layer'
world["L1_expert_seeds_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# -- Performs evaluation:
evaluation = evaluate_positions_detection(detected_topomesh, expert_topomesh, max_distance=np.linalg.norm(size*voxelsize))
evaluations['Expert'] = evaluation
L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_topomesh, max_distance=np.linalg.norm(size*voxelsize))
L1_evaluations['Expert'] = L1_evaluation


# RIGID Registration:
###########################
trsfs = {}
if image_registration:
    for filename in filenames:
        trsf_fname = image_dirname + filename[:-4] + "_rigid_to_expert.trsf"
        if exists(trsf_fname):
            trsfs[filename] = np.loadtxt(trsf_fname)
        else:
            float_image = imread(image_dirname + xp_filename)
            ref_image = imread(image_dirname + filename)
            trsfs[filename], res_img = registration(float_image, ref_image, method="rigid_registration", pyramid_lowest_level=1)
            del res_img
            mat = trsfs[filename].mat.to_np_array()
            np.savetxt(trsf_fname, mat, fmt='%1.8f')


for filename in filenames:
    print "\n\n# Comparing expert seeds to auto seeds from {}...".format(filename)
    evaluations[filename] = []
    L1_evaluations[filename] = []
    suffix = "_" + filename
    topomesh_file = image_dirname + filename + "_seed_wat_detection_topomesh.ply"
    # reload image, might be necessary for size and voxelsize variables:
    image_filename = image_dirname + filename
    img = imread(image_filename)
    img = isometric_resampling(img)
    # world.add(img, "iso_ref_image"+suffix, colormap="invert_grey", voxelsize=microscope_orientation*voxelsize)
    size = np.array(img.get_shape())
    voxelsize = np.array(img.get_voxelsize())
    if exists(topomesh_file):
        detected_topomesh = read_ply_property_topomesh(topomesh_file)
    else:
        print "Shape:", size, "; Voxelsize:", voxelsize
        # - Performs seed detection:
        print "\n# - Automatic seed detection..."
        smooth_img = linear_filtering(img, std_dev=std_dev, method='gaussian_smoothing')
        asf_img = morphology(img, max_radius=morpho_radius, method='co_alternate_sequential_filter')
        ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
        seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
        # world.add(seed_img, 'labelled_seeds', voxelsize=voxelsize)
        print "\n# - Seeded watershed from automatic seed detection..."
        seg_im = segmentation(smooth_img, seed_img)
        # world.add(seg_im,"seg"+suffix, colormap="glasbey", voxelsize=microscope_orientation*voxelsize)

        # Use bounding box to determine background value:
        background = get_background_value(seg_im, microscope_orientation)
        print "Detected background value:", background
        # -- Create a vertex_topomesh from detected cell positions:
        print "\n# - Extracting 'barycenter' & 'L1' properties from segmented image..."
        # --- Compute 'L1' and 'barycenter' properties using 'graph_from_image'
        img_graph = graph_from_image(seg_im, background=background, spatio_temporal_properties=['L1', 'barycenter'], ignore_cells_at_stack_margins=True)
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

    if image_registration:
        print "\n# - Rigid registration for detected seeds..."
        # - Get expertised cell barycenters to apply the rigid transformation:
        corrected_coords = detected_topomesh.wisp_property('barycenter', 0).values()
        rigid_coords = apply_trsf2pts(trsfs[filename], corrected_coords)
        # - Re-create the topomesh:
        rigid_topomesh = vertex_topomesh(rigid_coords)
        # - Get other properties from the original topomesh 'expert_topomesh'
        vtx = list(detected_topomesh.wisps(0))
        for ppty in detected_topomesh.wisp_property_names(0):
            if ppty == 'barycenter':
                continue
            rigid_topomesh.add_wisp_property(ppty, 0, array_dict(detected_topomesh.wisp_property(ppty, 0).values(vtx), keys=vtx))
        detected_topomesh = rigid_topomesh
        # world.add(detected_topomesh, "rigid_detected_topomesh"+suffix)
        # world["rigid_detected_topomesh"+suffix]["property_name_0"] = 'layer'
        # world["rigid_detected_topomesh"+suffix]["polydata_colormap"] = load_colormaps()['Blues']
        # - Filter L1 seeds for rigid registered topomesh:
        L1_detected_topomesh = filter_topomesh_vertices(rigid_topomesh, "L1")
        world.add(L1_detected_topomesh,"L1_rigid_detected_seed"+suffix)
        world["L1_rigid_detected_seed"+ suffix]["property_name_0"] = 'layer'
        world["L1_rigid_detected_seed{}_vertices".format(suffix)]["polydata_colormap"] = load_colormaps()['Reds']

    # - Evaluate seeds detection
    # -- for all cells:
    evaluation = evaluate_positions_detection(detected_topomesh, expert_topomesh, max_distance=np.linalg.norm(size*voxelsize))
    evaluations[filename] = evaluation
    # -- for L1 filtered seed:
    L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_topomesh, max_distance=np.linalg.norm(size*voxelsize))
    L1_evaluations[filename] = L1_evaluation


eval_fname = image_dirname+"_seed_wat_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(evaluations)
evaluation_df.to_csv(eval_fname)

L1_eval_fname = image_dirname+"_L1_seed_wat_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(L1_evaluations)
evaluation_df.to_csv(L1_eval_fname)
