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
from timagetk.plugins import linear_filtering, morphology, h_transform, region_labeling
from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    sys.path.append('/data/Meristems/Carlos/SamMaps/scripts/TissueLab/')
elif platform.uname()[1] == "RDP-T3600-AL":
    sys.path.append('/home/marie/SamMaps/scripts/TissueLab/')
else:
    raise ValueError("Unknown system...")

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

image_dirname = dirname+"Carlos/nuclei_images"

# filename = 'DR5N_6.1_151124_sam01_z0.50_t00'
# filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t05'
filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t00'
microscope_orientation = -1

# memb_sig_name = "tdT"
memb_sig_name = "PI"
nuc_sig_name = "TagBFP"

memb_sig_fname = image_dirname+"/"+filename+"/"+filename+"_"+memb_sig_name+".inr.gz"
nuc_sig_fname = image_dirname+"/"+filename+"/"+filename+"_"+nuc_sig_name+".inr.gz"

# Original MEMBRANE image
#------------------------------
img = imread(memb_sig_fname)
size = np.array(img.get_shape())
vxs = np.array(img.get_voxelsize())

# Mask
#------------------------------
## mask image obtein by maximum intensity projection :
# mask_filename = image_dirname+"/"+filename+"/"+filename+"_projection_mask.inr.gz"
## 3D mask image obtein by piling a mask for each slice :
mask_filename = image_dirname+"/"+filename+"/"+filename+"_mask.inr.gz"
if os.path.exists(mask_filename):
    mask_img = imread(mask_filename)
    print "Found mask image: '{}'".format(mask_filename)
    img[mask_img == 0] = 0
    img = SpatialImage(img, voxelsize=vxs)

# world.add(mask_img,"mask",voxelsize=microscope_orientation*np.array(mask_img.voxelsize),colormap='grey',alphamap='constant',bg_id=255)
# world.add(img,"reference_image",colormap="invert_grey",voxelsize=microscope_orientation*vxs)


# Expertised detected seed = ground truth
#-------------------------------------------------------------------------------
expert_topomesh_fname = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_topomesh_corrected.ply"
# expert_topomesh_fname = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_topomesh_corrected_AdaptHistEq.ply"
expert_topomesh = read_ply_property_topomesh(expert_topomesh_fname)

# - Create a 'expert_topomesh' (ground truth) out of non-masked cells only:
expert_positions = expert_topomesh.wisp_property('barycenter',0)
# -- Convert coordinates into voxel units:
expert_coords = expert_positions.values()/(microscope_orientation*vxs)
# -- ???
expert_coords = np.maximum(0,np.minimum(size-1,expert_coords)).astype(np.uint16)
expert_coords = tuple(np.transpose(expert_coords))
# -- Detect coordinates outside the mask and remove them:
expert_mask_value = mask_img[expert_coords]
expert_cells_to_remove = expert_positions.keys()[expert_mask_value==0]
expert_topomesh = filter_topomesh_vertices(expert_topomesh, expert_cells_to_remove.tolist())
# world.add(expert_topomesh,"masked_expert_seed")
# world["masked_expert_seed"]["property_name_0"] = 'layer'
# world["masked_expert_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']

# - Create a 'L1_expert_topomesh' (L1 ground truth) out of L1 non-masked cells only:
L1_expert_topomesh = filter_topomesh_vertices(expert_topomesh, "L1")
# world.add(L1_expert_topomesh,"L1_masked_expert_seed")
# world["L1_masked_expert_seed"]["property_name_0"] = 'layer'
# world["L1_masked_expert_seed_vertices"]["polydata_colormap"] = load_colormaps()['Greens']


# EVALUATION of seed detection by 'h_transform' and 'region_labeling':
#-------------------------------------------------------------------------------
evaluations = {}
L1_evaluations={}

## Parameters
radius_min = 0.8
radius_max = 1.2
threshold = 2000

std_dev = 1.0
morpho_radius = 1
h_min = 1700

rescale_type = ['Original', 'AdaptHistEq', 'ContrastStretch']
for rescaling in rescale_type:
    evaluations[rescaling] = []
    L1_evaluations[rescaling] = []
    prefix = "_" + rescaling
    print "\n\n# - Rescale_type : " + rescaling
    for no_nuc in [False, True]:
        suffix = '-nuclei' if no_nuc else ""
        topomesh_file = image_dirname+"/"+filename+"/"+filename+"{}{}_seed_detection_topomesh.ply".format(prefix, suffix)
        # - Read MEMBRANE image filename:
        print "\n# Reading membrane signal image file: {}".format(memb_sig_fname)
        img = imread(memb_sig_fname)
        vxs = np.array(img.get_voxelsize())
        ori = np.array(img.get_origin())
        world.add(img, 'img', colormap='invert_grey')
        # - If topomesh exist use it, else create it
        if exists(topomesh_file):
            print "Found PLY topomesh: '{}'".format(topomesh_file)
            detected_topomesh = read_ply_property_topomesh(topomesh_file)
            size, vxs = isometric_resampling(img, dry_run=True)
            size, vxs = np.array(size), np.array(vxs)
        else:
            # -- Intensity rescaling for membrane image:
            if rescaling == 'AdaptHistEq':
                print " --> Adaptative Histogram Equalization..."
                img = z_slice_equalize_adapthist(img)
            if rescaling == 'ContrastStretch':
                print " --> Contrast Stretching..."
                img = z_slice_contrast_stretch(img)
            # -- Intensity rescaling returns numpy array, convert it to SpatialImage:
            img = SpatialImage(img, voxelsize=vxs, origin=ori)
            # -- Isometric resampling:
            print " --> Interpolation of z-slices..."
            img = isometric_resampling(img)
            size = np.array(img.get_shape())
            vxs = np.array(img.get_voxelsize())
            print "Shape:", size, "; Voxelsize:", vxs
            world.add(img, rescaling+'_'+'img_iso', colormap='invert_grey')
            # - Read NUCLEI image if required (no_nuc == True):
            if no_nuc:
                # -- Reading NUCLEI intensity image:
                print " --> Reading nuclei image file {}".format(nuc_sig_fname)
                nuc_im = imread(nuc_sig_fname)
                print " --> Interpolation of z-slices..."
                nuc_im = isometric_resampling(nuc_im)
                # -- Gaussian Smoothing:
                print " --> Smoothing..."
                nuc_im = linear_filtering(nuc_im, std_dev=2.0,
                                                  method='gaussian_smoothing')
                # -- Remove the NUCLEI signal from the MEMBRANE signal image:
                print " --> Substracting NUCLEI signal from the MEMBRANE signal image..."
                zero_mask = np.greater(nuc_im.get_array(), img.get_array())
                img = np.subtract(img.get_array(), nuc_im.get_array())
                img[zero_mask] = 0
                img = SpatialImage(img, voxelsize=vxs, origin=ori)
                world.add(nuc_im, 'nuc_im', colormap='invert_grey')
                world.add(img, 'img'+suffix, colormap='invert_grey')
                print "Done!"
                del nuc_im

            # - Performs masked subtraction of signal:
            try:
                print "Applying mask to membrane signal...",
                img[mask_img == 0] = 0
            except:
                print "Failed!"
            else:
                img = SpatialImage(img, voxelsize=vxs, origin=ori)
                print "Done!"

            # - Performs seed detection:
            print " --> Smoothing membrane signal..."
            smooth_img = linear_filtering(img, std_dev=std_dev, method='gaussian_smoothing')
            print " --> Closing-Opening Alternate Sequencial Filter..."
            asf_img = morphology(img, max_radius=morpho_radius, method='co_alternate_sequential_filter')
            print " --> H-transform: local minima detection..."
            ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
            print " --> Components labelling..."
            con_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
            nb_connectec_comp = len(np.unique(con_img.get_array()))
            print nb_connectec_comp, "seeds detected!"
            # world.add(con_img, 'labelled_seeds', voxelsize=vxs)
            del ext_img

            # - Performs topomesh creation from detected seeds:
            print " --> Analysing seed image...",
            img_graph = graph_from_image(con_img, background=0, spatio_temporal_properties=['barycenter'], ignore_cells_at_stack_margins=False)
            print img_graph.nb_vertices(),"seeds extracted by 'graph_from_image'!"
            del con_img
            # -- Create a topomesh from 'seed_positions':
            print " --> Topomesh creation..."
            seed_positions = {v: img_graph.vertex_property('barycenter')[v]*microscope_orientation for v in img_graph.vertices()}
            detected_topomesh = vertex_topomesh(seed_positions)
            # -- Detect L1 using 'nuclei_layer':
            oriented_seeds = {k: np.array([1.,1.,-1.])*v for k, v in seed_positions.items()}
            cell_layer = nuclei_layer(oriented_seeds, size*[1,1,2], vxs, subsampling=5.)
            # -- Update the topomesh with the 'layer' property:
            detected_topomesh.update_wisp_property('layer', 0, cell_layer)
            # -- Save the detected topomesh:
            ppty2ply = dict([(0, ['layer']), (1,[]),(2,[]),(3,[])])
            save_ply_property_topomesh(detected_topomesh, topomesh_file, properties_to_save=ppty2ply, color_faces=False)

        # - Create a 'detected_topomesh' out of L1 cells only:
        L1_detected_topomesh = filter_topomesh_vertices(detected_topomesh, "L1")
        # world.add(L1_detected_topomesh, prefix+"L1_detected_seed"+suffix)
        # world[prefix+"L1_detected_seed"+suffix]["property_name_0"] = 'layer'
        # world[prefix+"L1_detected_seed{}_vertices".format(suffix)]["polydata_colormap"] = load_colormaps()['Reds']

        # - Performs evaluation of detected seeds agains expert:
        print "\n# Evaluation: comparison to EXPERT seeds..."
        evaluation = evaluate_positions_detection(detected_topomesh, expert_topomesh, max_distance=np.linalg.norm(size*vxs))
        evaluations[prefix] = evaluation
        L1_evaluation = evaluate_positions_detection(L1_detected_topomesh, L1_expert_topomesh, max_distance=np.linalg.norm(size*vxs))
        L1_evaluations[prefix] = L1_evaluation


eval_fname = image_dirname+"/"+filename+"/"+filename+"_nuclei_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(evaluations)
evaluation_df.to_csv(eval_fname)

L1_eval_fname = image_dirname+"/"+filename+"/"+filename+"_L1_nuclei_detection_eval.csv"
evaluation_df = pd.DataFrame().from_dict(L1_evaluations)
evaluation_df.to_csv(L1_eval_fname)
