# -*- coding: utf-8 -*-
from mayavi import mlab
from os.path import split
import numpy as np
import pandas as pd
import scipy.ndimage as nd

from timagetk.components.io import imread, imsave, SpatialImage
# from openalea.tissue_nukem_3d.microscopy_images import imread as read_czi
from timagetk.algorithms import isometric_resampling
from timagetk.algorithms.resample import resample

from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
from vplants.tissue_analysis.signal_quantification import MembraneQuantif
from vplants.tissue_analysis.misc import rtuple

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

from nomenclature import splitext_zip
from nomenclature import get_nomenclature_channel_fname
from nomenclature import get_nomenclature_segmentation_name
from nomenclature import get_res_img_fname
from nomenclature import get_res_trsf_fname


# XP = 'E35'
# SAM = '4'
# tp = 0
# membrane_dist = 0.6
XP = sys.argv[1]
SAM = sys.argv[2]
tp = int(sys.argv[3])
membrane_dist = sys.argv[4]

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

quantif_method = "mean"
walls_min_area = 5.  # to avoid too small walls arising from segmentation errors
# - If you want to plot PIN signal image AND polarity field, you should use barycenters with voxel units:
real_bary = False


# By default we register all other channels:
extra_channels = list(set(channel_names) - set([membrane_ch_name]))
# By default do not recompute deformation when an associated file exist:
force = True

# Get unregistered image filename:
path_suffix, PI_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, membrane_ch_name)
path_suffix, PIN_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, 'PIN1')
path_suffix, seg_img_fname = get_nomenclature_segmentation_name(czi_fname, nomenclature_file, membrane_ch_name + "_raw")


# if tp != time_steps[-1]:
#     path_suffix += 'rigid_registrations/'
#     # Get RIDIG registered on last time-point filename:
#     PI_signal_fname = get_res_img_fname(PI_signal_fname, time_steps[-1], tp, 'rigid')
#     PIN_signal_fname = get_res_img_fname(PIN_signal_fname, time_steps[-1], tp, 'rigid')
#     seg_img_fname = get_res_img_fname(seg_img_fname, time_steps[-1], tp, 'rigid')

back_id = 1

# - To create a mask, edit a MaxIntensityProj with an image editor (GIMP) by adding 'black' (0 value):
# mask = image_dirname+'PIN1-GFP-CLV3-CH-MS-E1-LD-SAM4-MIP_PI-mask.png'
mask = ''

print "\n\n# - Reading PIN1 intensity image file {}...".format(PIN_signal_fname)
PIN_signal_im = imread(image_dirname + path_suffix + PIN_signal_fname)
# PIN_signal_im = isometric_resampling(PIN_signal_im)
# world.add(PIN_signal_im, 'PIN1 intensity image', colormap='viridis', voxelsize=PIN_signal_im.get_voxelsize())

print "\n\n# - Reading PI intensity image file {}...".format(PI_signal_fname)
PI_signal_im = imread(image_dirname + path_suffix + PI_signal_fname)
# PI_signal_im = isometric_resampling(PI_signal_im)
# world.add(PI_signal_im, 'PI intensity image', colormap='invert_grey', voxelsize=PI_signal_im.get_voxelsize())

print "\n\n# - Reading segmented image file {}...".format(seg_img_fname)
seg_im = imread(image_dirname + path_suffix + seg_img_fname)
# -- Detect small regions defined as background and remove them:
# mask_img = PI_signal_im == 0
# connected_mask_components, n_components = nd.label(mask_img)
# components_area = nd.sum(np.ones_like(connected_mask_components), connected_mask_components, index=np.arange(n_components)+1)
# largest_component = (np.arange(n_components)+1)[np.argmax(components_area)]
# mask_img = (connected_mask_components == largest_component).astype(np.uint16)
# seg_im[mask_img == 1] = back_id
# del mask_img, connected_mask_components, n_components, components_area
seg_im[seg_im == 0] = back_id
# world.add(seg_im, 'segmented image', colormap='glasbey', voxelsize=seg_im.get_voxelsize(), alphamap='constant')


################################################################################
# FUNCTIONS:
################################################################################
# - Functions to compute vectors scale, orientation and direction:
def PIN_polarity(PIN_ratio, *args):
    """
    """
    return 1 / PIN_ratio - 1


def PIN_polarity_area(PIN_ratio, area):
    """
    """
    return area * PIN_polarity(PIN_ratio)


def compute_vect_orientation(bary_ori, bary_dest):
    """
    Compute distance & direction as 'bary_ori -> bary_dest'.
    !! Not normalized, contains distance and direction !!
    """
    return [b-a for a,b in zip(bary_ori, bary_dest)]


def compute_vect_direction(bary_ori, bary_dest):
    """
    Compute direction as 'bary_ori -> bary_dest'.
    !! Normalized, contains only direction !!
    """
    dx, dy, dz = compute_vect_orientation(bary_ori, bary_dest)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    return [dx/norm, dy/norm, dz/norm]


# - Functions to create label-pair dictionary from pandas DataFrame:
def df2labelpair_dict(df, values_cname, lab1_cname="Unnamed: 0", lab2_cname="Unnamed: 1"):
    """
    Return a label-pair dictionary with df['values_cname'] as values, if this
    values is different from NaN.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the values
    values_cname : str
        dataframe column name containing the values to add to the dictionary
    lab1_cname : str, optional
        dataframe column name containing the first labels of the label-pairs
    lab2_cname : str, optional
        dataframe column name containing the second labels of the label-pairs

    Returns
    -------
    dictionary {(label_1, label_2): values}
    """
    lab1 = df[lab1_cname].to_dict()
    lab2 = df[lab2_cname].to_dict()
    values = df[values_cname].to_dict()
    return {(lab1[k], lab2[k]): v for k, v in values.items() if not np.isnan(v)}


###############################################################################
# -- PIN1/PI signal & PIN1 polarity quatification:
###############################################################################
print "\n\n# - Initialise signal quantification class:"
memb = MembraneQuantif(seg_im, [PIN_signal_im, PI_signal_im], ["PIN1", "PI"])

# - Cell-based information (barycenters):
cell_df_fname = image_dirname + splitext_zip(PI_signal_fname)[0] + '_cell_barycenters.csv'
# -- Get list of 'L1' labels:
labels = memb.labels_checker('L1')
# -- Compute the barycenters of each selected cells:
print "\n# - Compute the barycenters of each selected cells:"
bary = memb.center_of_mass(labels, real_bary, verbose=True)
print "Done."
bary_x = {k: v[0] for k, v in bary.items()}
bary_y = {k: v[1] for k, v in bary.items()}
bary_z = {k: v[2] for k, v in bary.items()}
# -- EXPORT data to csv:
cell_df = pd.DataFrame().from_dict({'bary_x': bary_x,
                                    'bary_y': bary_y,
                                    'bary_z': bary_z})
cell_df.to_csv(cell_df_fname)


#  - Wall-based information (barycenters):
wall_pd_fname = image_dirname + splitext_zip(PI_signal_fname)[0] + '_wall_PIN_PI_signal-D{}.csv'.format(membrane_dist)
# -- Create a list of anticlinal walls (ordered pairs of labels):
L1_anticlinal_walls = memb.L1_anticlinal_walls(min_area=walls_min_area, real_area=True)
# -- Compute the area of each walls (L1 anticlinal walls):
print "\n# - Compute the area of each walls (L1 anticlinal walls):"
wall_area = memb.wall_area_from_labelpairs(L1_anticlinal_walls, real=True)
print "Done."

# -- Compute the wall median of each selected walls (L1 anticlinal walls):
print "\n# - Compute the wall median of each selected walls (L1 anticlinal walls):"
wall_median = {}
for lab1, lab2 in L1_anticlinal_walls:
    wall_median[(lab1, lab2)] = memb.wall_median_from_labelpairs(lab1, lab2, real=False, min_area=walls_min_area, real_area=True)
print "Done."

# -- Compute the epidermis wall edge median of each selected walls (L1 anticlinal walls):
print "\n# - Compute the epidermis wall edge median of each selected walls (L1 anticlinal walls):"
ep_wall_median = {}
for lab1, lab2 in wall_median.keys():
    ep_wall_median[(lab1, lab2)] = memb.epidermal_wall_edges_median(lab1, lab2, real=False)
print "Done."

# -- Compute PIN1 and PI signal for each side of the walls:
print "\n# - Compute PIN1 and PI {} signal intensities:".format(quantif_method)
PIN_signal = memb.get_membrane_mean_signal("PIN1", L1_anticlinal_walls, membrane_dist, quantif_method)
PI_signal = memb.get_membrane_mean_signal("PI", L1_anticlinal_walls, membrane_dist, quantif_method)
print "Done."

# -- Compute PIN1 and PI signal ratios:
print "\n# - Compute PIN1 and PI {} signal ratios:".format(quantif_method)
PIN_ratio = memb.get_membrane_signal_ratio("PIN1", L1_anticlinal_walls, membrane_dist, quantif_method)
PI_ratio = memb.get_membrane_signal_ratio("PI", L1_anticlinal_walls, membrane_dist, quantif_method)
print "Done."

# -- Compute PIN1 and PI total signal (sum each side of the wall):
print "\n# - Compute PIN1 and PI total {} signal:".format(quantif_method)
PIN_signal = memb.get_membrane_signal_total("PIN1", L1_anticlinal_walls, membrane_dist, quantif_method)
PI_signal = memb.get_membrane_signal_total("PI", L1_anticlinal_walls, membrane_dist, quantif_method)
print "Done."

wall_median_x = {k: v[0] for k, v in wall_median.items()}
wall_median_y = {k: v[1] for k, v in wall_median.items()}
wall_median_z = {k: v[2] for k, v in wall_median.items()}
ep_wall_median_x = {k: v[0] for k, v in ep_wall_median.items()}
ep_wall_median_y = {k: v[1] for k, v in ep_wall_median.items()}
ep_wall_median_z = {k: v[2] for k, v in ep_wall_median.items()}
# -- EXPORT data to csv:
wall_df = pd.DataFrame().from_dict({'PIN_{}_signal'.format(quantif_method): PIN_signal,
                                    'PI_{}_signal'.format(quantif_method): PI_signal,
                                    'PIN_{}_ratio'.format(quantif_method): PIN_ratio,
                                    'PI_{}_ratio'.format(quantif_method): PI_ratio,
                                    'PI_total_{}'.format(quantif_method): PI_signal,
                                    'PIN_total_{}'.format(quantif_method): PIN_signal,
                                    'wall_median_x': wall_median_x,
                                    'wall_median_y': wall_median_y,
                                    'wall_median_z': wall_median_z,
                                    'epidermis_wall_edge_median_x': ep_wall_median_x,
                                    'epidermis_wall_edge_median_y': ep_wall_median_y,
                                    'epidermis_wall_edge_median_z': ep_wall_median_z,
                                    'wall_area': wall_area})
wall_df.to_csv(wall_pd_fname)
