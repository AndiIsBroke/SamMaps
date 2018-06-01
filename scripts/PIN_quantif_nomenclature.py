# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from timagetk.components.io import imread
from vplants.tissue_analysis.misc import rtuple
from vplants.tissue_analysis.signal_quantification import MembraneQuantif

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

from nomenclature import splitext_zip
from nomenclature import get_nomenclature_channel_fname
from nomenclature import get_nomenclature_segmentation_name
from nomenclature import get_res_img_fname
# Nomenclature file location:
nomenclature_file = SamMaps_dir + "nomenclature.csv"
# OUTPUT directory:
image_dirname = dirname + "nuclei_images/"

# - DEFAULT variables:
# Distance to membrane to consider during PIN1 levels quantifications:
DEF_MEMBRANE_DIST = 0.6
# Reference channel name used to compute tranformation matrix:
DEF_MEMB_CH = 'PI'
# Quantification method:
DEF_QUANTIF = "mean"
# Miminal area to consider a wall (contact between two labels) as valid:
DEF_MIN_AREA = 5.  # to avoid too small walls arising from segmentation errors
# Background value: (not handled by parser)
back_id = 1

# PARAMETERS:
# -----------
import argparse
parser = argparse.ArgumentParser(description='Performs quantification of membrane localized signal for L1 anticlinal walls.')
# positional arguments:
parser.add_argument('xp_id', type=str,
                    help="basename of the experiment (CZI file) without time-step and extension")
parser.add_argument('tp', type=int,
                    help="time step for which to quantify PIN1-GFP levels")
# optional arguments:
parser.add_argument('--membrane_dist', type=float, default=DEF_MEMBRANE_DIST,
                    help="distance to the membrane to consider when computing PIN1 intensity, '{}' by default".format(DEF_MEMBRANE_DIST))
parser.add_argument('--membrane_ch_name', type=str, default=DEF_MEMB_CH,
                    help="CZI channel name containing membrane intensity image, '{}' by default".format(DEF_MEMB_CH))
parser.add_argument('--quantif_method', type=str, default=DEF_QUANTIF,
                    help="quantification method to use to estimate PIN intensity value for a given wall, '{}' by default".format(DEF_QUANTIF))
parser.add_argument('--walls_min_area', type=float, default=DEF_MIN_AREA,
                    help="miminal REAL area to consider a wall (contact between two labels) as valid, '{}µm2' by default".format(DEF_MIN_AREA))
parser.add_argument('--force', action='store_true',
                    help="if given, force computation of values even if the *.CSV already exists, else skip it, 'False' by default")
parser.add_argument('--real_bary', action='store_true',
                    help="if given, export real-world barycenters to CSV, else use voxel unit, 'False' by default")
# - If you want to plot PIN signal image AND polarity field, you should use barycenters with voxel units:
args = parser.parse_args()

# - Variables definition from argument parsing:
base_fname = args.xp_id
tp = args.tp
# - Variables definition from optional arguments:
membrane_ch_name = args.membrane_ch_name
quantif_method = args.quantif_method
membrane_dist = args.membrane_dist
try:
    assert membrane_dist > 0.
except:
    raise ValueError("Negative distance provided!")
walls_min_area = args.walls_min_area
try:
    assert walls_min_area > 0.
except:
    raise ValueError("Negative minimal area!")
force =  args.force
if force:
    print "WARNING: any existing CSV files will be overwritten!"
else:
    print "Existing files will be kept."

# - Define variables AFTER argument parsing:
czi_fname = base_fname + "-T{}.czi".format(tp)

# Get unregistered image filename:
path_suffix, PI_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, membrane_ch_name)
path_suffix, PIN_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, 'PIN1')
path_suffix, seg_img_fname = get_nomenclature_segmentation_name(czi_fname, nomenclature_file, membrane_ch_name + "_raw")

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
seg_im[seg_im == 0] = back_id


###############################################################################
# -- PIN1/PI signal & PIN1 polarity quatification:
###############################################################################
print "\n\n# - Initialise signal quantification class:"
memb = MembraneQuantif(seg_im, [PIN_signal_im, PI_signal_im], ["PIN1", "PI"])

# - Cell-based information (barycenters):
# -- Get list of 'L1' labels:
labels = memb.labels_checker('L1')
# -- Compute the barycenters of each selected cells:
print "\n# - Compute the barycenters of each selected cells:"
bary = memb.center_of_mass(labels, real_bary, verbose=True)
print "Done."
# bary_x = {k: v[0] for k, v in bary.items()}
# bary_y = {k: v[1] for k, v in bary.items()}
# bary_z = {k: v[2] for k, v in bary.items()}


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


def PI_significance(PI_left, PI_right):
    """
    Compute a signal "significance" as:
     s = min(PI_left, PI_right) / max(PI_left, PI_right)
    """
    return min([PI_left, PI_right]) / max([PI_left, PI_right])


def vector2dim(dico):
    """
    Transfrom a dictionary k: length-3 vector into 3 dictionary for each
    dimension.
    """
    dico = {k: v for k, v in dico.items() if v is not None}
    x = {k: v[0] for k, v in dico.items()}
    y = {k: v[1] for k, v in dico.items()}
    z = {k: v[2] for k, v in dico.items()}
    return x, y, z


def symetrize_labelpair_dict(dico, oppose=False):
    """
    Returns a fully symetric dictionary with labelpairs as keys, ie. contains
    (label_1, label_2) & (label_2, label_1) as keys.
    """
    tmp = {}
    for (label_1, label_2), v in dico.items():
        tmp[(label_1, label_2)] = v
        if dico.has_key((label_2, label_1)):
            v2 = dico[(label_2, label_1)]
            if oppose and v2 != -v:
                print "dict[({}, {})] != -dict[({}, {})]".format(label_1, label_2, label_2, label_1)
                print "dict[({}, {})] = {}".format(label_1, label_2, v)
                print "dict[({}, {})] = {}".format(label_2, label_1, v2)
            elif not oppose and v2 != v:
                print "dict[({}, {})] != -dict[({}, {})]".format(label_1, label_2, label_2, label_1)
                print "dict[({}, {})] = {}".format(label_1, label_2, v)
                print "dict[({}, {})] = {}".format(label_2, label_1, v2)
            else:
                pass
        else:
            if oppose:
                tmp[(label_2, label_1)] = -v
            else:
                tmp[(label_2, label_1)] = v
    return tmp


def invert_labelpair_dict(dico):
    """
    Return a dict with inverted labelpairs.
    """
    return {(label_2, label_1): v for (label_1, label_2), v in dico.items()}


# -- Create a list of L1 anticlinal walls (ordered pairs of labels):
print "\n# - Compute the labelpair list of L1 anticlinal walls:"
L1_anticlinal_walls = memb.list_epidermis_anticlinal_walls(min_area=walls_min_area, real_area=True)
n_lp = len(L1_anticlinal_walls)
print "Found {} unique (sorted) labelpairs".format(n_lp)

# -- Compute the area of each walls (L1 anticlinal walls):
print "\n# - Compute the area of each walls (L1 anticlinal walls):"
wall_area = memb.wall_area_from_labelpairs(L1_anticlinal_walls, real=True)
print "Done."
n = len(set([stuple(k) for k, v in wall_area.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

# # -- Compute the wall median of each selected walls (L1 anticlinal walls):
# print "\n# - Compute the wall median of each selected walls (L1 anticlinal walls):"
# wall_median = {}
# for lab1, lab2 in L1_anticlinal_walls:
#     wall_median[(lab1, lab2)] = memb.wall_median_from_labelpairs(lab1, lab2, real=False, min_area=walls_min_area, real_area=True)
# print "Done."

# -- Compute the epidermis wall edge median of each selected walls (L1 anticlinal walls):
print "\n# - Compute the epidermis wall edge median of each selected walls (L1 anticlinal walls):"
ep_wall_median = memb.epidermal_wall_edges_median(L1_anticlinal_walls, real=False, verbose=True)
n = len(set([stuple(k) for k, v in ep_wall_median.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

# -- Compute PIN1 and PI signal for each side of the walls:
print "\n# - Compute PIN1 {} signal intensities:".format(quantif_method)
PIN_left_signal = memb.get_membrane_mean_signal("PIN1", L1_anticlinal_walls, membrane_dist, quantif_method)
n = len(set([stuple(k) for k, v in PIN_left_signal.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

print "\n# - Compute PI {} signal intensities:".format(quantif_method)
PI_left_signal = memb.get_membrane_mean_signal("PI", L1_anticlinal_walls, membrane_dist, quantif_method)
n = len(set([stuple(k) for k, v in PI_left_signal.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

PI_left_signal = symetrize_labelpair_dict(PI_left_signal, oppose=False)
PIN_left_signal = symetrize_labelpair_dict(PIN_left_signal, oppose=False)
PI_right_signal = invert_labelpair_dict(PI_left_signal)
PIN_right_signal = invert_labelpair_dict(PIN_left_signal)

# -- Compute PIN1 and PI signal ratios:
print "\n# - Compute PIN1 {} signal ratios:".format(quantif_method)
PIN_ratio = memb.get_membrane_signal_ratio("PIN1", L1_anticlinal_walls, membrane_dist, quantif_method)
n = len(set([stuple(k) for k, v in PIN_ratio.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

print "\n# - Compute PI {} signal ratios:".format(quantif_method)
PI_ratio = memb.get_membrane_signal_ratio("PI", L1_anticlinal_walls, membrane_dist, quantif_method)
n = len(set([stuple(k) for k, v in PI_ratio.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

PIN_ratio = symetrize_labelpair_dict(PIN_ratio, oppose=True)
PI_ratio = symetrize_labelpair_dict(PI_ratio, oppose=True)


# -- Compute PIN1 and PI total signal (sum each side of the wall):
print "\n# - Compute PIN1 and PI total {} signal:".format(quantif_method)
PIN_signal = memb.get_membrane_signal_total("PIN1", L1_anticlinal_walls, membrane_dist, quantif_method)
PIN_signal = symetrize_labelpair_dict(PIN_signal, oppose=False)
PI_signal = memb.get_membrane_signal_total("PI", L1_anticlinal_walls, membrane_dist, quantif_method)
PI_signal = symetrize_labelpair_dict(PI_signal, oppose=False)
print "Done."

wall_normal_dir = {(lab1, lab2): compute_vect_direction(bary[lab1], bary[lab2]) for (lab1, lab2) in L1_anticlinal_walls}
dir_x, dir_y, dir_z = vector2dim(wall_normal_dir)
ori_x, ori_y, ori_z = vector2dim(ep_wall_median)

PIN1_orientation = {lp: 1 if v>0 else -1 for lp, v in PIN_ratio.items()}

significance = {lp: PI_significance(PI_left_signal[lp], PI_right_signal[lp]) for lp in PI_left_signal.keys()}

wall_area = symetrize_labelpair_dict(wall_area, oppose=False)

# -- Create a Pandas DataFrame:
wall_df = pd.DataFrame().from_dict({'PI_signal': PI_signal,
                                    'PIN1_signal': PIN_signal,
                                    'PI_left': PI_left_signal,
                                    'PI_right': PI_right_signal,
                                    'PIN1_left': PIN_left_signal,
                                    'PIN1_right': PIN_right_signal,
                                    'PIN1_orientation': PIN1_orientation,
                                    'significance': significance,
                                    'ori_x': ori_x, 'ori_y': ori_y, 'ori_z': ori_z,
                                    'dir_x': dir_x, 'dir_y': dir_y, 'dir_z': dir_z,
                                    'wall_area': wall_area})

# - CSV filename change with 'membrane_dist':
wall_pd_fname = image_dirname + path_suffix + splitext_zip(PI_signal_fname)[0] + '_wall_PIN_PI_{}_signal-D{}.csv'.format(quantif_method, membrane_dist)
# - Export to CSV:
wall_df.to_csv(wall_pd_fname, index_label=['left_label', 'right_label'])
