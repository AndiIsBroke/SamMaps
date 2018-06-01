# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from vplants.tissue_analysis.misc import rtuple
from vplants.tissue_analysis.misc import stuple
from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
from vplants.tissue_analysis.signal_quantification import MembraneQuantif, POSS_QUANTIF_METHODS

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/lib/')

from nomenclature import splitext_zip
from segmentation_pipeline import read_image


# - DEFAULT variables:
# -- Distance to membrane to consider during PIN1 levels quantifications:
DEF_MEMBRANE_DIST = 0.6
# -- Quantification method:
DEF_QUANTIF = "mean"
# -- Miminal area to consider a wall (contact between two labels) as valid:
DEF_MIN_AREA = 5.  # to avoid too small walls arising from segmentation errors
# -- Reference channel name used to compute tranformation matrix:
DEF_MEMB_CH = 'PI'
# -- Reference channel name used to compute tranformation matrix:
DEF_SIG_CH = 'PIN1'


# PARAMETERS:
# -----------
import argparse
parser = argparse.ArgumentParser(description='Performs quantification of membrane localized signal for L1 anticlinal walls.')
# positional arguments:
parser.add_argument('czi', type=str,
                    help="CZI file with the 'membrane labelling' and 'membrane-targetted signal of interest' channels.")
parser.add_argument('segmented_im', type=str,
                    help="segmented image corresponding to the 'membrane labelling' channel")
parser.add_argument('channel_names', type=str, nargs='+',
                    help="list of channel names found in the given CZI")

# optional arguments:
parser.add_argument('--back_id', type=int, default=None,
                    help="background id to be found in the segmented image")
parser.add_argument('--membrane_dist', type=float, default=DEF_MEMBRANE_DIST,
                    help="distance to the membrane to consider when computing PIN1 intensity, '{}' by default".format(DEF_MEMBRANE_DIST))
parser.add_argument('--membrane_ch_name', type=str, default=DEF_MEMB_CH,
                    help="CZI channel name containing the 'membrane labelling' channel, '{}' by default".format(DEF_MEMB_CH))
parser.add_argument('--signal_ch_name', type=str, default=DEF_SIG_CH,
                    help="CZI channel name containing the 'membrane-targetted signal of interest', '{}' by default".format(DEF_SIG_CH))
parser.add_argument('--quantif_method', type=str, default=DEF_QUANTIF,
                    help="quantification method to use to estimate PIN intensity value for a given wall, '{}' by default".format(DEF_QUANTIF))
parser.add_argument('--walls_min_area', type=float, default=DEF_MIN_AREA,
                    help="miminal REAL area to consider a wall (contact between two labels) as valid, '{}µm2' by default".format(DEF_MIN_AREA))
parser.add_argument('--force', action='store_true',
                    help="if given, force computation of values even if the *.CSV already exists, else skip it, 'False' by default")
parser.add_argument('--real_bary', action='store_true',
                    help="if given, export real-world barycenters to CSV, else use voxel unit, 'False' by default")
DEF_MIN_VOL = 8000
parser.add_argument('--cell_min_vol', type=float, default=None,
                    help="miminal voxel volume to consider a cell, 'None' by default")
DEF_MAX_VOL = 50000
parser.add_argument('--cell_max_vol', type=float, default=None,
                    help="maximal voxel volume to consider a cell, 'None' by default")


# - If you want to plot PIN signal image AND polarity field, you should use barycenters with voxel units:
args = parser.parse_args()

# - Variables definition from argument parsing:
czi_fname = args.czi
seg_img_fname = args.segmented_im
channel_names = args.channel_names

# - Variables definition from optional arguments:
# -- Background label:
back_id = args.back_id
membrane_dist = args.membrane_dist
membrane_ch_name = args.membrane_ch_name
signal_ch_name = args.signal_ch_name
quantif_method = args.quantif_method
real_bary = args.real_bary
try:
    assert membrane_dist > 0.
except:
    raise ValueError("Negative distance provided!")
walls_min_area = args.walls_min_area
try:
    assert walls_min_area > 0.
except:
    raise ValueError("Negative minimal area!")
# -- Force overwritting of existing files:
force =  args.force
if force:
    print "WARNING: any existing CSV files will be overwritten!"
else:
    print "Existing files will be kept."

print "\n\n# - Reading CZI intensity image file {}...".format(czi_fname)
czi_im = read_image(czi_fname, channel_names)
PIN_signal_im = czi_im[signal_ch_name]
PI_signal_im = czi_im[membrane_ch_name]
print "Done."

print "\n\n# - Reading segmented image file {}...".format(seg_img_fname)
seg_im = read_image(seg_img_fname)
print "Done."

min_cell_volume = args.cell_min_vol
if min_cell_volume is None:
    min_cell_volume = 0
max_cell_volume = args.cell_max_vol
if max_cell_volume is None:
    max_cell_volume = 0
if min_cell_volume > 0 or max_cell_volume > 0:
    print "\n\n# - Filtering for cells with {} < volume < {}".format(min_cell_volume, max_cell_volume)
    spia = SpatialImageAnalysis(seg_im, background=None)
    print " -- computing volumes dictionary..."
    volumes = spia.volume(real=False)
    cells2rm = [k for k,v in volumes.items() if v<=min_cell_volume]
    cells2rm.extend([k for k,v in volumes.items() if v>=max_cell_volume])
    print " -- remove {} filtered labels from image...".format(len(cells2rm))
    seg_im = spia.get_image_without_labels(cells2rm, no_label_value=back_id)
    print "Done."

print np.unique(seg_im.get_array())
print back_id in seg_im
###############################################################################
# -- PIN1/PI signal & PIN1 polarity quatification:
###############################################################################
print "\n\n# - Initialise signal quantification class:"
memb = MembraneQuantif(seg_im, [PIN_signal_im, PI_signal_im], [signal_ch_name, membrane_ch_name], background=back_id)

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
print "\n# - Compute {} {} signal intensities:".format(sig_ch_name, quantif_method)
PIN_left_signal = memb.get_membrane_mean_signal(sig_ch_name, L1_anticlinal_walls, membrane_dist, quantif_method)
n = len(set([stuple(k) for k, v in PIN_left_signal.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

print "\n# - Compute {} {} signal intensities:".format(membrane_ch_name, quantif_method)
PI_left_signal = memb.get_membrane_mean_signal(membrane_ch_name, L1_anticlinal_walls, membrane_dist, quantif_method)
n = len(set([stuple(k) for k, v in PI_left_signal.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

PI_left_signal = symetrize_labelpair_dict(PI_left_signal, oppose=False)
PIN_left_signal = symetrize_labelpair_dict(PIN_left_signal, oppose=False)
PI_right_signal = invert_labelpair_dict(PI_left_signal)
PIN_right_signal = invert_labelpair_dict(PIN_left_signal)

# -- Compute PIN1 and PI signal ratios:
print "\n# - Compute {} {} signal ratios:".format(sig_ch_name, quantif_method)
PIN_ratio = memb.get_membrane_signal_ratio(sig_ch_name, L1_anticlinal_walls, membrane_dist, quantif_method)
n = len(set([stuple(k) for k, v in PIN_ratio.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

print "\n# - Compute {} {} signal ratios:".format(membrane_ch_name, quantif_method)
PI_ratio = memb.get_membrane_signal_ratio(membrane_ch_name, L1_anticlinal_walls, membrane_dist, quantif_method)
n = len(set([stuple(k) for k, v in PI_ratio.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

PIN_ratio = symetrize_labelpair_dict(PIN_ratio, oppose=True)
PI_ratio = symetrize_labelpair_dict(PI_ratio, oppose=True)


# -- Compute PIN1 and PI total signal (sum each side of the wall):
print "\n# - Compute {} and {} total {} signal:".format(sig_ch_name, membrane_ch_name, quantif_method)
PIN_signal = memb.get_membrane_signal_total(sig_ch_name, L1_anticlinal_walls, membrane_dist, quantif_method)
PIN_signal = symetrize_labelpair_dict(PIN_signal, oppose=False)
PI_signal = memb.get_membrane_signal_total(membrane_ch_name, L1_anticlinal_walls, membrane_dist, quantif_method)
PI_signal = symetrize_labelpair_dict(PI_signal, oppose=False)
print "Done."

wall_normal_dir = {(lab1, lab2): compute_vect_direction(bary[lab1], bary[lab2]) for (lab1, lab2) in L1_anticlinal_walls}
dir_x, dir_y, dir_z = vector2dim(wall_normal_dir)
ori_x, ori_y, ori_z = vector2dim(ep_wall_median)

PIN1_orientation = {lp: 1 if v>0 else -1 for lp, v in PIN_ratio.items()}

significance = {lp: PI_significance(PI_left_signal[lp], PI_right_signal[lp]) for lp in PI_left_signal.keys()}

wall_area = symetrize_labelpair_dict(wall_area, oppose=False)

# -- Create a Pandas DataFrame:
wall_df = pd.DataFrame().from_dict({membrane_ch_name+'_signal': PI_signal,
                                    sig_ch_name+'_signal': PIN_signal,
                                    membrane_ch_name+'_left': PI_left_signal,
                                    membrane_ch_name+'_right': PI_right_signal,
                                    sig_ch_name+'_left': PIN_left_signal,
                                    sig_ch_name+'_right': PIN_right_signal,
                                    sig_ch_name+'_orientation': PIN1_orientation,
                                    'significance': significance,
                                    'wall_center_x': ori_x, 'wall_center_y': ori_y, 'wall_center_z': ori_z,
                                    'wall_normal_x': dir_x, 'wall_normal_y': dir_y, 'wall_normal_z': dir_z,
                                    'wall_area': wall_area})

# - CSV filename change with 'membrane_dist':
wall_pd_fname = image_dirname + path_suffix + splitext_zip(PI_signal_fname)[0] + '_wall_{}_{}_{}_signal-D{}.csv'.format(sig_ch_name, membrane_ch_name, quantif_method, membrane_dist)
# - Export to CSV:
wall_df.to_csv(wall_pd_fname, index_label=['left_label', 'right_label'])
