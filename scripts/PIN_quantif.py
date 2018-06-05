# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon - INRIA
#
#       File author(s): Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
################################################################################

import numpy as np
import pandas as pd

from os.path import exists
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
# -- Channel name for the 'membrane labelling' image:
DEF_MEMB_CH = 'PI'
# -- Channel name for the 'membrane-targetted signal' image:
DEF_SIG_CH = 'PIN1'


# PARAMETERS:
# -----------
import argparse
parser = argparse.ArgumentParser(description='Performs quantification of membrane localized signal.')
# positional arguments:
parser.add_argument('membrane_im', type=str,
                    help="file containing the 'membrane labelling' channel.")
parser.add_argument('signal_im', type=str,
                    help="file containing the 'membrane-targetted signal of interest'.")
parser.add_argument('segmented_im', type=str,
                    help="segmented image corresponding to the 'membrane labelling' channel")

POSS_LABELS = ['all', 'L1', 'L2']
POSS_WALLS = ['all', 'L1_anticlinal', 'L1/L2']
# optional arguments:
parser.add_argument('--labels', type=str, default='all',
                    help="restrict to this list of labels, availables are: {}".format(POSS_LABELS))
parser.add_argument('--walls', type=str, default='all',
                    help="restrict to this list of walls, availables are: {}".format(POSS_LABELS))
parser.add_argument('--back_id', type=int, default=None,
                    help="background id to be found in the segmented image")
parser.add_argument('--membrane_dist', type=float, default=DEF_MEMBRANE_DIST,
                    help="distance to the membrane to consider when computing 'membrane-targetted signal' intensity, '{}' by default".format(DEF_MEMBRANE_DIST))
parser.add_argument('--quantif_method', type=str, default=DEF_QUANTIF,
                    help="quantification method to use to estimate 'membrane-targetted signal' intensity value for a given wall, '{}' by default".format(DEF_QUANTIF))
parser.add_argument('--membrane_ch_name', type=str, default=DEF_MEMB_CH,
                    help="channel name for the 'membrane labelling' channel, '{}' by default".format(DEF_MEMB_CH))
parser.add_argument('--signal_ch_name', type=str, default=DEF_SIG_CH,
                    help="channel name for the 'membrane-targetted signal of interest', '{}' by default".format(DEF_SIG_CH))
parser.add_argument('--walls_min_area', type=float, default=DEF_MIN_AREA,
                    help="miminal REAL area to consider a wall (contact between two labels) as valid, '{}µm2' by default".format(DEF_MIN_AREA))
parser.add_argument('--force', action='store_true',
                    help="if given, force computation of values even if the *.CSV already exists, else skip it, 'False' by default")
parser.add_argument('--real_bary', action='store_true',
                    help="if given, export real-world barycenters to CSV, else use voxel unit, 'False' by default")
parser.add_argument('--bounding_box', type=int, nargs='+', default=None,
                    help="if given, used to crop the image around these voxel coordinates, use '0' for the beginning and '-1' for the end")


args = parser.parse_args()

# - Variables definition from mandatory arguments parsing:
# -- Membrane labelling signal image:
memb_im_fname = args.membrane_im
print "\n\n# - Reading membrane labelling signal image file {}...".format(memb_im_fname)
memb_im = read_image(memb_im_fname)
print "Done."
# -- Membrane-targetted signal image:
sig_im_fname = args.signal_im
print "\n\n# - Reading membrane-targetted signal image file {}...".format(sig_im_fname)
sig_im = read_image(sig_im_fname)
print "Done."
# -- Segmented images:
seg_im_fname = args.segmented_im
print "\n\n# - Reading segmented image file {}...".format(seg_im_fname)
seg_im = read_image(seg_im_fname)
print "Done."

# - Variables definition from optional arguments parsing:
# -- Labels:
labels_str = args.labels
try:
    assert labels_str in POSS_LABELS
except AssertionError:
    raise ValueError("Unknown list of labels '{}', availables are: {}".format(labels_str, POSS_LABELS))
# -- Walls:
walls_str = args.walls
try:
    assert walls_str in POSS_WALLS
except AssertionError:
    raise ValueError("Unknown list of labels '{}', availables are: {}".format(walls_str, POSS_LABELS))
# -- Background label:
back_id = args.back_id
if back_id is not None:
    try:
        assert back_id in seg_im
    except AssertionError:
        raise ValueError("Background id not found in the segmented image!")
else:
    try:
        assert walls_str != 'L1_anticlinal' and walls_str != 'L1/L2'
    except AssertionError:
        raise ValueError("No background value defined, cannot detect L1!")
    print "No background id defined!"
# -- Distance to the membrane:
membrane_dist = args.membrane_dist
try:
    assert membrane_dist > 0.
except:
    raise ValueError("Negative distance provided!")
# -- Quantification method:
quantif_method = args.quantif_method
try:
    assert quantif_method in POSS_QUANTIF_METHODS
except AssertionError:
    raise ValueError("Unknown quantification method '{}', availables are {}".format(quantif_method, POSS_QUANTIF_METHODS))
# -- Channel names:
membrane_ch_name = args.membrane_ch_name
signal_ch_name = args.signal_ch_name
# -- Real/voxel units:
real_bary = args.real_bary
# -- Minimal wall area to cosider:
walls_min_area = args.walls_min_area
try:
    assert walls_min_area > 0.
except:
    raise ValueError("Negative minimal area!")

# - CSV filename change with 'signal_ch_name', 'membrane_ch_name', 'quantif_method' & 'membrane_dist':
out_fname = splitext_zip(memb_im_fname)[0] + '_wall_{}_{}_{}_signal-D{}'.format(signal_ch_name, membrane_ch_name, quantif_method, membrane_dist)

# -- Image cropping:
bounding_box =  args.bounding_box
if bounding_box:
    print "Got a cropping bounding box: {}".format(bounding_box)
    shape = memb_im.get_shape()
    axis = ['x', 'y', 'z']
    axis = axis[:memb_im.get_dim()]
    # - Define lower and upper bounds:
    lower_bounds = bounding_box[::2]
    upper_bounds = bounding_box[1::2]
    # -- Change '-1' into max shape value for given axis:
    for n, upper_bound in enumerate(upper_bounds):
        if upper_bound == -1:
            upper_bounds[n] = shape[n] - 1
    # - Check lower bound values are positive and strictly inferior to the max shape of the image:
    for n, lower_bound in enumerate(lower_bounds):
        try:
            assert lower_bound >= 0
        except AssertionError:
            raise ValueError("Lower bound '{}' ({}) is negative, please check!".format(axis[n], lower_bound))
        try:
            assert lower_bound < shape[n] - 1
        except AssertionError:
            raise ValueError("Lower bound '{}' ({}) is not strictly inferior to the max shape ({}), please check!".format(axis[n], lower_bound, shape[n] - 1))
    # - Check upper bound values are strictly positive and inferior to the max shape of the image:
    for n, upper_bound in enumerate(upper_bounds):
        try:
            assert upper_bound > 0
        except AssertionError:
            raise ValueError("Upper bound '{}' ({}) is not superior to zero, please check!".format(axis[n], upper_bound))
        try:
            assert upper_bound <= shape[n] - 1
        except AssertionError:
            raise ValueError("Upper bound '{}' ({}) is greater than the max shape ({}), please check!".format(axis[n], upper_bound, shape[n] - 1))
    # - Add the cropping values to the filename:
    for n, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        if lower != 0  or upper != shape[n] - 1:
            out_fname += '-{}{}_{}'.format(axis[n], lower, upper)
else:
    pass

out_fname += '.csv'
# -- Force overwritting of existing files:
force =  args.force
if exists(out_fname):
    print "\nFound existing CSV file: {}".format(out_fname)
    if force:
        print "WARNING: existing CSV file will be overwritten!"
        print out_fname
    else:
        print "Aborted!"
        print "(to enable overwriting add '--force')."
        sys.exit(0)


###############################################################################
# -- EXTRA functions:
###############################################################################
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


def symetrize_labelpair_dict(dico, oppose=False, verbose=False):
    """
    Returns a fully symetric dictionary with labelpairs as keys, ie. contains
    (label_1, label_2) & (label_2, label_1) as keys.
    """
    tmp = {}
    for (label_1, label_2), v in dico.items():
        tmp[(label_1, label_2)] = v
        if dico.has_key((label_2, label_1)):
            v2 = dico[(label_2, label_1)]
            if verbose:
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


###############################################################################
# -- PIN1/PI signal & PIN1 polarity quatification:
###############################################################################
print "\n\n# - Initialise signal quantification class:"
memb = MembraneQuantif(seg_im, [sig_im, memb_im], [signal_ch_name, membrane_ch_name], background=back_id, slice=bounding_box)

# - Cell-based information (barycenters):
# -- Get list of labels:
labels = memb.labels_checker(labels_str)
# -- Compute the barycenters of each selected cells:
print "\n# - Compute the barycenters of each selected cells:"
bary = memb.center_of_mass(labels, real=real_bary, verbose=True)
print "Done."
# bary_x = {k: v[0] for k, v in bary.items()}
# bary_y = {k: v[1] for k, v in bary.items()}
# bary_z = {k: v[2] for k, v in bary.items()}

# -- Create a list of walls (ordered pairs of labels):
print "\n# - Compute the labelpair list of {} walls:".format(walls_str)
if walls_str == 'all':
    wall_labelpairs = memb.list_all_walls(min_area=walls_min_area, real_area=True)
    if back_id is not None:
        epidermal_walls_labelpairs = memb.list_epidermal_walls(min_area=walls_min_area, real_area=True)
        wall_labelpairs = [lp for lp in wall_labelpairs if lp not in epidermal_walls_labelpairs]
elif walls_str == 'L1_anticlinal':
    wall_labelpairs = memb.list_epidermis_anticlinal_walls(min_area=walls_min_area, real_area=True)
elif walls_str == 'L1/L2':
    wall_labelpairs = memb.list_l1_l2_walls(min_area=walls_min_area, real_area=True)
else:
    pass

n_lp = len(wall_labelpairs)
print "Found {} unique (ie. 'sorted') wall labelpairs!".format(n_lp)

# -- Compute the area of each walls:
print "\n# - Compute the area of each walls:"
wall_area = memb.wall_area_from_labelpairs(wall_labelpairs, real=True)
print "Done."
n = len(set([stuple(k) for k, v in wall_area.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

# -- Compute the (epidermis) wall (edge) median of each selected walls:
if back_id is not None and walls_str == 'L1_anticlinal':
    print "\n# - Compute the epidermis wall edge median for selected walls:"
    wall_median = memb.epidermal_wall_edges_median(wall_labelpairs, real=False, verbose=True)
else:
    print "\n# - Compute the wall median for selected walls:"
    wall_median = memb.wall_medians(wall_labelpairs, real=False, min_area=None, verbose=True)
n = len(set([stuple(k) for k, v in wall_median.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

# -- Compute PIN1 and PI signal for each side of the walls:
print "\n# - Compute {} {} signal intensities:".format(signal_ch_name, quantif_method)
PIN_left_signal = memb.get_membrane_signal(signal_ch_name, wall_labelpairs, membrane_dist, method=quantif_method, min_area=None)
n = len(set([stuple(k) for k, v in PIN_left_signal.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

print "\n# - Compute {} {} signal intensities:".format(membrane_ch_name, quantif_method)
PI_left_signal = memb.get_membrane_signal(membrane_ch_name, wall_labelpairs, membrane_dist, method=quantif_method, min_area=None)
n = len(set([stuple(k) for k, v in PI_left_signal.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

# PI_left_signal = symetrize_labelpair_dict(PI_left_signal, oppose=False)
# PIN_left_signal = symetrize_labelpair_dict(PIN_left_signal, oppose=False)
PI_right_signal = invert_labelpair_dict(PI_left_signal)
PIN_right_signal = invert_labelpair_dict(PIN_left_signal)

# -- Compute PIN1 signal ratios:
print "\n# - Compute {} {} signal ratios:".format(signal_ch_name, quantif_method)
PIN_ratio = memb.get_membrane_signal_ratio(signal_ch_name, wall_labelpairs, membrane_dist, method=quantif_method, min_area=None)
n = len(set([stuple(k) for k, v in PIN_ratio.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

# -- Compute PI signal ratios:
print "\n# - Compute {} {} signal ratios:".format(membrane_ch_name, quantif_method)
PI_ratio = memb.get_membrane_signal_ratio(membrane_ch_name, wall_labelpairs, membrane_dist, method=quantif_method, min_area=None)
n = len(set([stuple(k) for k, v in PI_ratio.items() if v is not None]))
print "Success rate: {}%".format(round(n/float(n_lp), 3)*100)

PIN_ratio = symetrize_labelpair_dict(PIN_ratio, oppose=True, verbose=True)
PI_ratio = symetrize_labelpair_dict(PI_ratio, oppose=True, verbose=True)


# -- Compute PIN1 and PI total signal (sum each side of the wall):
print "\n# - Compute {} and {} total {} signal:".format(signal_ch_name, membrane_ch_name, quantif_method)
PIN_signal = memb.get_membrane_signal_total(signal_ch_name, wall_labelpairs, membrane_dist, method=quantif_method, min_area=None)
PIN_signal = symetrize_labelpair_dict(PIN_signal, oppose=False, verbose=True)
PI_signal = memb.get_membrane_signal_total(membrane_ch_name, wall_labelpairs, membrane_dist, method=quantif_method, min_area=None)
PI_signal = symetrize_labelpair_dict(PI_signal, oppose=False, verbose=True)
print "Done."

wall_normal_dir = {(lab1, lab2): compute_vect_direction(bary[lab1], bary[lab2]) for (lab1, lab2) in wall_labelpairs}
dir_x, dir_y, dir_z = vector2dim(wall_normal_dir)
ori_x, ori_y, ori_z = vector2dim(wall_median)

PIN1_orientation = {lp: 1 if v>0 else -1 for lp, v in PIN_ratio.items()}

significance = {lp: PI_significance(PI_left_signal[lp], PI_right_signal[lp]) for lp in PI_left_signal.keys()}

wall_area = symetrize_labelpair_dict(wall_area, oppose=False, verbose=True)

# -- Create a Pandas DataFrame:
wall_df = pd.DataFrame().from_dict({membrane_ch_name+'_signal': PI_signal,
                                    signal_ch_name+'_signal': PIN_signal,
                                    membrane_ch_name+'_signal_left': PI_left_signal,
                                    membrane_ch_name+'_signal_right': PI_right_signal,
                                    signal_ch_name+'_signal_left': PIN_left_signal,
                                    signal_ch_name+'_signal_right': PIN_right_signal,
                                    signal_ch_name+'_orientation': PIN1_orientation,
                                    'significance': significance,
                                    'wall_center_x': ori_x, 'wall_center_y': ori_y, 'wall_center_z': ori_z,
                                    'wall_normal_x': dir_x, 'wall_normal_y': dir_y, 'wall_normal_z': dir_z,
                                    'wall_area': wall_area})

# - Export to CSV:
print "\nSaving CVS file: '{}'".format(out_fname)
wall_df.to_csv(out_fname, index_label=['left_label', 'right_label'])
