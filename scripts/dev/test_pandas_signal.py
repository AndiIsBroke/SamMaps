# -*- coding: utf-8 -*-
from os.path import split
from os.path import exists
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec

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
sys.path.append(SamMaps_dir+'/scripts/lib/')

from nomenclature import splitext_zip
from nomenclature import get_nomenclature_channel_fname
from nomenclature import get_nomenclature_segmentation_name
from nomenclature import get_res_img_fname
from nomenclature import get_res_trsf_fname

# XP = sys.argv[1]
# SAM = sys.argv[2]
XP = 'E35'
SAM=4

image_dirname = dirname + "nuclei_images/"
nomenclature_file = SamMaps_dir + "nomenclature.csv"

# -4- Define CZI channel names, the microscope orientation, nuclei and membrane channel names and extra channels that should also be registered:
time_steps = [0, 5, 10, 14]
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
membrane_ch_name = 'PI'
back_id = 1

membrane_dist = 0.6
base_fname = "qDII-CLV3-PIN1-PI-{}-LD-SAM{}".format(XP, SAM)

ratio_method = "mean"
PIN_signal_method = "mean"
PI_signal_method = "mean"
walls_min_area = 5.  # to avoid too small walls arising from segmentation errors

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


# - Filter displayed PIN1 ratios by:
#    - a minimal wall area (already done when computing list of L1 anticlinal walls)
#    - a minimum PI ratio to prevent badbly placed wall to bias the PIN ratio
min_PI_ratio = 0.95
#    - a maximum (invert) PIN ratio value to prevent oultiers to change the scale of vectors
max_signal_ratio = 1/0.7
#    - a maximum wall area to avoid creating ouliers with big values!
walls_max_area = 250.
# TODO ???

###############################################################################
# -- PIN1/PI signal quantification:
###############################################################################
#  - Wall-based information:
for tp in time_steps:
    czi_fname = base_fname + "-T{}.czi".format(tp)
    # Get unregistered image filename:
    path_suffix, PI_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, membrane_ch_name)
    if tp != time_steps[-1]:
        path_suffix += 'rigid_registrations/'
        # Get RIDIG registered on last time-point filename:
        PI_signal_fname = get_res_img_fname(PI_signal_fname, time_steps[-1], tp, 'rigid')
    #  - Wall-based information (barycenters):
    wall_pd_fname = image_dirname + splitext_zip(PI_signal_fname)[0] + '_wall_PIN_PI_signal-D{}.csv'.format(membrane_dist)
    if not exists(wall_pd_fname):
        print "Could not find PIN1 quantification for {} SAM{} t_{}h at D={}, computing it!".format(XP, SAM, tp, membrane_dist)
        %run PIN_quantif.py $XP $SAM $tp $membrane_dist



###############################################################################
# -- PI intensity ratio (minPI/maxPI) distributions histogram:
###############################################################################
PI_ratio_list = []
label_list = []
for tp in time_steps:
    czi_fname = base_fname + "-T{}.czi".format(tp)
    # Get unregistered image filename:
    path_suffix, PI_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, membrane_ch_name)
    if tp != time_steps[-1]:
        path_suffix += 'rigid_registrations/'
        # Get RIDIG registered on last time-point filename:
        PI_signal_fname = get_res_img_fname(PI_signal_fname, time_steps[-1], tp, 'rigid')

    # - Wall-based information:
    wall_pd_fname = image_dirname + splitext_zip(PI_signal_fname)[0] + '_wall_PIN_PI_signal-D{}.csv'.format(membrane_dist)
    wall_df = pd.read_csv(wall_pd_fname)
    PI_ratio = df2labelpair_dict(wall_df, 'PI_{}_ratio'.format(ratio_method))
    PI_ratio_list.append(PI_ratio.values())
    label_list.append('t{}h (n={})'.format(tp, len(PI_ratio)))


mini = 0
maxi = 1
fig, subaxes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
subaxes[0].set_title("PI {} intensity ratios for {} SAM{} (D={}).".format(ratio_method.upper(), XP, SAM, membrane_dist))

for n, tp in enumerate(time_steps):
    subaxes[0].hist(PI_ratio_list[n], label=label_list[n], bins=(maxi-mini) * 100, range=(mini, maxi), alpha=0.3)
subaxes[0].set_xlabel('PI {} intensity ratios (minPI/maxPI)'.format(ratio_method))
subaxes[0].set_ylabel('Frequency')
subaxes[0].set_xlim(mini, maxi)
subaxes[0].legend(loc=0)

subaxes[1].boxplot(PI_ratio_list, vert=False, labels=label_list)
subaxes[1].set_xlabel('PI {} intensity ratios (minPI/maxPI)'.format(ratio_method))
subaxes[1].set_ylabel('Time step (h)')
subaxes[1].xaxis.grid(True)
subaxes[1].set_xlim(mini, maxi)
subaxes[1].legend()

plt.tight_layout()
plt.savefig(image_dirname + "{}-SAM{}-{}_PI_ratio-D{}-over_time.pdf".format(XP, SAM, ratio_method, membrane_dist))

plt.show()




###############################################################################
# -- PIN1/PI signal distributions:
###############################################################################
# - Boxplots of PI & PIN signal over time:
label_list = []
PIN_total_mean_list = []
PI_total_mean_list = []
mini = 0
maxi = 0
for tp in time_steps:
    czi_fname = base_fname + "-T{}.czi".format(tp)
    # Get unregistered image filename:
    path_suffix, PI_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, membrane_ch_name)
    if tp != time_steps[-1]:
        path_suffix += 'rigid_registrations/'
        # Get RIDIG registered on last time-point filename:
        PI_signal_fname = get_res_img_fname(PI_signal_fname, time_steps[-1], tp, 'rigid')

    #  - Wall-based information:
    wall_pd_fname = image_dirname + splitext_zip(PI_signal_fname)[0] + '_wall_PIN_PI_signal-D{}.csv'.format(membrane_dist)
    wall_df = pd.read_csv(wall_pd_fname)
    PIN_total_mean = df2labelpair_dict(wall_df, 'PIN_total_mean')
    PI_total_mean = df2labelpair_dict(wall_df, 'PI_total_mean')
    PIN_total_mean_list.append(PIN_total_mean.values())
    PI_total_mean_list.append(PI_total_mean.values())
    label_list.append("{}".format(tp))
    maxi = np.max([maxi, np.max(PIN_total_mean.values() + PI_total_mean.values())]) * 1.01


# fig, subaxes = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
# subaxes[0].set_title("Wall based signal quantification for {} SAM{}.".format(XP, SAM))
#
# subaxes[0].boxplot(PIN_total_mean_list, vert=False, labels=label_list)
# subaxes[0].set_xlabel('PIN mean signal')
#
# subaxes[1].boxplot(PI_total_mean_list, vert=False, labels=label_list)
# subaxes[1].set_xlabel('PI mean signal')
#
# for ax in subaxes:
#     ax.set_ylabel('Time step (h)')
#     ax.xaxis.grid(True)
#     ax.set_xlim(mini, maxi)
#
# plt.tight_layout()
# plt.savefig(image_dirname + "{}-SAM{}-PIN_PI_boxplot_over_time.pdf".format(XP, SAM))
# plt.show()


# - Correlation plot between PIN and PI:
mini = 0
maxi = 0
fig, subaxes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
for tp in time_steps:
    czi_fname = base_fname + "-T{}.czi".format(tp)
    # Get unregistered image filename:
    path_suffix, PI_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, membrane_ch_name)
    if tp != time_steps[-1]:
        path_suffix += 'rigid_registrations/'
        # Get RIDIG registered on last time-point filename:
        PI_signal_fname = get_res_img_fname(PI_signal_fname, time_steps[-1], tp, 'rigid')

    wall_df = pd.read_csv(wall_pd_fname)
    PIN_total_mean = df2labelpair_dict(wall_df, 'PIN_total_mean')
    PI_total_mean = df2labelpair_dict(wall_df, 'PI_total_mean')
    maxi = np.max([maxi, np.max(PIN_total_mean.values() + PI_total_mean.values())]) * 1.01
    subaxes.plot(PIN_total_mean.values(), PI_total_mean.values(), '.', label=czi_fname[:-4])

plt.title("Wall based signal quantification for {} SAM{}.".format(XP, SAM))
subaxes.set_xlim(mini, maxi)
subaxes.set_ylim(mini, maxi)
subaxes.set_xlabel('PIN mean signal')
subaxes.set_ylabel('PI mean signal')
plt.legend()
basename = image_dirname + "{}-SAM{}-PIN_PI_correlation_plot.{}"
plt.savefig(basename.format(XP, SAM, 'pdf'))
# plt.show()

import os
cmd = "gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={} {}".format(basename.format(XP, SAM, 'eps'), basename.format(XP, SAM, 'pdf'))
os.system(cmd)
