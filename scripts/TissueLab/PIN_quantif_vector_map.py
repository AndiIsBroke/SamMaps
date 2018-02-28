# -*- coding: utf-8 -*-
from mayavi import mlab
from os.path import split
import numpy as np
import pandas as pd

from timagetk.components.io import imread, imsave, SpatialImage
# from openalea.tissue_nukem_3d.microscopy_images import imread as read_czi
from timagetk.algorithms import isometric_resampling
from timagetk.algorithms import resample_isotropic

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


# XP = 'E37'
XP = sys.argv[1]
# SAM = '5'
SAM = sys.argv[2]
# tp = 0
tp = int(sys.argv[3])

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
# membrane_ch_name += '_raw'


# By default we register all other channels:
extra_channels = list(set(channel_names) - set([membrane_ch_name]))
# By default do not recompute deformation when an associated file exist:
force = False

# Get unregistered image filename:
path_suffix, PI_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, membrane_ch_name)
path_suffix, PIN_signal_fname = get_nomenclature_channel_fname(czi_fname, nomenclature_file, 'PIN1')
path_suffix, seg_img_fname = get_nomenclature_segmentation_name(czi_fname, nomenclature_file)
path_suffix += 'rigid_registrations/'

# Get RIDIG registered on last time-point filename:
PI_signal_fname = get_res_img_fname(PI_signal_fname, time_steps[-1], tp, 'rigid')
PIN_signal_fname = get_res_img_fname(PIN_signal_fname, time_steps[-1], tp, 'rigid')
seg_img_fname = get_res_img_fname(seg_img_fname, time_steps[-1], tp, 'rigid')

background_value = 1

# - To create a mask, edit a MaxIntensityProj with an image editor (GIMP) by adding 'black' (0 value):
# mask = image_dirname+'PIN1-GFP-CLV3-CH-MS-E1-LD-SAM4-MIP_PI-mask.png'
mask = ''

print "\n\n# - Reading PIN1 intensity image file {}...".format(PIN_signal_fname)
PIN_signal_im = imread(image_dirname + path_suffix + PIN_signal_fname)
# -- Resample signal intensity images since segmentation was performed on isometric image:
# print "\n# - Isometric resampling of PIN1 signal image:"
# PIN_signal_im = isometric_resampling(PIN_signal_im)
# world.add(PIN_signal_im, 'PIN1 intensity image', colormap='viridis', voxelsize=PIN_signal_im.get_voxelsize())

print "\n\n# - Reading PI intensity image file {}...".format(PI_signal_fname)
PI_signal_im = imread(image_dirname + path_suffix + PI_signal_fname)
# -- Resample signal intensity images since segmentation was performed on isometric image:
# print "\n# - Isometric resampling of PI signal image:"
# PI_signal_im = isometric_resampling(PI_signal_im)
# world.add(PI_signal_im, 'PI intensity image', colormap='invert_grey', voxelsize=PI_signal_im.get_voxelsize())

print "\n\n# - Reading segmented image file {}...".format(seg_img_fname)
seg_im = imread(image_dirname + path_suffix + seg_img_fname)
# world.add(seg_im, 'segmented image', colormap='glasbey', voxelsize=seg_im.get_voxelsize())
ori_vxs = PI_signal_im.get_voxelsize()
print "\n# - Resampling segmented image to original voxelsize: {}".format(ori_vxs)
seg_im = resample_isotropic(seg_im, ori_vxs, option='label')


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
def df2labelpair_dict(df, lab1_cname, lab2_cname, values_cname):
    """
    Return a label-pair dictionary with df['values_cname'] as values.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the values
    lab1_cname : str
        dataframe column name containing the first labels of the label-pairs
    lab2_cname : str
        dataframe column name containing the second labels of the label-pairs
    values_cname : str
        dataframe column name containing the values to add to the dictionary

    Returns
    -------
    dictionary {(label_1, label_2): values}
    """
    lab1 = df[lab1_cname]
    lab2 = df[lab2_cname]
    labelpairs = [(lab1[k], lab2[k]) for k in lab1.keys()]
    values = df[values_cname]
    return {lp: values[lp] for lp in labelpairs}


def df2sortedlabelpair_dict(df, lab1_cname, lab2_cname, values_cname):
    """
    Return a label-pair dictionary with df['values_cname'] as values, if this
    values is different from NaN.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the values
    lab1_cname : str
        dataframe column name containing the first labels of the label-pairs
    lab2_cname : str
        dataframe column name containing the second labels of the label-pairs
    values_cname : str
        dataframe column name containing the values to add to the dictionary

    Returns
    -------
    dictionary {(label_1, label_2): values}
    """
    lab1 = df[lab1_cname]
    lab2 = df[lab2_cname]
    labelpairs = [(lab1[k], lab2[k]) for k in lab1.keys()]
    values = df[values_cname]
    return {lp: values[lp] for lp in labelpairs if not np.isnan(values[lp])}


###############################################################################
# -- PIN1/PI signal & PIN1 polarity quatification:
###############################################################################
membrane_dist = 0.6

print "\n\n# - Initialise signal quantification class:"
memb = MembraneQuantif(seg_im, [PIN_signal_im, PI_signal_im], ["PIN1", "PI"])


# - Get the SAMPLED SIGNAL: this is what have been used during quantification!!
sampled_PIN_signal_im = memb.get_whole_membrane_signal_image("PIN1", membrane_dist)
# world.add(PIN_signal_im, 'Sampled PIN1 signal image', colormap='viridis', voxelsize=sampled_PIN_signal_im.get_voxelsize())

# - If you want to plot PIN signal image AND polarity field, you should use barycenters with voxel units:
real_bary = False

# - Get list of 'L1' labels:
labels = memb.labels_checker('L1')

# - Create a list of anticlinal walls (ordered pairs of labels):
walls_min_area = 5.  # to avoid too small walls arising from segmentation errors
L1_anticlinal_walls = memb.L1_anticlinal_walls(min_area=walls_min_area, real_area=True)

# - Cell-based information (barycenters):
cell_df_fname = image_dirname + splitext_zip(PI_signal_fname)[0] + '_cell_barycenters.csv'
try:
    cell_df = pd.read_csv(cell_df_fname)
    assert False
except:
    # - Compute the barycenters of each selected cells:
    bary = memb.center_of_mass(labels, real_bary, verbose=True)

    bary_x = {k: v[0] for k, v in bary.items()}
    bary_y = {k: v[1] for k, v in bary.items()}
    bary_z = {k: v[2] for k, v in bary.items()}
    # - EXPORT data to csv:
    cell_df = pd.DataFrame().from_dict({'bary_x': bary_x,
                                        'bary_y': bary_y,
                                        'bary_z': bary_z})
    cell_df.to_csv(cell_df_fname)
else:
    lab1 = cell_df['Unnamed: 0']
    bary = {k: np.array([cell_df.bary_x[k], cell_df.bary_y[k], cell_df.bary_z[k]]) for k in lab1.keys()}

#  - Wall-based information (barycenters):
wall_pd_fname = image_dirname + splitext_zip(PI_signal_fname)[0] + '_wall_PIN_PI_signal-D{}.csv'.format(membrane_dist)
ratio_method = "mean"
PIN_signal_method = "mean"
PI_signal_method = "mean"
try:
    wall_df = pd.read_csv(wall_pd_fname)
    assert False
except:
    # - Compute the area of each walls (L1 anticlinal walls):
    wall_area = memb.wall_area_from_labelpairs(L1_anticlinal_walls, real=True)

    # - Compute the wall median of each selected walls (L1 anticlinal walls):
    wall_median = {}
    for (lab1, lab2) in L1_anticlinal_walls:
        wall_median[(lab1, lab2)] = memb.wall_median_from_labelpairs(lab1, lab2, real=False, min_area=walls_min_area, real_area=True)

    # - Compute the necessary PIN1 and PI signal ratios:
    PIN_ratio = memb.get_membrane_signal_ratio("PIN1", L1_anticlinal_walls, membrane_dist, ratio_method)
    PI_ratio = memb.get_membrane_signal_ratio("PI", L1_anticlinal_walls, membrane_dist, ratio_method)

    # - Compute the necessary PIN1 and PI total signal (sum each side of the wall):
    PIN_signal = memb.get_membrane_signal_total("PIN1", L1_anticlinal_walls, membrane_dist, PIN_signal_method)
    PI_signal = memb.get_membrane_signal_total("PI", L1_anticlinal_walls, membrane_dist, PI_signal_method)

    wall_median_x = {k: v[0] for k, v in wall_median.items()}
    wall_median_y = {k: v[1] for k, v in wall_median.items()}
    wall_median_z = {k: v[2] for k, v in wall_median.items()}
    # - EXPORT data to csv:
    wall_df = pd.DataFrame().from_dict({'PIN_{}_ratio'.format(ratio_method): PIN_ratio,
                                        'PI_{}_ratio'.format(ratio_method): PI_ratio,
                                        'PI_total_{}'.format(PI_signal_method): PI_signal,
                                        'PIN_total_{}'.format(PIN_signal_method): PIN_signal,
                                        'wall_median_x': wall_median_x,
                                        'wall_median_y': wall_median_y,
                                        'wall_median_z': wall_median_z,
                                        'wall_area': wall_area})
    wall_df.to_csv(wall_pd_fname)
else:
    # - List of labels:
    lab1 = wall_df['Unnamed: 0']
    lab2 = wall_df['Unnamed: 1']
    labelpairs = [(lab1[k], lab2[k]) for k in lab1.keys()]
    # - label-pair dictionaries:
    wall_median_x = {(lab1[k], lab2[k]): v for k, v in wall_df.wall_median_x.items()}
    wall_median_y = {(lab1[k], lab2[k]): v for k, v in wall_df.wall_median_y.items()}
    wall_median_z = {(lab1[k], lab2[k]): v for k, v in wall_df.wall_median_z.items()}
    wall_median = {lp: np.array([wall_median_x[lp], wall_median_y[lp], wall_median_z[lp]]) for lp in labelpairs}
    wall_area = df2labelpair_dict(wall_df, 'Unnamed: 0', 'Unnamed: 1', 'wall_area')
    PI_signal = df2sortedlabelpair_dict(wall_df, 'Unnamed: 0', 'Unnamed: 1', 'PI_total_{}'.format(PI_signal_method))
    PIN_signal = df2sortedlabelpair_dict(wall_df, 'Unnamed: 0', 'Unnamed: 1', 'PIN_total_{}'.format(PIN_signal_method))
    PI_ratio = df2sortedlabelpair_dict(wall_df, 'Unnamed: 0', 'Unnamed: 1', 'PI_{}_ratio'.format(ratio_method))
    PIN_ratio = df2sortedlabelpair_dict(wall_df, 'Unnamed: 0', 'Unnamed: 1', 'PIN_{}_ratio'.format(ratio_method))


# - Filter displayed PIN1 ratios by:
#    - a minimal wall area (already done when computing list of L1 anticlinal walls)
#    - a minimum PI ratio to prevent badbly placed wall to bias the PIN ratio
min_PI_ratio = 0.95
#    - a maximum (invert) PIN ratio value to prevent oultiers to change the scale of vectors
max_signal_ratio = 1/0.7
#    - a maximum wall area to avoid creating ouliers with big values!
walls_max_area = 250.

# - Create the list of filtered label_pairs (found in PIN_ratio) according to previous rules:
label_pairs = []
for lp in L1_anticlinal_walls:
    # filter for max PIN ratio:
    if lp not in PIN_ratio.keys():
        continue
    PIN_r = PIN_ratio[lp]
    if 1/PIN_r > max_signal_ratio:
        continue
    # filter for min PI ratio:
    if lp not in PI_ratio.keys() and rtuple(lp) not in PI_ratio.keys():
        continue
    try:
        PI_r = PI_ratio[lp]
    except:
        PI_r = PI_ratio[rtuple(lp)]
    if PI_r < min_PI_ratio:
        continue
    # filter for max area:
    if wall_area[lp] >= walls_max_area:
        continue
    # append the labelpair:
    label_pairs.append(lp)


# - Define display mode:
object_outline = False
anticlinal_walls = False
cell_outlines = False
PIN_channel = True

PIN_func = PIN_polarity_area
if PIN_func.__name__ == 'PIN_polarity_area':
    max_PIN_polarity = 20
    formulae = 'wall area * (1/PIN1_ratio-1)'
elif PIN_func.__name__ == 'PIN_polarity':
    max_PIN_polarity = 0.4
    formulae = '1 / PIN1_ratio - 1'
else:
    raise ValueError("Unknown function '{}'!".format(PIN_func.__name__))


# - Call a mayavi engine to represent the PIN pumps orientations "WALL MEDIAN":
################################################################################
fig = mlab.figure(size=(3200, 1800))
# - Make ordered lists of neighbor pairs (lab_i, lab_j) for L1 anticlinal walls, used for:
#    - vector origins, here 'lab_i, lab_j' wall median
#    - vector directions, here 'lab_j -> lab_i' using barycenters (ie. flux goes from max to min intensity values)
#    - extent, here 1/PIN_ratio(lab_i, lab_j)

# - Compute vectors orientations and prepare variables for display:
print "\nComputing PIN1 polarity vectors..."
sig_origin, sig_direction, sig_intensity = [], [], []
for lab1, lab2 in label_pairs:
    try:
        w_area = wall_area[(lab1, lab2)]
    except:
        w_area = wall_area[(lab2, lab1)]
    PIN_p = PIN_func(PIN_ratio[(lab1, lab2)], w_area)
    if PIN_p <= max_PIN_polarity:
        try:
            sig_origin.append(wall_median[lab1, lab2])
        except:
            sig_origin.append(wall_median[lab2, lab1])
        try:
            bary_lab1 = bary[lab1]
        except:
            bary_lab1 = memb.center_of_mass([lab1], real_bary, verbose=False)
            bary.update({lab1: bary_lab1})
        try:
            bary_lab2 = bary[lab2]
        except:
            bary_lab2 = memb.center_of_mass([lab2], real_bary, verbose=False)
            bary.update({lab2: bary_lab2})

        sig_direction.append(compute_vect_direction(bary_lab2, bary_lab1))
        sig_intensity.append(PIN_p)

print 'Done!'

if object_outline:
    print "\nObject outline detection (first layer of voxel in contact with background)..."
    import scipy.ndimage as nd
    epidermis_im = memb.voxel_first_layer(False)
    epidermis_im = nd.laplace(epidermis_im)
    x,y,z = np.array(seg_im.voxelsize * np.array(np.where(epidermis_im != 0)).T).T
    mlab.points3d(x, y, z, color=(1.,1.,1.), mode='point', scale_mode='none', opacity=0.3)
    print 'Done!'

if anticlinal_walls:
    print "\nL1 anticlinal walls detection (L1/L1 cell walls)..."
    import scipy.ndimage as nd
#    epidermis_im = memb.voxel_first_layer(False)
    L1_im = memb.get_image_with_labels('L1')
    L1_im = nd.laplace(L1_im)
    x,y,z = np.array(seg_im.voxelsize * np.array(np.where(L1_im != 0)).T).T
    mlab.points3d(x, y, z, color=(1.,1.,1.), mode='point', scale_mode='none', opacity=0.3)
    print 'Done!'

if cell_outlines:
    print "TODO!"
    print 'Done!'
    pass

if PIN_channel:
    signal_threshold = 5000
    print "\nPIN1 signal rendering (threshold={})...".format(signal_threshold)
    sx, sy, sz = sampled_PIN_signal_im.shape
    x, y, z = np.ogrid[0:sx, 0:sy, 0:sz]
    probe_array = sampled_PIN_signal_im[x, y, z]
    # mlab.pipeline.volume(mlab.pipeline.scalar_field(probe_array), color=(0., 1., 0.), vmin=signal_threshold)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(probe_array), color=(1., 0., 1.), vmin=signal_threshold, vmax=2**16)
    print 'Done!'

print "\nPopulating display windows..."
# - Create lists of points of origin for PIN vectors:
xs,ys,zs = np.array(sig_origin).T
# - Create lists of orientations for PIN vectors:
u,v,w = np.array(sig_direction).T
# - Plot the PIN vectors:
PIN_vect = mlab.quiver3d(xs, ys, zs, u, v, w, scalars=sig_intensity, colormap="viridis", scale_mode='scalar', mode='arrow', line_width=5.)
# PIN_vect = mlab.quiver3d(xs, ys, zs, u, v, w, scalars=sig_intensity, colormap="viridis", scale_mode='none', mode='arrow', line_width=5.)
# Set glyph positions
PIN_vect.glyph.glyph_source.glyph_position = 'center'  # 'head', 'center', 'tail'
# Set arrow tip length (to max value), radius and resolution:
PIN_vect.glyph.glyph_source.glyph_source.tip_length = 1.0
PIN_vect.glyph.glyph_source.glyph_source.tip_radius = 0.15
PIN_vect.glyph.glyph_source.glyph_source.tip_resolution = 16
# Set data range for colormap:
PIN_vect.glyph.glyph.range = np.array([0., max_PIN_polarity])
# Set color mode for arrows:
PIN_vect.glyph.color_mode = 'color_by_scalar'
# Set arrows scaling factor:
PIN_vect.glyph.glyph.scale_factor = 25.0

cbar = mlab.colorbar(PIN_vect, title=formulae, orientation='vertical')

engine = mlab.get_engine()
scene = engine.scenes[0]
scene.scene.camera.position = [512.45000004768372, 509.74700911343098, -1204.5392953696139]
scene.scene.camera.focal_point = [512.45000004768372, 509.74700911343098, 107.02180659770966]
scene.scene.camera.view_angle = 30.0
scene.scene.camera.view_up = [0.0, 1.0, 0.0]
scene.scene.camera.clipping_range = [1091.4327201530202, 1591.8209460548978]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()

print "Done!"

png_fname = splitext_zip(image_dirname + PIN_signal_fname)[0]+ '{}-D_{}-scaled_arrowheads.png'.format(PIN_func.__name__[3:], membrane_dist)
print "Saving PNG: {}".format(png_fname)
mlab.savefig(png_fname)
mlab.close()
# mlab.show()

# TODO: compute for all wall with an intensity following the rules
# - min PI ratio = 0.92
# - max PIN ratio = 1/0.7 - 1
# - min area = 6.
# - weight the ratio by total signal detected!
# think to compute the wall medians only for those to save time !
