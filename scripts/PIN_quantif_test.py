import numpy as np
import pandas as pd

from timagetk.components import SpatialImage
from vplants.tissue_analysis.signal_quantification import MembraneQuantif
from vplants.tissue_analysis.misc import stuple

import platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
    dirname = "/data/Meristems/Carlos/PIN_maps/"
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
    dirname = "/projects/SamMaps/"
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")

import sys
sys.path.append(SamMaps_dir+'/scripts/lib/')

from nomenclature import splitext_zip
# Nomenclature file location:
nomenclature_file = SamMaps_dir + "nomenclature.csv"
# OUTPUT directory:
image_dirname = dirname + "nuclei_images/"

back_id = 1
walls_min_area = 5.
membrane_dist = 0.6
real_bary=True
quantif_method = 'mean'
membrane_ch_name = 'PI'
path_suffix = "test_offset/"
# im_tif = "qDII-CLV3-PIN1-PI-E37-LD-SAM7-T5-P2.tif"
im_tif = "qDII-CLV3-PIN1-PI-E37-LD-SAM7-T14-P2.tif"

im_fname = image_dirname + path_suffix + im_tif

from vplants.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_tiff_image
ori = [0, 0, 0]
voxelsize = (0.208, 0.208, 0.677)
signal_names = ['DIIV','PIN1','PI','TagBFP','CLV3']

img_dict = read_tiff_image(im_fname, channel_names=signal_names, voxelsize=voxelsize)
img_dict['PI'] = SpatialImage(img_dict['PI'], voxelsize=voxelsize, origin=ori)
img_dict['PIN1'] = SpatialImage(img_dict['PIN1'], voxelsize=voxelsize, origin=ori)
PIN_signal_im  = img_dict['PIN1']
PI_signal_im = img_dict['PI']

###############################################################################
# -- PI signal segmentation:
###############################################################################
from os.path import exists
seg_img_fname = image_dirname + path_suffix + splitext_zip(im_tif)[0] + '_seg.inr'
if not exists(seg_img_fname):
    print "\n - Performing isometric resampling of the image to segment..."
    from timagetk.algorithms import isometric_resampling
    img2seg = isometric_resampling(PI_signal_im)
    iso_vxs = np.array(img2seg.get_voxelsize())
    iso_shape = img2seg.get_shape()

    print "\n - Performing adaptative histogram equalization of the image to segment..."
    # from equalization import z_slice_equalize_adapthist
    # img2seg = z_slice_equalize_adapthist(img2seg)
    from equalization import z_slice_contrast_stretch
    img2seg = z_slice_contrast_stretch(img2seg)

    # print "\n - Performing TagBFP signal substraction..."
    # import copy as cp
    # vxs = PI_signal_im.get_voxelsize()
    # img_dict['TagBFP'] = SpatialImage(img_dict['TagBFP'], voxelsize=voxelsize, origin=ori)
    # substract_img = morphology(img_dict['TagBFP'], method='erosion', radius=3., iterations=3)
    # img2seg = cp.deepcopy(PI_signal_im)
    # tmp_im = img2seg - substract_img
    # tmp_im[img2seg <= substract_img] = 0
    # img2seg = SpatialImage(tmp_im, voxelsize=vxs, origin=ori)
    # del tmp_im

    print "\n# - Automatic seed detection..."
    from timagetk.plugins import morphology
    from timagetk.plugins import h_transform
    from timagetk.plugins import region_labeling
    from timagetk.plugins import linear_filtering
    std_dev = 1.0
    morpho_radius = 1.0
    h_min = 2200
    # asf_img = morphology(img2seg, max_radius=morpho_radius, method='co_alternate_sequential_filter')
    # ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
    smooth_img = linear_filtering(img2seg, std_dev=std_dev, method='gaussian_smoothing')
    ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')
    seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
    print "Detected {} seeds!".format(len(np.unique(seed_img)))

    print "\n - Performing seeded watershed segmentation..."
    from timagetk.plugins import segmentation
    std_dev = 1.0
    # smooth_img = linear_filtering(img2seg, std_dev=std_dev, method='gaussian_smoothing')
    seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed', try_plugin=False)
    seg_im[seg_im == 0] = back_id
    # world.add(seg_im, 'seg', colormap='glasbey', alphamap='constant')

    from timagetk.io import imsave
    imsave(seg_img_fname, seg_im)
else:
    from timagetk.io import imread
    seg_im = imread(seg_img_fname)


###############################################################################
# -- MAYAVI VISU:
###############################################################################
# from mayavi import mlab
# from os.path import splitext
# import numpy as np
# from timagetk.io import imread
# from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
# dirname = "/data/Meristems/Carlos/PIN_maps/"
# image_dirname = dirname + "nuclei_images/"
# path_suffix = "test_offset/"
# # im_tif = "qDII-CLV3-PIN1-PI-E37-LD-SAM7-T5-P2.tif"
# im_tif = "qDII-CLV3-PIN1-PI-E37-LD-SAM7-T14-P2.tif"
#
# seg_im = imread(image_dirname + path_suffix + splitext(im_tif)[0] + '_seg.inr')
# spia = SpatialImageAnalysis(seg_im, background=1)
#
# l1 = spia.voxel_first_layer(False)
# # -- Compute the wall median of each selected walls (L1 anticlinal walls):
# wall_median = {}
# for lab1, lab2 in spia.list_epidermal_walls():
#     wall_median[(lab1, lab2)] = spia.wall_median_from_labelpairs(lab1, lab2, real=False, min_area=5., real_area=True)
#
# # Add the first layer of voxels:
# x, y, z = np.where(l1 != 0)
# c = [l1[x[n], y[n], z[n]] for n, xx in enumerate(x)]
# mlab.points3d(x, y, z, c, mode='cube', scale_mode='none', scale_factor=1)
# # Add the label ids at the epidermal walls barycenter
# for (lab1, lab2), wm in wall_median.items():
#     if wm is not None:
#         mlab.text3d(wm[0], wm[1], wm[2]-2, str(lab2), scale=5)
# # Start interaction mode:
# mlab.show()


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
wall_pd_fname = image_dirname + path_suffix + splitext_zip(im_tif)[0] + '_wall_PIN_PI_{}_signal-D{}.csv'.format(quantif_method, membrane_dist)
# - Export to CSV:
wall_df.to_csv(wall_pd_fname, index_label=['left_label', 'right_label'])
