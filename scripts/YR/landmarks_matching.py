############################################################################
# - Sequence RIGID registration of 'YR_01':
############################################################################
import numpy as np
from os import mkdir
from os.path import exists
from timagetk.io import imread
from timagetk.io import imsave
from timagetk.io.io_trsf import save_trsf
from timagetk.algorithms.trsf import apply_trsf
from timagetk.plugins.resampling import isometric_resampling
from timagetk.plugins import sequence_registration

dir_path = '/data/Yassin/YR_01/'
# -- Intensity images:
ext = "tif"
int_path = dir_path + 'tifs/'
int_fname = 't{}{}.{}'  # tp, ?, ext
# -- Segmented images:
seg_path = dir_path + 'segmentations/'
seg_fname = 't{}_segmented{}.{}'

time_steps = [0, 10, 18, 24, 32, 40, 48, 57, 64, 72, 81, 88, 96, 104, 112, 120, 128, 132]

time_points = range(len(time_steps))
tp2ts = dict(zip(time_points, time_steps))
ts2tp = dict(zip(time_steps, time_points))

# - Registered intensity image, segmented images & transformation file path:
reg_path = dir_path + 'registered_sequence/'
trsf_name = "t{}-t{}_{}.trsf"  # t0, t1, trsf_type
try:
    mkdir(reg_path)
except OSError:
    pass

# - Load images:
list_image = []
for tp in time_points:
    fname = int_fname.format(tp2ts[tp], '', ext)
    list_image.append(imread(int_path + fname))
    print(list_image[-1].filename)
    print(list_image[-1].shape)
    print(list_image[-1].voxelsize)

ref_tp = time_points[-1]
t_ref = time_steps[-1]
vxs = 0.2415

# - Resample the last time-point (reference) to the registered path:
# -- ISOMETRIC resampling of INTENSITY image:
list_image[ref_tp] = isometric_resampling(list_image[ref_tp], method=vxs, option='gray')
# Save rigid sequence registered intensity images:
fname_res_int = reg_path + int_fname.format(t_ref, '', ext)
print("Saving:", fname_res_int)
imsave(fname_res_int, list_image[ref_tp])
print("Done!\n")
# -- ISOMETRIC resampling of SEGMENTED image:
# Load the segmented image:
seg_im = imread(seg_path + seg_fname.format(t_ref, '', ext))
seg_im = isometric_resampling(seg_im, method=vxs, option='label')
# Save rigid sequence registered segmented images:
fname_res_seg = reg_path + seg_fname.format(t_ref, '', ext)
print("Saving:", fname_res_seg)
imsave(fname_res_seg, seg_im)
print("Done!\n")


# - Performs sequence registration:
list_trsf = sequence_registration(list_image, method='rigid', pyramid_lowest_level=2)

global_shape = list_image[ref_tp].shape
global_vxs = [vxs, vxs, vxs]

# - Apply sequence transformations to intensity and segmented images:
for n, trsf in enumerate(list_trsf):
    float_tp = n
    t_float, t_ref = tp2ts[float_tp], tp2ts[ref_tp]
    fname_trsf = reg_path + trsf_name.format(t_float, t_ref, 'rigid')
    # Save sequence rigid transformations:
    print("Saving:", fname_trsf)
    save_trsf(trsf, fname_trsf)
    print("Done!\n")
    # Defines new template for intensity image:
    int_template = {"shape": global_shape,
                    # "voxelsize": list_image[n].voxelsize,
                    "voxelsize": global_vxs,
                    "np_type": list_image[n].dtype}
    # Save rigid sequence registered intensity images:
    fname_res_int = reg_path + int_fname.format(t_float, '', ext)
    print("Saving:", fname_res_int)
    # list_image[n] = apply_trsf(list_image[n], trsf, template_img=list_image[ref_tp], param_str_2='-linear')
    list_image[n] = apply_trsf(list_image[n], trsf, template_img=int_template, param_str_2='-linear')
    imsave(fname_res_int, list_image[n])
    print("Done!\n")
    # Load the segmented image:
    seg_im = imread(seg_path + seg_fname.format(t_float, '', ext))
    # Defines new template for intensity image:
    seg_template = {"shape": list_image[ref_tp].shape,
                    "voxelsize": list_image[n].voxelsize,
                    "np_type": np.uint16}
    # Save rigid sequence registered segmented images:
    fname_res_seg = reg_path + seg_fname.format(t_float, '', ext)
    print("Saving:", fname_res_seg)
    # imsave(fname_res_seg, apply_trsf(seg_im, trsf, template_img=list_image[ref_tp].astype('uint16'), param_str_2='-nearest'))
    imsave(fname_res_seg, apply_trsf(seg_im, trsf, template_img=seg_template, param_str_2='-nearest'))
    print("Done!\n")



############################################################################
# - Animate the registered sequence using 2D projections:
############################################################################
from timagetk.io import imread
from timagetk.algorithms.exposure import global_contrast_stretch
from timagetk.visu.temporal_mip import sequence_mips
from timagetk.visu.temporal_mip import mips2gif

dir_path = '/data/Yassin/YR_01/'
ext = "tif"
# -- Intensity images:
int_path = dir_path + 'tifs/'
int_fname = 't{}{}.{}'  # tp, ?, ext
# -- Segmented images:
seg_path = dir_path + 'segmentations/'
seg_fname = 't{}_segmented{}.{}'

time_steps = [0, 10, 18, 24, 32, 40, 48, 57, 64, 72, 81, 88, 96, 104, 112, 120, 128, 132]

time_points = range(len(time_steps))
tp2ts = dict(zip(time_points, time_steps))
ts2tp = dict(zip(time_steps, time_points))

# - Registered intensity image, segmented images & transformation file path:
reg_path = dir_path + 'registered_sequence/'
trsf_name = "t{}-t{}_{}.trsf"  # t0, t1, trsf_type

list_image = []
for tp in time_points:
    fname = int_fname.format(tp2ts[tp], '', ext)
    list_image.append(imread(reg_path + fname))


list_image = [global_contrast_stretch(img, pc_min=1, pc_max=99) for img in list_image]

mips = sequence_mips(list_image, method='maximum', export_png=True, invert_axis=True)
mips2gif(mips, dir_path + "YR01_TimeLapse-maximum_proj.gif", reshape=[512, 512])

# mips = sequence_mips(list_image, method='contour', export_png=True, invert_axis=True)
# mips2gif(mips, dir_path + "YR01_TimeLapse-contour_proj.gif", reshape=[512, 512])


############################################################################
# - COPY to 'YR_01_ATLAS_iso' (temporal down-sampling):
############################################################################
from shutil import copy2 as copy
ext = "tif"
dir_path = '/data/Yassin/YR_01/'
reg_path = dir_path + 'registered_sequence/'
trsf_name = "t{}-t{}_{}.trsf"  # t0, t1, trsf_type

atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
# -- Intensity images:
int_path = atlas_path + 'tifs/'
int_fname = 't{}{}.{}'  # tp, ?, ext
# -- Segmented images:
seg_path = atlas_path + 'segmentations/'
seg_fname = 't{}_segmented{}.{}'

# -- Registered intensity image, segmented images & transformation file path:
atlas_reg_path = atlas_path + 'registered_sequence/'

atlas_time_steps = [0, 10, 40, 96, 120, 132]
time_points = range(len(atlas_time_steps))
tp2ts = dict(zip(time_points, atlas_time_steps))
ts2tp = dict(zip(atlas_time_steps, time_points))

t_ref = atlas_time_steps[-1]
for t_float in atlas_time_steps[:-1]:
    # - Intensity image:
    fname = int_fname.format(t_float, '', ext)
    copy(reg_path + fname, atlas_reg_path)
    # - Segmented image:
    fname = seg_fname.format(t_float, '', ext)
    copy(reg_path + fname, atlas_reg_path)
    # - Transformation:
    fname = trsf_name.format(t_float, t_ref, 'rigid')
    copy(reg_path + fname, atlas_reg_path)

# - Also copy the last time-point (reference) to the ATLAS registered path:
from shutil import copy2 as copy
copy(reg_path + int_fname.format(t_ref, '', ext), atlas_reg_path)
copy(reg_path + seg_fname.format(t_ref, '', ext), atlas_reg_path)


############################################################################
def ldmk_pairing(ldmk0, ldmk1, ids=None):
    """ Returns two paired arrays from two dictionary of landmarks. """
    if ids is None:
        ids = set(ldmk0.keys()) & set(ldmk1.keys())
    ldmk = np.array([[ldmk0[id], ldmk1[id]] for id in ids if id in ldmk0 and id in ldmk1])
    return ldmk[:, 0], ldmk[:, 1]


############################################################################
# - EXTRACT barycenters and COMPUTE pointmatching non-linear deformation:
############################################################################
import pickle
import numpy as np
from timagetk.io import imread
from timagetk.io import imsave
from timagetk.components import TissueImage
from timagetk.components.labelled_image import relabel_from_mapping

from vplants.tissue_analysis.lineage import Lineage
from vplants.tissue_analysis.tissue_analysis import TissueAnalysis

dir_path = '/data/Yassin/YR_01/'
atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
int_path = atlas_path + 'tifs/'
seg_path = atlas_path + 'segmentations/'
# -- Registered intensity image, segmented images & transformation file path:
atlas_reg_path = atlas_path + 'registered_sequence/'

# -- Intensity images:
ext = "tif"
int_fname = 't{}{}.{}'  # tp, ?, ext
# -- Segmented images:
seg_fname = 't{}_segmented{}.{}'
# -- Registered intensity image, segmented images & transformation file path:
trsf_name = "t{}-t{}_{}.trsf"  # t0, t1, trsf_type

atlas_time_steps = [0, 10, 40, 96, 120, 132]
time_points = range(len(atlas_time_steps))
tp2ts = dict(zip(time_points, atlas_time_steps))
ts2tp = dict(zip(atlas_time_steps, time_points))

for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    # - Defines segmented image filenames:
    t0_seg = atlas_reg_path + seg_fname.format(t0, '', ext)
    t1_seg = atlas_reg_path + seg_fname.format(t1, '', ext)

    # -- Lineage files:
    lin_file = atlas_path + 'lineages/' + 'lineage_{}h_to_{}h.txt'.format(t0, t1)

    # - File names and locations for landmarks:
    pts1_fname = atlas_reg_path + '{}_landmarks-t{}_relab_t{}.txt'
    pts0_fname = atlas_reg_path + '{}_landmarks-t{}{}.txt'

    # - Load images:
    print("\n# - READING INPUT IMAGES:")
    # Load t_n & t_n+1 SEGMENTED images:
    print("Reading file: '{}'".format(t0_seg))
    sim0 = imread(t0_seg)
    print("Reading file: '{}'".format(t1_seg))
    sim1 = imread(t1_seg)

    print("\n# - LOADING LINEAGE:")
    lin = Lineage(lin_file, lineage_fmt='marsalt')
    print("Got {} ancestors!".format(len(lin.ancestors_list())))

    print("\n# - INITIALIZING TissueImage:")
    # - Create TissueImage:
    sim0 = TissueImage(sim0, background=1, no_label_id=0, voxelsize=sim0.voxelsize)
    sim1 = TissueImage(sim1, background=1, no_label_id=0, voxelsize=sim1.voxelsize)

    print("\n# - RELABELLING T_n SEGMENTATION:")
    # -- Keep only lineaged cells @ T_n:
    sim0 = sim0.get_image_with_labels(lin.ancestors_list()+[1])  # also keep the background
    sim0 = TissueImage(sim0, background=1, no_label_id=0, voxelsize=sim0.voxelsize)

    print("\n# - RELABELLING T_n+1 SEGMENTATION:")
    # -- Relabel T_n+1 image with T_n lineaged labels:
    mapping = {desc: anc for anc, desc_list in lin.lineage_dict.items() for desc in desc_list}
    mapping.update({1: 1})  # add background to mapping to avoid loosing it with `clear_unmapped=True`
    sim1 = relabel_from_mapping(sim1, mapping, clear_unmapped=True, verbose=True)

    print("\n# - INITIALIZING TissueAnalysis OBJECTS:")
    tissue0 = TissueAnalysis(sim0, background=1)
    tissue1 = TissueAnalysis(sim1, background=1)

    print("\n# - DEFINING LABELS OF INTEREST:")
    # - Defines list of labels of interest (ie. labels not neighbors of `no_label_id`):
    nal_neighbors0 = tissue0.label.image.neighbors(0)  # nal = Not A Label
    print("Got {} neighbors to `no_label_id` ({})".format(len(nal_neighbors0), tissue0.label.image.no_label_id))
    nal_neighbors1 = tissue1.label.image.neighbors(0)  # nal = Not A Label
    print("Got {} neighbors to `no_label_id` ({})".format(len(nal_neighbors1), tissue1.label.image.no_label_id))
    # sim0.cells() only contains lineaged cells since we did: ```sim0.get_image_with_labels(lin.ancestors_list())```
    loi = list(set(sim0.cells()) - set(nal_neighbors0) - set(nal_neighbors1))
    print("Found {} labels of interest!".format(len(loi)))

    print("\n# - EXTRACTING POINTS OF INTEREST:")
    # - Extract Points Of Interest:
    pts0, pts1 = {}, {}

    # - Add cell barycenters as POI:
    com0 = tissue0.label.center_of_mass(real=True)
    com1 = tissue1.label.center_of_mass(real=True)
    c_ids = set(com0.keys()) & set(com1.keys()) - set([1])
    print(
        "Found {} common ids between T_n (n={}) and T_n+1 (n={}) cell center of mass!".format(
            len(c_ids), len(com0), len(com1)))
    for k in c_ids:
        pts0.update({k: v for k, v in com0.items() if k in c_ids})
        pts1.update({k: v for k, v in com1.items() if k in c_ids})

    # - Add wall medians as POI:
    wm0 = tissue0.wall.get_medians(real=True)
    wm1 = tissue1.wall.get_medians(real=True)
    w_ids = set(wm0.keys()) & set(wm1.keys())
    print(
        "Found {} common ids between T_n (n={}) and T_n+1 (n={}) wall median!".format(
            len(w_ids), len(wm0), len(wm1)))
    for k in w_ids:
        pts0.update({k: v for k, v in wm0.items() if k in w_ids})
        pts1.update({k: v for k, v in wm1.items() if k in w_ids})

    # - Add edge medians as POI:
    em0 = tissue0.edge.get_medians(real=True)
    em1 = tissue1.edge.get_medians(real=True)
    e_ids = set(em0.keys()) & set(em1.keys())
    print(
        "Found {} common ids between T_n (n={}) and T_n+1 (n={}) edge median!".format(
            len(e_ids), len(em0), len(em1)))
    for k in e_ids:
        pts0.update({k: v for k, v in em0.items() if k in e_ids})
        pts1.update({k: v for k, v in em1.items() if k in e_ids})

    # - Add vertex medians as POI:
    vm0 = tissue0.vertex.get_medians(real=True)
    vm1 = tissue1.vertex.get_medians(real=True)
    v_ids = set(vm0.keys()) & set(vm1.keys())
    print(
        "Found {} common ids between T_n (n={}) and T_n+1 (n={}) vertex median!".format(
            len(v_ids), len(vm0), len(vm1)))
    for k in v_ids:
        pts0.update({k: v for k, v in vm0.items() if k in v_ids})
        pts1.update({k: v for k, v in vm1.items() if k in v_ids})

    # - Save the landmarks:
    ldmk_fname = atlas_reg_path + 'all_landmarks-t{}_t{}.pkl'  # t_n, t_n+1
    pkl_file = open(ldmk_fname.format(t0, t1), 'wb')
    pickle.dump([pts0, pts1], pkl_file)
    pkl_file.close()
    # -- Save ALL landmarks:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1)
    np.savetxt(pts1_fname.format('all', t1, t0), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('all', t0, ''), ldmk0, fmt="%.8f")
    # -- Save the cell barycenters:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1, c_ids)
    np.savetxt(pts1_fname.format('cell_barycenters', t1, t0), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('cell_barycenters', t0, ''), ldmk0, fmt="%.8f")
    # -- Save the wall medians:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1, w_ids)
    np.savetxt(pts1_fname.format('wall_medians', t1, t0), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('wall_medians', t0, ''), ldmk0, fmt="%.8f")
    # -- Save the edge medians:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1, e_ids)
    np.savetxt(pts1_fname.format('edge_medians', t1, t0), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('edge_medians', t0, ''), ldmk0, fmt="%.8f")
    # -- Save the wall medians:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1, v_ids)
    np.savetxt(pts1_fname.format('vertex_medians', t1, t0), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('vertex_medians', t0, ''), ldmk0, fmt="%.8f")


%run /data/Meristems/Carlos/SamMaps/scripts/YR/quiver3d_visu.py /data/Yassin/YR_01_ATLAS_iso/registered_sequence/cell_barycenters_landmarks-t10_relab_t0.txt /data/Yassin/YR_01_ATLAS_iso/registered_sequence/cell_barycenters_landmarks-t0.txt

%run /data/Meristems/Carlos/SamMaps/scripts/YR/quiver3d_visu.py /data/Yassin/YR_01_ATLAS_iso/registered_sequence/all_landmarks-t10_relab_t0.txt /data/Yassin/YR_01_ATLAS_iso/registered_sequence/all_landmarks-t0.txt


############################################################################
############################################################################
def mad_outlier(points, thresh=3.5):
    """
    The median absolute deviation (MAD) is a robust measure of the variability
    of a univariate sample of quantitative data.

    For a univariate data set X1, X2, ..., Xn, the MAD is defined as the median
    of the absolute deviations from the data's median:
    $$ MAD(X) = median_i(|X_i - median_j(X_j)|) $$

    In order to use the MAD as a consistent estimator for the estimation of the
    standard deviation $\sigma$, one takes:
    $$ \hat{\sigma} = K . MAD, $$
    where $K$ is a constant scale factor, which depends on the distribution.

    Parameters
    ----------
    points : iterable
        a list or array of values to evaluate for outliers presence
    thresh : float
        the z-score threshold defining outliers

    Returns
    -------
    list(bool)
        True if points[i] is an outliers else False
    """
    # - Compute the mean absolute deviation of each value:
    median = np.median(points, axis=0)
    abs_dev2median = np.abs(points - median)
    median_abs_dev2median = 1.4826 * np.median(abs_dev2median)
    # - Compute a modified Z-score:
    score = abs_dev2median / median_abs_dev2median

    return np.array(score) > thresh


############################################################################
# - Filtering WEIRD lineages based on the norm of the vector defined between their barycenters
############################################################################
import numpy as np
from vplants.tissue_analysis.lineage import Lineage
from vplants.tissue_analysis.tissue_analysis import TissueAnalysis

atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
int_path = atlas_path + 'tifs/'
seg_path = atlas_path + 'segmentations/'
# -- Registered intensity image, segmented images & transformation file path:
atlas_reg_path = atlas_path + 'registered_sequence/'

# -- Intensity images:
ext = "tif"
int_fname = 't{}{}.{}'  # tp, ?, ext
# -- Segmented images:
seg_fname = 't{}_segmented{}.{}'
# -- Registered intensity image, segmented images & transformation file path:
trsf_name = "t{}-t{}_{}.trsf"  # t0, t1, trsf_type

atlas_time_steps = [0, 10, 40, 96, 120, 132]
time_points = range(len(atlas_time_steps))

vxs = 0.2415
global_vxs = [vxs, vxs, vxs]

norms = []
for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    poi = "cell_barycenters"  # can be among: "all", "cell_barycenters", "wall_medians", "edge_medians", "vertex_medians"

    # - File names and locations for landmarks:
    pts1_fname = atlas_reg_path + '{}_landmarks-t{}_relab_t{}.txt'.format(poi, t1, t0)
    pts0_fname = atlas_reg_path + '{}_landmarks-t{}{}.txt'.format(poi, t0, '')
    # - File names and locations for landmarks without outliers:
    pts1_no_fname = atlas_reg_path + '{}_landmarks-no_outliers-t{}_relab_t{}.txt'.format(poi, t1, t0)
    pts0_no_fname = atlas_reg_path + '{}_landmarks-no_outliers-t{}{}.txt'.format(poi, t0, '')

    # - Load landmarks:
    print("\n# - READING LANDMARKS FILES:")
    pts0 = np.loadtxt(pts0_fname)
    pts1 = np.loadtxt(pts1_fname)

    # - Compute the norms of the vectors defined by landmarks (lineaged cell barycenters)
    norms.append(np.linalg.norm(pts1 - pts0, axis=1))

    # - Detecting outliers:
    print("\n# - DETECTING OUTLIERS:")
    norms_outliers = mad_outlier(norms[-1])
    print("Found {}!".format(np.sum(norms_outliers)))

    print("\n# - SAVING LANDMARKS FILES (WITHOUT OUTLIERS):")
    pts0 = [pts for n, pts in enumerate(pts0) if not norms_outliers[n]]
    pts1 = [pts for n, pts in enumerate(pts1) if not norms_outliers[n]]
    np.savetxt(pts0_no_fname, pts0, fmt="%.8f")
    print("Saved {}".format(pts0_no_fname))
    np.savetxt(pts1_no_fname, pts1, fmt="%.8f")
    print("Saved {}".format(pts1_no_fname))


# - Histogram of vectorfield norms with homogeneous x-axis range:
import matplotlib.pylab as plt

max_norm = 0
for tp, ts in enumerate(atlas_time_steps[:-1]):
    mn = np.max(norms[tp])
    if mn > max_norm:
        max_norm = mn

fig, axes = plt.subplots(nrows=len(norms))
for tp, ts in enumerate(atlas_time_steps[:-1]):
    axes[tp].hist(norms[tp], bins=256, range=[0, max_norm])
    axes[tp].set_title("t{} - t{}".format(ts, atlas_time_steps[tp+1]))
    axes[tp].set_ylabel("Frequency")

axes[tp].set_xlabel("Lineaged barycenter distances (µm)")


# - Boxplot of vectorfield norms
plt.figure()
plt.boxplot(norms, labels=["t{} - t{}".format(ts, atlas_time_steps[tp+1]) for tp, ts in enumerate(atlas_time_steps[:-1])], vert=False)
plt.title("Lineaged barycenter distances (µm)")


############################################################################
# - Visualisation of landmarks vectors:
############################################################################
import numpy as np
from mayavi import mlab

def mlab_vectorfield(pts0, pts1):
    n_pts = len(pts0)
    x0, y0, z0 = pts0.T
    x1, y1, z1 = pts1.T

    mlab.figure()
    mlab.points3d(x0, y0, z0, range(n_pts), mode="sphere",
                  scale_mode='none', scale_factor=0.2, colormap='prism')
    mlab.points3d(x1, y1, z1, range(n_pts), mode="cube",
                  scale_mode='none', scale_factor=0.2, colormap='prism')
    mlab.quiver3d(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, line_width=1.5,
                  mode="2darrow", scale_factor=1)
    mlab.show()

# - ALL POI AS LANDMARKS:
atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
atlas_reg_path = atlas_path + 'registered_sequence/'
atlas_time_steps = [0, 10, 40, 96, 120, 132]

for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    # - File names and locations for landmarks:
    pts1_fname = atlas_reg_path + 'all_landmarks-t{}_relab_t{}.txt'.format(t1, t0)
    pts0_fname = atlas_reg_path + 'all_landmarks-t{}.txt'.format(t0)
    pts0 = np.loadtxt(pts0_fname)
    pts1 = np.loadtxt(pts1_fname)
    mlab_vectorfield(pts0, pts1)

# - CELL BARYCENTER AS LANDMARKS:
atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
atlas_reg_path = atlas_path + 'registered_sequence/'
atlas_time_steps = [0, 10, 40, 96, 120, 132]

for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    # - File names and locations for landmarks:
    pts1_fname = atlas_reg_path + 'cell_barycenters_landmarks-t{}_relab_t{}.txt'.format(t1, t0)
    pts0_fname = atlas_reg_path + 'cell_barycenters_landmarks-t{}.txt'.format(t0)
    pts0 = np.loadtxt(pts0_fname)
    pts1 = np.loadtxt(pts1_fname)
    mlab_vectorfield(pts0, pts1)


############################################################################
# - POINTMATCHING PARAMETERS ESTIMATION
# In order to set the `-vector-propagation-distance` (voxel unit) we estimate
# the mean initialisation vectorfield norm as the average distance to neighbors
############################################################################
import numpy as np
from timagetk.io import imread
from timagetk.io import imsave
from timagetk.components import TissueImage

from vplants.tissue_analysis.lineage import Lineage
from vplants.tissue_analysis.tissue_analysis import LabelProperty3D

dir_path = '/data/Yassin/YR_01/'
atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
int_path = atlas_path + 'tifs/'
seg_path = atlas_path + 'segmentations/'
# -- Registered intensity image, segmented images & transformation file path:
atlas_reg_path = atlas_path + 'registered_sequence/'

# -- Intensity images:
ext = "tif"
int_fname = 't{}{}.{}'  # tp, ?, ext
# -- Segmented images:
seg_fname = 't{}_segmented{}.{}'
# -- Registered intensity image, segmented images & transformation file path:
trsf_name = "t{}-t{}_{}.trsf"  # t0, t1, trsf_type

atlas_time_steps = [0, 10, 40, 96, 120, 132]
time_points = range(len(atlas_time_steps))
tp2ts = dict(zip(time_points, atlas_time_steps))
ts2tp = dict(zip(atlas_time_steps, time_points))

av_dist = []
for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    # - Defines segmented image filenames:
    t0_seg = atlas_reg_path + seg_fname.format(t0, '', ext)
    t1_seg = atlas_reg_path + seg_fname.format(t1, '', ext)

    # -- Lineage files:
    lin_file = atlas_path + 'lineages/' + 'lineage_{}h_to_{}h.txt'.format(t0, t1)

    # - Load images:
    print("\n# - READING INPUT IMAGES:")
    # Load t_n & t_n+1 SEGMENTED images:
    print("Reading file: '{}'".format(t0_seg))
    sim0 = imread(t0_seg)
    print("Reading file: '{}'".format(t1_seg))
    sim1 = imread(t1_seg)

    print("\n# - LOADING LINEAGE:")
    lin = Lineage(lin_file, lineage_fmt='marsalt')
    print("Got {} ancestors!".format(len(lin.ancestors_list())))

    print("\n# - INITIALIZING TissueImage:")
    # - Create TissueImage:
    sim0 = TissueImage(sim0, background=1, no_label_id=0, voxelsize=sim0.voxelsize)
    sim1 = TissueImage(sim1, background=1, no_label_id=0, voxelsize=sim1.voxelsize)

    print("\n# - RELABELLING T_n SEGMENTATION:")
    # -- Keep only lineaged cells @ T_n:
    sim0 = sim0.get_image_with_labels(lin.ancestors_list()+[1])  # also keep the background
    sim0 = TissueImage(sim0, background=1, no_label_id=0, voxelsize=sim0.voxelsize)

    print("\n# - RELABELLING T_n+1 SEGMENTATION:")
    # -- Relabel T_n+1 image with T_n lineaged labels:
    mapping = {desc: anc for anc, desc_list in lin.lineage_dict.items() for desc in desc_list}
    mapping.update({1: 1})  # add background to mapping to avoid loosing it with `clear_unmapped=True`
    sim1 = relabel_from_mapping(sim1, mapping, clear_unmapped=True, verbose=True)

    print("\n# - INITIALIZING LabelProperty3D OBJECTS:")
    from vplants.tissue_analysis.tissue_analysis import TissueAnalysis
    tissue0 = LabelProperty3D(sim0, background=1)
    tissue1 = LabelProperty3D(sim1, background=1)

    print("\n# - EXTRACTING AVERAGE DISTANCE TO NEIGHBORS:")
    # - Compute average distance to neighbors (in VOXELS):
    d2n0 = tissue0.distance_to_neighbors(real=False)
    d2n1 = tissue1.distance_to_neighbors(real=False)

    dist = []
    for k, v in d2n0.items():
        dist.extend(v)
    av_dist.append(np.mean(dist))

print("\n# - AVERAGE DISTANCE TO NEIGHBORS:", av_dist)


############################################################################
# - POINTMATCHING DEFORMATION
############################################################################
import os
import copy as cp
import subprocess

# - Get environment variables:
var_env = os.environ
new_venv = cp.copy(var_env)
# - Edit them to remove reference to timagetk library:
new_venv['LD_LIBRARY_PATH'] = ''

# t_float, t_ref, path, vpd, fd, fs
cmd = "/data/Meristems/Carlos/SamMaps/scripts/YR/pointmatching.sh {} {} {} {} {} {}"

atlas_time_steps = [0, 10, 40, 96, 120, 132]
for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    vpd = 2/3. * av_dist[tp]
    fd = 1/3. * av_dist[tp]
    fs = 2/5. * av_dist[tp]
    shell_cmd = cmd.format(t0, t1, '/data/Yassin/YR_01_ATLAS_iso/registered_sequence', vpd, fd, fs)
    print(shell_cmd)
    subprocess.run(shell_cmd, shell=True, capture_output=True, env=new_venv)

subprocess.run('export', shell=True, env=var_env)


############################################################################
# - BLEND reference and registered floating images:
# This is done to visualy assess registration quality
############################################################################
import numpy as np
from timagetk.io import imread
from timagetk.visu.animations import animate_channel_blending

atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
# -- Registered intensity image, segmented images & transformation file path:
atlas_reg_path = atlas_path + 'registered_sequence/'

# -- Intensity images:
ext = "tif"
int_fname = 't{}{}.{}'  # tp, ?, ext
vf_int_fname = 't{}-on-t{}.{}'  # ts, ts+1, ext

atlas_time_steps = [0, 10, 40, 96, 120, 132]

for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]

    # - Defines intensity image filenames:
    t0_int = atlas_reg_path + vf_int_fname.format(t0, t1, ext)
    t1_int = atlas_reg_path + int_fname.format(t1, '', ext)

    # - Load images:
    print("\n# - READING INPUT IMAGES:")
    # Load t_n & t_n+1 INTENSITY images:
    print("Reading file: '{}'".format(t0_int))
    iim0 = imread(t0_int)
    print("Reading file: '{}'".format(t1_int))
    iim1 = imread(t1_int)

    fname = atlas_reg_path + vf_int_fname.format(t0, t1, 'mp4')
    animate_channel_blending([iim0, iim1], fname, colors=('red', 'green'), duration=10.)


############################################################################
# - EXTRACT LANDMARKS FROM REGISTERED FLOATING IMAGES:
############################################################################
import pickle
import numpy as np
from timagetk.io import imread
from timagetk.components import TissueImage
from vplants.tissue_analysis.lineage import Lineage
from vplants.tissue_analysis.tissue_analysis import TissueAnalysis
import pandas as pd

atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
atlas_reg_path = atlas_path + 'registered_sequence/'

ext = 'tif'
# -- Segmented images:
seg_path = atlas_path + 'segmentations/'
seg_fname = 't{}_segmented{}.{}'
vf_seg_fname = 't{}-on-t{}_segmented.{}'  # ts, ts+1, ext

atlas_time_steps = [0, 10, 40, 96, 120, 132]

for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    # - Defines segmented image filenames:
    t0_seg = atlas_reg_path + vf_seg_fname.format(t0, t1, ext)
    t1_seg = atlas_reg_path + seg_fname.format(t1, '', ext)

    # -- Lineage files:
    lin_file = atlas_path + 'lineages/' + 'lineage_{}h_to_{}h.txt'.format(t0, t1)

    # - File names and locations for landmarks:
    pts0_fname = atlas_reg_path + '{}_landmarks-t{}{}.txt'  # POI_type, t0, '-vf'
    pts1_fname = atlas_reg_path + '{}_landmarks-t{}_relab_t{}{}.txt'  # POI_type, t0, t1, '-vf'

    # - Load images:
    print("\n# - READING INPUT IMAGES:")
    # Load t_n & t_n+1 SEGMENTED images:
    print("Reading file: '{}'".format(t0_seg))
    sim0 = imread(t0_seg)
    print("Reading file: '{}'".format(t1_seg))
    sim1 = imread(t1_seg)

    print("\n# - LOADING LINEAGE:")
    lin = Lineage(lin_file, lineage_fmt='marsalt')
    print("Got {} ancestors!".format(len(lin.ancestors_list())))

    print("\n# - INITIALIZING TissueImage :")
    # - Create TissueImage:
    sim0 = TissueImage(sim0, background=1, no_label_id=0, voxelsize=sim0.voxelsize)
    sim1 = TissueImage(sim1, background=1, no_label_id=0, voxelsize=sim1.voxelsize)

    print("\n# - RELABELLING T_n SEGMENTATION:")
    # -- Keep only lineaged cells @ T_n:
    sim0 = sim0.get_image_with_labels(lin.ancestors_list()+[1])  # also keep the background
    sim0 = TissueImage(sim0, background=1, no_label_id=0, voxelsize=sim0.voxelsize)

    print("\n# - RELABELLING T_n+1 SEGMENTATION:")
    # -- Relabel T_n+1 image with T_n lineaged labels:
    mapping = {desc: anc for anc, desc_list in lin.lineage_dict.items() for desc in desc_list}
    mapping.update({1: 1})  # add background to mapping to avoid loosing it with `clear_unmapped=True`
    sim1 = relabel_from_mapping(sim1, mapping, clear_unmapped=True, verbose=True)

    print("\n# - INITIALIZING TissueAnalysis OBJECTS:")
    tissue0 = TissueAnalysis(sim0, background=1)
    tissue1 = TissueAnalysis(sim1, background=1)

    print("\n# - EXTRACTING POINTS OF INTEREST:")
    # - Extract Points Of Interest:
    pts0, pts1 = {}, {}

    # - Add cell barycenters as POI:
    com0 = tissue0.label.center_of_mass(real=True)
    com1 = tissue1.label.center_of_mass(real=True)
    c_ids = set(com0.keys()) & set(com1.keys()) - set([1])
    print(
        "Found {} common ids between T_n (n={}) and T_n+1 (n={}) cell center of mass!".format(
            len(c_ids), len(com0), len(com1)))
    for k in c_ids:
        pts0.update({k: v for k, v in com0.items() if k in c_ids})
        pts1.update({k: v for k, v in com1.items() if k in c_ids})

    # - Add wall medians as POI:
    wm0 = tissue0.wall.get_medians(real=True)
    wm1 = tissue1.wall.get_medians(real=True)
    w_ids = set(wm0.keys()) & set(wm1.keys())
    print(
        "Found {} common ids between reference (n={}) and floating (n={}) wall median!".format(
            len(w_ids), len(wm0), len(wm1)))
    for k in w_ids:
        pts0.update({k: v for k, v in wm0.items() if k in w_ids})
        pts1.update({k: v for k, v in wm1.items() if k in w_ids})

    # - Add edge medians as POI:
    em0 = tissue0.edge.get_medians(real=True)
    em1 = tissue1.edge.get_medians(real=True)
    e_ids = set(em0.keys()) & set(em1.keys())
    print(
        "Found {} common ids between reference (n={}) and floating (n={}) edge median!".format(
            len(e_ids), len(em0), len(em1)))
    for k in e_ids:
        pts0.update({k: v for k, v in em0.items() if k in e_ids})
        pts1.update({k: v for k, v in em1.items() if k in e_ids})

    # - Add vertex medians as POI:
    vm0 = tissue0.vertex.get_medians(real=True)
    vm1 = tissue1.vertex.get_medians(real=True)
    v_ids = set(vm0.keys()) & set(vm1.keys())
    print(
        "Found {} common ids between reference (n={}) and floating (n={}) vertex median!".format(
            len(v_ids), len(vm0), len(vm1)))
    for k in v_ids:
        pts0.update({k: v for k, v in vm0.items() if k in v_ids})
        pts1.update({k: v for k, v in vm1.items() if k in v_ids})

    # - Save the landmarks:
    ldmk_fname = atlas_reg_path + 'all_landmarks-t{}vf_t{}.pkl'  # t_n, t_n+1
    pkl_file = open(ldmk_fname.format(t0, t1), 'wb')
    pickle.dump([pts0, pts1], pkl_file)
    pkl_file.close()
    # -- Save ALL landmarks:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1)
    np.savetxt(pts1_fname.format('all', t1, t0, '-vf'), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('all', t0, '-vf'), ldmk0, fmt="%.8f")
    # -- Save the cell barycenters:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1, c_ids)
    np.savetxt(pts1_fname.format('cell_barycenters', t1, t0, '-vf'), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('cell_barycenters', t0, '-vf'), ldmk0, fmt="%.8f")
    # -- Save the wall medians:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1, w_ids)
    np.savetxt(pts1_fname.format('wall_medians', t1, t0, '-vf'), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('wall_medians', t0, '-vf'), ldmk0, fmt="%.8f")
    # -- Save the edge medians:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1, e_ids)
    np.savetxt(pts1_fname.format('edge_medians', t1, t0, '-vf'), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('edge_medians', t0, '-vf'), ldmk0, fmt="%.8f")
    # -- Save the wall medians:
    ldmk0, ldmk1 = ldmk_pairing(pts0, pts1, v_ids)
    np.savetxt(pts1_fname.format('vertex_medians', t1, t0, '-vf'), ldmk1, fmt="%.8f")
    np.savetxt(pts0_fname.format('vertex_medians', t0, '-vf'), ldmk0, fmt="%.8f")


############################################################################
# - LANDMARKS PAIRING BY FLOW GRAPH OPRIMIZATION:
############################################################################
print("\n# - LANDMARKS PAIRING BY FLOW GRAPH OPRIMIZATION:")
from vplants.tissue_analysis.flow_graph import flow_graph_from_coordinates
from vplants.tissue_analysis.flow_graph import labels_pairing_from_flow_graph

atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
atlas_reg_path = atlas_path + 'registered_sequence/'
t0 = 0
t1 = 10

# - File names and locations for landmarks:
ldmk_fname = atlas_reg_path + 'all_landmarks-t{}vf_t{}.pkl'  # t_n, t_n+1
ldmk_fname = ldmk_fname.format(t0, t1)
pkl_file = open(ldmk_fname, 'rb')
pts0, pts1 = pickle.load(pkl_file)
pkl_file.close()

n_pts = len(pts0)

G = flow_graph_from_coordinates(pts1, pts0, n_edges=100)
lpairs = labels_pairing_from_flow_graph(G)
# draw_flow_graph(G, font_size=10)

common_pts0, common_pts1 = {}, {}
for n, (p1, p0) in enumerate(lpairs):
    # Filter to keep L1 points only:
    if 1 not in p0 or 1 not in p1:
        continue
    # Remove points not associated:
    if p1 != 'fake_x' and p0 != 'fake_y':
        common_pts0[n] = pts0[p0]
        common_pts1[n] = pts1[p1]

x0, y0, z0 = np.array(common_pts0.values()).T
x1, y1, z1 = np.array(common_pts1.values()).T
mlab.figure()
mlab.points3d(x0, y0, z0, common_pts0.keys(), mode="sphere",
              scale_mode='none', scale_factor=0.2, colormap='prism')
mlab.points3d(x1, y1, z1, common_pts1.keys(), mode="cube",
              scale_mode='none', scale_factor=0.2, colormap='prism')
mlab.quiver3d(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, line_width=1.5,
              mode="2darrow", scale_factor=1)
mlab.show()


################################################################################
# - Compute deformation estimators:
################################################################################
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import svd
from timagetk.io import imread
from timagetk.visu.util import color_map
from timagetk.components import TissueImage
from vplants.tissue_analysis.array_tools import affine_deformation_tensor
from vplants.tissue_analysis.array_tools import fractional_anisotropy_eigenval

from vplants.tissue_analysis.lineage import Lineage
from vplants.tissue_analysis.tissue_analysis import LabelProperty3D

atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
atlas_reg_path = atlas_path + 'registered_sequence/'
atlas_time_steps = [0, 10, 40, 96, 120, 132]
seg_fname = 't{}_segmented{}.{}'
ext = "tif"

for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    print("\nComputing cells deformation between t{} & t{}".format(t0, t1))
    # - File names and locations for landmarks:
    ldmk_fname = atlas_reg_path + 'all_landmarks-t{}_t{}.pkl'  # t_n, t_n+1
    ldmk_fname = ldmk_fname.format(t0, t1)

    pkl_file = open(ldmk_fname, 'rb')
    pts0, pts1 = pickle.load(pkl_file)
    pkl_file.close()

    ldmk = {}
    missing_key = []
    for k, coord0 in pts0.items():
        try:
            coord1 = pts1[k]
        except:
            continue
        if isinstance(k, int):
            if k == 0 or k == 1:
                continue
            if k in ldmk:
                ldmk[k].append([coord0, coord1])
            else:
                ldmk[k] = [[coord0, coord1]]
        else:
            for kk in k:
                if kk == 0 or kk == 1:
                    continue
                if kk in ldmk:
                    ldmk[kk].append([coord0, coord1])
                else:
                    ldmk[kk] = [[coord0, coord1]]

    # - Checking for keys defined @ T_n and not @ T_n+1 or the other way around:
    missing_t0_keys = list(set(pts1.keys()) - set(pts0.keys()))
    missing_t1_keys = list(set(pts0.keys()) - set(pts1.keys()))
    print("Missing keys @ T_n: {}".format(missing_t0_keys))
    print("Missing keys @ T_n+1: {}".format(missing_t1_keys))

    # - Counting number of landmarks per cell:
    n_wall, n_edge, n_vertex = {}, {}, {}
    for k, coord0 in pts0.items():
        if isinstance(k, int):
            n_wall[k] = 0
            n_edge[k] = 0
            n_vertex[k] = 0
        elif len(k) == 2:
            for kk in k:
                if kk != 0 and kk != 1:
                    n_wall[kk]+=1
        elif len(k) == 3:
            for kk in k:
                if kk != 0 and kk != 1:
                    n_edge[kk]+=1
        elif len(k) == 4:
            for kk in k:
                if kk != 0 and kk != 1:
                    n_vertex[kk]+=1
        else:
            pass

    # - Checking for "undefined landmarks" (ie. with 0 as neighbors):
    undef_wall, undef_edge, undef_vertex = {}, {}, {}
    for k, coord0 in pts0.items():
        if isinstance(k, int):
            undef_wall[k] = 0
            undef_edge[k] = 0
            undef_vertex[k] = 0
        elif len(k) == 2 and 0 in k:
            for kk in k:
                if kk != 0 and kk != 1:
                    undef_wall[kk]+=1
        elif len(k) == 3 and 0 in k:
            for kk in k:
                if kk != 0 and kk != 1:
                    undef_edge[kk]+=1
        elif len(k) == 4 and 0 in k:
            for kk in k:
                if kk != 0 and kk != 1:
                    undef_vertex[kk]+=1
        else:
            pass

    # - Compute the landmarks de
    ldmk_score = {}
    for k, coord0 in pts0.items():
        if isinstance(k, int):
            n_ldmk = n_wall[k] + n_edge[k] + n_vertex[k]
            n_undef_ldmk = undef_wall[k] + undef_edge[k] + undef_vertex[k]
            ldmk_score[k] = (1 - (n_ldmk - n_undef_ldmk)/n_ldmk) * 100

    # - Deformation estimators:
    def_tensor = {}
    stretch_tensors, r2_score = {}, {}
    strain_rate = {}
    strain_rates_anisotropy = {}
    volumetric_strain_rates = {}
    DeltaT = t1 - t0
    for k, arr in ldmk.items():
        arr = np.array(arr)
        ldmk_0 = arr[:, 0]
        ldmk_1 = arr[:, 1]
        def_tensor[k], r2_score[k] = affine_deformation_tensor(ldmk_0, ldmk_1)
        stretch_tensors[k] = svd(def_tensor[k])
        stretch_ratios = stretch_tensors[k][1]
        strain_rate[k] = [np.log(dk)/float(DeltaT) for dk in stretch_ratios]
        strain_rates_anisotropy[k] = fractional_anisotropy_eigenval(strain_rate[k])
        volumetric_strain_rates[k] = sum(strain_rate[k])

    # -- EXTRACT VOLUMES & COMPUTE GROWTH RATES:
    # - Defines segmented image filenames:
    t0_seg = atlas_reg_path + seg_fname.format(t0, '', ext)
    t1_seg = atlas_reg_path + seg_fname.format(t1, '', ext)

    # -- Lineage files:
    lin_file = atlas_path + 'lineages/' + 'lineage_{}h_to_{}h.txt'.format(t0, t1)

    # - Load images:
    print("\n# - READING INPUT IMAGES:")
    # Load t_n & t_n+1 SEGMENTED images:
    print("Reading file: '{}'".format(t0_seg))
    sim0 = imread(t0_seg)
    print("Reading file: '{}'".format(t1_seg))
    sim1 = imread(t1_seg)

    print("\n# - LOADING LINEAGE:")
    lin = Lineage(lin_file, lineage_fmt='marsalt')
    print("Got {} ancestors!".format(len(lin.ancestors_list())))

    print("\n# - INITIALIZING TissueImage :")
    # - Create TissueImage:
    sim0 = TissueImage(sim0, background=1, no_label_id=0, voxelsize=sim0.voxelsize)
    sim1 = TissueImage(sim1, background=1, no_label_id=0, voxelsize=sim1.voxelsize)

    print("\n# - RELABELLING T_n SEGMENTATION:")
    # -- Keep only lineaged cells @ T_n:
    sim0 = sim0.get_image_with_labels(lin.ancestors_list()+[1])  # also keep the background
    sim0 = TissueImage(sim0, background=1, no_label_id=0, voxelsize=sim0.voxelsize)

    print("\n# - RELABELLING T_n+1 SEGMENTATION:")
    # -- Relabel T_n+1 image with T_n lineaged labels:
    mapping = {desc: anc for anc, desc_list in lin.lineage_dict.items() for desc in desc_list}
    mapping.update({1: 1})  # add background to mapping to avoid loosing it with `clear_unmapped=True`
    sim1 = relabel_from_mapping(sim1, mapping, clear_unmapped=True, verbose=True)

    print("\n# - INITIALIZING TissueAnalysis OBJECTS:")
    tissue0 = LabelProperty3D(sim0, background=1)
    tissue1 = LabelProperty3D(sim1, background=1)

    print("\n# - EXTRACTING POINTS OF INTEREST:")
    # - Extract Points Of Interest:
    pts0, pts1 = {}, {}

    # - Add cell barycenters as POI:
    vol0 = tissue0.volume(real=True)
    vol1 = tissue1.volume(real=True)

    growth_rate = {k: (vol1[k] - vol0[k]) / vol0[k] * 1/DeltaT for k in vol0.keys()}
    log_growth_rate = {k: np.log(vol1[k] / vol0[k]) * 1/DeltaT for k in vol0.keys()}

    df = pd.DataFrame([vol0, vol1, strain_rate, strain_rates_anisotropy, volumetric_strain_rates, ldmk_score, growth_rate, log_growth_rate], columns=['volume mother', 'volume daugther', 'strain rates', 'strain rates anisotropy', 'volumetric strain rate', 'percentage of undefined landmarks', 'growth_rate', 'log growth_rate'])
    df.to_csv(atlas_reg_path+"deformations_t{}-t{}.csv".format(t0, t1), index_label='labels')

    big_dict = {k: [strain_rate[k], strain_rates_anisotropy[k], volumetric_strain_rates[k], ldmk_score[k], growth_rate[k], log_growth_rate[k]] for k in ldmk.keys()}
    df = pd.DataFrame.from_dict(big_dict, orient='index', columns=['strain rates', 'strain rates anisotropy', 'volumetric strain rate', 'percentage of undefined landmarks', 'growth_rate', 'log growth_rate'])
    df.to_csv(atlas_reg_path+"deformations_t{}-t{}.csv".format(t0, t1), index_label='labels')

    data_fname = atlas_reg_path + 'growth_data-t{}_t{}.pkl'  # t_n, t_n+1
    pkl_file = open(data_fname.format(t0, t1), 'wb')
    pickle.dump([vol0, vol1, strain_rate, strain_rates_anisotropy, volumetric_strain_rates, ldmk_score, growth_rate, log_growth_rate], pkl_file)
    pkl_file.close()


############################################################################
# - Correlate 'volumetric strain rate' & 'growth rate'
############################################################################
import pickle
import numpy as np
import matplotlib.pyplot as plt

def correlate(x, y, color, title, x_label, y_label, cb_title, fig_name=''):
    fig, axe = plt.subplots(nrows=1, figsize=(10.24, 10.24), dpi=100)
    mini = np.min(np.array([x, y]))
    maxi = np.max(np.array([x, y]))
    axe.plot([mini, maxi], [mini, maxi], ls=':', color='k')
    axe.axvline(0, ls='--', color='r')
    axe.axhline(0, ls='--', color='r')
    sc = axe.scatter(x, y, c=color, s=8, cmap='viridis')
    sc.set_clim(0, 100)
    axe.set_title(title)
    axe.set_xlabel(x_label)
    axe.set_ylabel(y_label)
    axe.set_aspect('equal')
    # -- Add colorbar:
    cb = plt.colorbar(sc)
    cb.set_label('percentage of undefined landmarks')
    if fig_name != '':
        plt.savefig(fig_name)
        plt.close()


atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
atlas_reg_path = atlas_path + 'registered_sequence/'
data_fname = atlas_reg_path + 'growth_data-t{}_t{}.pkl'  # t_n, t_n+1
atlas_time_steps = [0, 10, 40, 96, 120, 132]
for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    pkl_file = open(data_fname.format(t0, t1), 'rb')
    vol0, vol1, strain_rate, strain_rates_anisotropy, volumetric_strain_rates, ldmk_score, growth_rate, log_growth_rate = pickle.load(pkl_file)
    pkl_file.close()
    arr = np.array([[log_growth_rate[k], volumetric_strain_rates[k], ldmk_score[k]] for k in vol0.keys()])
    correlate(arr[:, 0], arr[:, 1], arr[:, 2], "t{}-t{}".format(t0, t1), x_label="log growth rate", y_label="volumetric strain rate", cb_title='percentage of undefined landmarks', fig_name=atlas_reg_path+"growth_correlation-t{}-t{}.png".format(t0, t1))


############################################################################
# - Spatialize 'volumetric strain rate'
############################################################################
import pickle
import numpy as np
from mayavi import mlab

def spatial_scalars(pts0, scalars, cb_title=''):
    n_pts = len(pts0)
    x0, y0, z0 = pts0.T
    mlab.figure()
    mlab.points3d(x0, y0, z0, scalars, mode="sphere",
                  scale_mode='none', scale_factor=2, colormap='viridis')
    mlab.colorbar(title=cb_title, orientation='vertical')
    mlab.show()


atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
atlas_reg_path = atlas_path + 'registered_sequence/'
data_fname = atlas_reg_path + 'growth_data-t{}_t{}.pkl'  # t_n, t_n+1
ldmk_fname = atlas_reg_path + 'all_landmarks-t{}_t{}.pkl'  # t_n, t_n+1
atlas_time_steps = [0, 10, 40, 96, 120, 132]
for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    # - Load landmark coordinate dictionaries:
    pkl_file = open(ldmk_fname.format(t0, t1), 'rb')
    pts0, pts1 = pickle.load(pkl_file)
    pkl_file.close()
    # - Load growth data dictionaries:
    pkl_file = open(data_fname.format(t0, t1), 'rb')
    vol0, vol1, strain_rate, strain_rates_anisotropy, volumetric_strain_rates, ldmk_score, growth_rate, log_growth_rate = pickle.load(pkl_file)
    pkl_file.close()
    # - 3D view of "volume"
    # arr = np.array([[pts0[k][0], pts0[k][1], pts0[k][2], vol0[k]] for k in vol0.keys()])
    # spatial_scalars(arr[:, 0:3], arr[:, 3])
    # - 3D view of "volumetric_strain_rates"
    # arr = np.array([[pts0[k][0], pts0[k][1], pts0[k][2], volumetric_strain_rates[k]] for k in vol0.keys()])
    # spatial_scalars(arr[:, 0:3], arr[:, 3])
    arr = np.array([[pts0[k][0], pts0[k][1], pts0[k][2], growth_rate[k]] for k in vol0.keys()])
    spatial_scalars(arr[:, 0:3], arr[:, 3])


############################################################################
# - Spatialaze strain rates:
############################################################################
import pickle
import numpy as np
from mayavi import mlab

def spatial_tensors(pts0, tensors, cb_title=''):
    n_pts = len(pts0)
    x0, y0, z0 = pts0.T
    u, v, w = tensors.T

    mlab.figure()
    mlab.quiver3d(x0, y0, z0, u, v, w, mode="sphere",
                  scale_mode='vector', scale_factor=10, colormap='viridis')
    mlab.colorbar(title=cb_title, orientation='vertical')
    mlab.show()


atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
atlas_reg_path = atlas_path + 'registered_sequence/'
data_fname = atlas_reg_path + 'growth_data-t{}_t{}.pkl'  # t_n, t_n+1
ldmk_fname = atlas_reg_path + 'all_landmarks-t{}_t{}.pkl'  # t_n, t_n+1
atlas_time_steps = [0, 10, 40, 96, 120, 132]
for tp, ts in enumerate(atlas_time_steps[:-1]):
    t0 = ts
    t1 = atlas_time_steps[tp+1]
    # - Load landmark coordinate dictionaries:
    pkl_file = open(ldmk_fname.format(t0, t1), 'rb')
    pts0, pts1 = pickle.load(pkl_file)
    pkl_file.close()
    # - Load growth data dictionaries:
    pkl_file = open(data_fname.format(t0, t1), 'rb')
    vol0, vol1, strain_rate, strain_rates_anisotropy, volumetric_strain_rates, ldmk_score, growth_rate, log_growth_rate = pickle.load(pkl_file)
    pkl_file.close()
    # - 3D view of "volumetric_strain_rates"
    arr = np.array([[pts0[k][0], pts0[k][1], pts0[k][2], strain_rate[k][0], strain_rate[k][1], strain_rate[k][2]] for k in vol0.keys()])
    spatial_tensors(arr[:, 0:3], arr[:, 3:6])
