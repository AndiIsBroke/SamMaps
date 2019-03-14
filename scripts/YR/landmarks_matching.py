############################################################################
# - Sequence RIGID registration of 'YR_01':
############################################################################
from os import mkdir
from os.path import exists
from timagetk.io import imread
from timagetk.io import imsave
from timagetk.io.io_trsf import save_trsf
from timagetk.algorithms.trsf import apply_trsf
from timagetk.plugins import sequence_registration

dir_path = '/data/Yassin/YR_01/'
time_steps = [0, 10, 18, 24, 32, 40, 48, 57, 64, 72, 81, 88, 96, 104, 112, 120, 128, 132]

time_points = range(len(time_steps))
tp2ts = dict(zip(time_points, time_steps))
ts2tp = dict(zip(time_steps, time_points))

# -- Intensity images:
ext = "tif"
int_path = dir_path + 'tifs/'
int_fname = 't{}{}.{}'  # tp, ?, ext
# -- Segmented images:
seg_path = dir_path + 'segmentations/'
seg_fname = 't{}_segmented{}.{}'
# -- Registered intensity image, segmented images & transformation file path:
reg_path = dir_path + 'registered_sequence/'
trsf_name = "t{}-t{}_{}.trsf"  # t0, t1, trsf_type

try:
    mkdir(reg_path)
except OSError:
    pass

############################################################################
list_image = []
for tp in time_points:
    fname = int_fname.format(tp2ts[tp], '', ext)
    list_image.append(imread(int_path + fname))

list_trsf = sequence_registration(list_image, method='rigid',
                                  pyramid_lowest_level=1)

ref_tp = time_points[-1]
for n, trsf in enumerate(list_trsf):
    float_tp = n
    t_float, t_ref = tp2ts[float_tp], tp2ts[ref_tp]
    fname_trsf = reg_path + trsf_name.format(t_float, t_ref, 'rigid')
    # Save sequence rigid transformations:
    print("Saving:", fname_trsf)
    save_trsf(trsf, fname_trsf)
    print("Done!\n")
    # Save rigid sequence registered intensity images:
    fname_res_int = reg_path + int_fname.format(t_float, '', ext)
    print("Saving:", fname_res_int)
    list_image[n] = apply_trsf(list_image[n], trsf, template_img=list_image[ref_tp], param_str_2='-linear')
    imsave(fname_res_int, list_image[n])
    print("Done!\n")
    # Load the segmented image:
    seg_im = imread(seg_path + seg_fname.format(t_float, '', ext))
    # Save rigid sequence registered segmented images:
    fname_res_seg = reg_path + seg_fname.format(t_float, '', ext)
    print("Saving:", fname_res_seg)
    imsave(fname_res_seg, apply_trsf(seg_im, trsf, template_img=list_image[ref_tp].shape, param_str_2='-nearest'))
    print("Done!\n")


############################################################################
# - COPY RIGID TRSFs to 'YR_01_ATLAS_iso' (temporal down-sampling):
############################################################################
from shutil import copy2

atlas_path = '/data/Yassin/YR_01_ATLAS_iso/'
int_path = atlas_path + 'tifs/'
seg_path = atlas_path + 'segmentations/'
# -- Registered intensity image, segmented images & transformation file path:
atlas_reg_path = atlas_path + 'registered_sequence/'

atlas_time_teps = [10, 40, 96, 120, 132]

try:
    mkdir(atlas_reg_path)
except OSError:
    pass

t_ref = atlas_time_teps[-1]
for t_float in atlas_time_teps[:-1]:
    # - Intensity image:
    fname = int_fname.format(t_float, '', ext)
    copy(reg_path + fname, atlas_reg_path)
    # - Segmented image:
    fname = seg_fname.format(t_float, '', ext)
    copy(reg_path + fname, atlas_reg_path)
    # - Transformation:
    fname = trsf_name.format(t_float, t_ref, 'rigid')
    copy(reg_path + fname, atlas_reg_path)


############################################################################
# - EXTRACT LANDMARKS and COMPUTE EPIDERMAL WALL DEFORMATION
############################################################################
from mayavi import mlab
import numpy as np
import networkx as nx
from os.path import exists
from timagetk.io import imread
from timagetk.io import imsave
from timagetk.components import TissueImage3D
from timagetk.plugins import registration
from timagetk.algorithms.trsf import apply_trsf
from timagetk.algorithms.resample import isometric_resampling
from timagetk.io.io_trsf import read_trsf
from timagetk.io.io_trsf import save_trsf

t0 = 10
t1 = 40
ext = 'tif'
reg_type = 'rigid'

# WARNING: Don't forget to remove the registered segmented image if you need too !!!

t0_int = atlas_reg_path + int_fname.format(t0, '', ext)
t1_int = atlas_reg_path + int_fname.format(t1, '', ext)

t0_seg = atlas_reg_path + seg_fname.format(t0, '', ext)
t1_seg = atlas_reg_path + seg_fname.format(t1, '', ext)
# -- Lineage files:
lin_file = atlas_path + 'lineages/' + 'lineage_{}h_to_{}h.txt'.format(t0, t1)

# - File names and locations for landmarks:
pts_ref_fname = atlas_reg_path + 'landmarks-t{}_relab_t{}.txt'.format(t1, t0)
pts_flo_fname = atlas_reg_path + 'landmarks-t{}.txt'.format(t0)

# - Load images:
print("\n# - READING INPUT IMAGES:")
print("Reading file: '{}'".format(t0_int))
iim0 = imread(t0_int)
print("Reading file: '{}'".format(t1_int))
iim1 = imread(t1_int)
print("Reading file: '{}'".format(t0_seg))
sim0 = imread(t0_seg)
print("Reading file: '{}'".format(t1_seg))
sim1 = imread(t1_seg)

# - Performs isometric resampling:
print("\n# - ISOMETRIC RESAMPLING:")
iim0 = isometric_resampling(iim0, method=iim0.voxelsize[0])
iim1 = isometric_resampling(iim1, method=iim1.voxelsize[0])
sim0 = isometric_resampling(sim0, method=sim0.voxelsize[0])
sim1 = isometric_resampling(sim1, method=sim1.voxelsize[0])
# iim0 = isometric_resampling(iim0, method=0.4)
# iim1 = isometric_resampling(iim1, method=0.4)
# sim0 = isometric_resampling(sim0, method=0.4)
# sim1 = isometric_resampling(sim1, method=0.4)

assert iim0.shape == sim0.shape
assert iim1.shape == sim1.shape
assert np.all(np.isclose(iim0.voxelsize, sim0.voxelsize, 6))
assert np.all(np.isclose(iim1.voxelsize, sim1.voxelsize, 6))

print("\n# - REGISTRATION:")
# - Registration:
# -- Filenames:
res_trsf_name = atlas_reg_path + trsf_name.format(t0, t1, reg_type)
t0_seg_reg = seg_fname.format(t0, '_{}'.format(reg_type), ext)
# -- Compute registration and save trsf:
if not exists(res_trsf_name):
    print("Computing {} transformation...".format(reg_type))
    res_trsf, _ = registration(iim0, iim1, method=reg_type, verbose=True)
    save_trsf(res_trsf, res_trsf_name)
else:
    print("Found existing transformation: {}.".format(res_trsf_name))
    res_trsf = read_trsf(res_trsf_name)

print("\n# - LOADING LINEAGE:")
# - Load lineage:
from vplants.tissue_analysis.graphs.lineage import Lineage
lin_dict = Lineage(lin_file, lineage_fmt='marsalt').lineage_dict

print("\n# - INITIALIZING TissueImage3D :")
# - Create TissueImage3D:
sim0 = TissueImage3D(sim0, background=1, no_label_id=0,
                     voxelsize=sim0.voxelsize)
sim1 = TissueImage3D(sim1, background=1, no_label_id=0,
                     voxelsize=sim1.voxelsize)

print("\n# - RELABELLING SEGMENTATION:")
# -- Relabel descendant image to ancestors labels:
mapping = {desc: anc for anc, desc_list in lin_dict.items() for desc in
           desc_list}
sim1.relabel_from_mapping(mapping, clear_unmapped=False, verbose=True)
sim1 = TissueImage3D(sim1, background=1, no_label_id=0)

from vplants.tissue_analysis.features.tissue_analysis import TissueAnalysis

# - Make TissueAnalysis:
tissue0 = TissueAnalysis(sim0, background=1)
tissue1 = TissueAnalysis(sim1, background=1)

print("\n# - EXTRACTING POINTS OF INTEREST:")
# - Extract Points Of Interest:
pts0, pts1 = {}, {}
# - Add wall medians as POI:
wm0 = tissue0.wall.get_medians()
wm1 = tissue1.wall.get_medians()
w_ids = set(wm0.keys()) & set(wm1.keys())
print(
    "Found {} common ids between reference (n={}) and floating (n={}) wall median!".format(
        len(w_ids), len(wm0), len(wm1)))
for k in w_ids:
    pts0.update({k: v for k, v in wm0.items() if k in w_ids})
    pts1.update({k: v for k, v in wm1.items() if k in w_ids})
# - Add edge medians as POI:
em0 = tissue0.edge.get_medians()
em1 = tissue1.edge.get_medians()
e_ids = set(em0.keys()) & set(em1.keys())
print(
    "Found {} common ids between reference (n={}) and floating (n={}) edge median!".format(
        len(e_ids), len(em0), len(em1)))
for k in e_ids:
    pts0.update({k: v for k, v in em0.items() if k in e_ids})
    pts1.update({k: v for k, v in em1.items() if k in e_ids})
# - Add vertex medians as POI:
vm0 = tissue0.vertex.get_medians()
vm1 = tissue1.vertex.get_medians()
v_ids = set(vm0.keys()) & set(vm1.keys())
print(
    "Found {} common ids between reference (n={}) and floating (n={}) vertex median!".format(
        len(v_ids), len(vm0), len(vm1)))
for k in v_ids:
    pts0.update({k: v for k, v in vm0.items() if k in v_ids})
    pts1.update({k: v for k, v in vm1.items() if k in v_ids})

common_ids = set(pts1.keys()) & set(pts0.keys())
# - Save the points:
np.savetxt(pts_ref_fname, [pts1[k] for k in common_ids], fmt="%.8f")
np.savetxt(pts_flo_fname, [pts0[k] for k in common_ids], fmt="%.8f")

# if init_registration and not exists(init_trsf_fname):
#     # -- Compute the VECTORFIELD tranformation with pointmatching:
#     vf_trsf_fname = int_fname.format(t0, '_vectorfield_on_t{}'.format(t1),
#                                      'trsf')
#     from vplants.tissue_analysis.graphs.temporal_analysis import \
#         pointmatching_cmd
#
#     pointmatching_cmd(pts_ref_fname, pts_flo_fname, trsf_type="vectorfield",
#                       trsf_fname=vf_trsf_fname, template_shape=sim1.shape,
#                       template_voxelsize=sim1.voxelsize)

print("\n# - LANDMARKS DEFINITION BY FLOW GRAPH PAIRING:")
from vplants.tissue_analysis.graphs.temporal_analysis import \
    flow_graph_from_coordinates
from vplants.tissue_analysis.graphs.temporal_analysis import \
    labels_pairing_from_flow_graph

G = flow_graph_from_coordinates(pts1, pts0)
lpairs = labels_pairing_from_flow_graph(G)
# draw_flow_graph(G, font_size=10)

common_pts0, common_pts1 = {}, {}
for n, (p1, p0) in enumerate(lpairs):
    # Filter for L1 points
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
              scale_mode='none', scale_factor=2, colormap='prism')
mlab.points3d(x1, y1, z1, common_pts1.keys(), mode="cube",
              scale_mode='none', scale_factor=2, colormap='prism')
mlab.quiver3d(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, line_width=1.5,
              mode="2darrow", scale_factor=1)
mlab.show()

# from openalea.mesh.property_topomesh_creation import vertex_topomesh
#
# x_vtx = vertex_topomesh(common_pts1.values())
# world.add(x_vtx, 't{} paired medians'.format(t1))
# y_vtx = vertex_topomesh(common_pts0.values())
# world.add(y_vtx, 't{} paired medians'.format(t0))
#
# # - Compute a transformation matrix from ref. to float. to initialise registration:
# from timagetk.wrapping.bal_trsf import BalTransformation
# from timagetk.algorithms.trsf import allocate_c_bal_matrix
# from timagetk.algorithms.reconstruction import pts2transfo
#
# res_trsf = pts2transfo(pts1, pts0)
# aff_trsf = create_trsf(param_str_2='-identity',
#                        trsf_type=BalTransformation.AFFINE_3D,
#                        trsf_unit=BalTransformation.VOXEL_UNIT)
# allocate_c_bal_matrix(aff_trsf.mat.c_struct, res_trsf)
#
# # sim0 = apply_trsf(sim0, trsf)
#
# import vplants.tissue_analysis.graphs.temporal_analysis
#
# reload(vplants.tissue_analysis.graphs.temporal_analysis)
# from vplants.tissue_analysis.graphs.temporal_analysis import \
#     flow_graph_from_coordinates
#
# G, target_id = flow_graph_from_coordinates(pts0, pts1)
#
# from timagetk.algorithms.reconstruction import pts2transfo
#
# res_trsf = pts2transfo(pts0, pts1)
# trsf = create_trsf(param_str_2='-identity',
#                    trsf_type=BalTransformation.AFFINE_3D,
#                    trsf_unit=BalTransformation.REAL_UNIT)
# allocate_c_bal_matrix(trsf.mat.c_struct, res_trsf)
#
# if init_registration and not exists(init_trsf_fname):
#     pts_ref_fname = int_path + 'landmarks-t{}_relab_t{}.txt'.format(t1, t0)
#     pts_flo_fname = int_path + 'landmarks-t{}.txt'.format(t0)
#     if not exists(pts_ref_fname) and not exists(pts_flo_fname):
#         sim0 = TissueImage3D(sim0, background=1, no_label_id=0,
#                              voxelsize=sim0.voxelsize)
#         sim1 = TissueImage3D(sim1, background=1, no_label_id=0,
#                              voxelsize=sim1.voxelsize)
#         # -- Relabel descendant image to ancestors labels:
#         mapping = {desc: anc for anc, desc_list in lin_dict.items() for desc
#                    in
#                    desc_list}
#         sim1.relabel_from_mapping(mapping, clear_unmapped=False,
#                                   verbose=True)
#         sim1 = TissueImage3D(sim1, background=1, no_label_id=0)
#         # - Make SpatialImageAnalysis:
#         spia0 = SpatialImageAnalysis(sim0, background=1)
#         spia1 = SpatialImageAnalysis(sim1, background=1)
#         # - Get the list of common cell ids after relabelling:
#         common_cell_ids = list(set(sim0.cells()) & set(sim1.cells()))
#         # - Get the cell barycenters (real units):
#         pts0 = spia0.center_of_mass(common_cell_ids, real=True)
#         pts1 = spia1.center_of_mass(common_cell_ids, real=True)
#         # - Save those landmarks coordinates:
#         np.savetxt(pts_ref_fname, pts1.values(), fmt="%.8f")
#         np.savetxt(pts_flo_fname, pts0.values(), fmt="%.8f")
#
#     # - Compute the initial VECTORFIELD tranformation using MORPHEME tools:
#     cmd = ""
#     # -- Compute the initial AFFINE tranformation with pointmatching:
#     # Note: registration of 'floating' landmarks onto 'reference' landmarks
#     # returns a tranformation oriented from 'reference' toward 'floating'!
#     aff_trsf_fname = int_fname.format(t0, '_affine_on_t{}'.format(t1),
#                                       'trsf')
#     cmd += pointmatching_cmd(pts_ref_fname, pts_flo_fname, 'affine',
#                              aff_trsf_fname, sim1.shape, sim1.voxelsize)
#
#     # -- Inverting the affine tranformation:
#     # (then oriented from 'floating' toward 'reference')
#     inv_aff_trsf_fname = int_fname.format(t0,
#                                           '_inv_affine_on_t{}'.format(t1),
#                                           'trsf')
#     cmd += invTrsf_cmd(aff_trsf_fname, inv_aff_trsf_fname)
#
#     # -- Applying this (inverted) affine transformation to 'floating' landmarks:
#     aff_pts_flo_fname = int_path + 'landmarks-t{}_affine.txt'.format(t0)
#     cmd += applyTrsfToPoints_cmd(pts_flo_fname, aff_pts_flo_fname,
#                                  inv_aff_trsf_fname)
#
#     # -- Compute the VECTORFIELD tranformation with pointmatching:
#     vf_trsf_fname = int_fname.format(t0, '_vectorfield_on_t{}'.format(t1),
#                                      'trsf')
#     cmd += pointmatching_cmd(pts_ref_fname, aff_pts_flo_fname,
#                              'vectorfield',
#                              vf_trsf_fname, sim1.shape, sim1.voxelsize)
#
#     # -- Compose 'affine' & 'non-linear' transformations to initialise blockmatching:
#     cmd += composeTrsf_cmd([aff_trsf_fname, vf_trsf_fname], init_trsf_fname)
#     print(cmd)
#
# # init_trsf = read_trsf(init_trsf_fname, template_img=iim1)
# init_trsf = read_trsf(init_trsf_fname)
# iim0_res = apply_trsf(iim0, trsf=init_trsf, template_img=iim1,
#                       param_str_2='-linear')
#
# if not exists(t0_seg_vf):
#     # - Make non-linear registration from intensity images:
#     trsf_res, iim0_res = registration(iim0, iim1, init_trsf=init_trsf,
#                                       method="deformable_registration",
#                                       verbose=True, time=True)
#     # - Save the transformation matrix:
#     # trsf_res.write(int_fname.format(t0, '_vf', 'trsf'))
#     # TODO: Fix "core dumped"...
#
#     # - Save the registered intensity image:
#     imsave(t0_int_vf, iim0_res)
#     # TODO: apply trsf(t_n/t_n+1) to the set of points from original t_n image... otherwise we create points around the t_n structure since the margins of the object do not touch the margins of the stack after being changed of "template", often bigger!
#
#     # - Apply non-linear registration to corresponding labelled image:
#     sim0_res = apply_trsf(sim0, trsf=trsf_res, template_img=sim1,
#                           param_str_2='-nearest')
#     # sim0_res[sim0_res == 0] = 1  # replace 0 by background id
#     imsave(t0_seg_vf, sim0_res)
# else:
#     # - Load non-linear registered image:
#     iim0_res = imread(t0_int_vf)
#     sim0_res = imread(t0_seg_vf)
#
# # - Display if in tissuelab:
# try:
#     world.add(t0_int_vf, 't0_int_vf')
# except:
#     pass
# # - Display if in tissuelab:
# try:
#     world.add(t0_seg_vf, 't0_seg_vf')
# except:
#     pass
#
# # - Create TissueImage instances:
# sim0_res = TissueImage3D(sim0_res, voxelsize=sim0_res.voxelsize,
#                          background=1, no_label_id=0)
# sim1 = TissueImage3D(sim1, voxelsize=sim1.voxelsize, background=1,
#                      no_label_id=0)
#
# # - Relabel descendant image to ancestors labels:
# mapping = {desc: anc for anc, desc_list in lin_dict.items() for desc in
#            desc_list}
# sim1 = sim1.relabel_from_mapping(mapping, clear_unmapped=False,
#                                  verbose=True)
#
# # - Make SpatialImageAnalysis:
# spia0 = SpatialImageAnalysis(sim0_res, background=1)
# spia1 = SpatialImageAnalysis(sim1, background=1)
#
# # - Compute all topological elements:
# spia0.compute_all_topological_elements(verbose=True)
# spia1.compute_all_topological_elements(verbose=True)
#
# # - Get the cell wall medians:
# wm_Xi = spia0.get_cell_wall_medians()
# wm_Yi = spia1.get_cell_wall_medians()
#
# for k, v in wm_Xi.items():
#     if 0 in k:
#         wm_Xi.pop(k)
#
# # - Define set of coordinates:
# x, y = {}, {}
# x.update(wm_Xi)
# y.update(wm_Yi)
#
# # - Display segmented image:
# world.add(spia0.image.get_image_with_labels([cell]),
#           't{} segmented image'.format(t0), colormap='glasbey',
#           alphamap='constant')
# world.add(spia1.image.get_image_with_labels([cell]),
#           't{} segmented image'.format(t1), colormap='glasbey',
#           alphamap='constant')
# # - Display intensity image:
# world.add(iim0_res, 't{} intensity image'.format(t0))
# world.add(iim1, 't{} intensity image'.format(t1))
# # - Display detected cell wall medians:
# from openalea.mesh.property_topomesh_creation import vertex_topomesh
#
# x_vtx = vertex_topomesh(x.values())
# world.add(x_vtx, 't{} cell wall medians'.format(t0))
# y_vtx = vertex_topomesh(y.values())
# world.add(y_vtx, 't{} cell wall medians'.format(t1))
#
# import vplants.tissue_analysis.graphs.temporal_analysis
#
# reload(vplants.tissue_analysis.graphs.temporal_analysis)
# from vplants.tissue_analysis.graphs.temporal_analysis import \
#     flow_graph_from_coordinates
#
# G, target_id = flow_graph_from_coordinates(x, y)
#
# from vplants.tissue_analysis.graphs.temporal_analysis import draw_flow_graph
#
# draw_flow_graph(G, font_size=10)
#
# mfmc = nx.algorithms.flow.max_flow_min_cost(G, 1, target_id)

# x.update(spia0.get_cell_edge_medians())
# y.update(spia1.get_cell_edge_medians())
# x.update(spia0.get_cell_vertex_medians())
# y.update(spia1.get_cell_vertex_medians())
#
#
# import matplotlib.pyplot as plt
#
# plt.subplot(111)
# import networkx as nx
# nx.draw(G, with_labels=True, font_weight='bold')
#
# # Coordinates matching (Xi, Yi):
#
# match = matching_nodes()
#
# import numpy as np
# from timagetk.io import imread
# from vplants.tissue_analysis.spatial_image_analysis import \
#     SpatialImageAnalysis
#
# # Load images:
# sim0 = imread(
#     '/data/TiFEx_paper/p194/Segmentations/p194-t1_beta0_alf0_lam10_dt100LinTh21-20_lsm_contour_ftseedsasf3_ftwatgaussian3_hmin3dist-2_wat01_SegExp_corr5.inr.gz')
# # Make SpatialImageAnalysis:
# spia0 = SpatialImageAnalysis(sim0, background=1)
# # Compute all topological elements:
# import cProfile
#
# cp = cProfile.Profile()
# cp.enable()
# spia0.compute_all_topological_elements(verbose=True)
# cp.disable()
# cp.print_stats()
