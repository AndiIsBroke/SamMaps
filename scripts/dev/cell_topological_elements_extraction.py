# -*- python -*-
# -*- coding: utf-8 -*-

import numpy as np

from scipy.cluster.vq import vq

from timagetk.io import imread
from vplants.tissue_analysis.spatial_image_analysis import cell_topological_elements_extraction
from openalea.cellcomplex.property_topomesh.utils.array_tools import array_unique

# a = np.array([[2, 2, 2, 3, 3, 3, 3, 3],
# [2, 2, 2, 3, 3, 3, 3, 3],
# [2, 2, 2, 2, 3, 3, 3, 3],
# [2, 2, 2, 2, 3, 3, 3, 3],
# [2, 2, 2, 2, 3, 3, 3, 3],
# [4, 4, 4, 4, 4, 4, 4, 4],
# [4, 4, 4, 4, 4, 4, 4, 4],
# [4, 4, 4, 4, 4, 4, 4, 4]])
# bkgd_im = np.ones_like(a)
# # Create an image by adding a background and repeat the previous array 6 times as a Z-axis:
# im = np.array([bkgd_im, a, a, a, a, a, a]).transpose(1, 2, 0)
# im.shape
#
# # Extract topological elements coordinates:
# elem = cell_topological_elements_extraction(im)
# # Get the cell-vertex coordinates between labels 1, 2, 3 and 4
# elem[0]
# # Get the wall-edge voxel coordinates between labels 1, 2 and 3:
# elem[1][(1, 2, 3)]
# # Get the wall voxel coordinates between labels 1 and 4:
# elem[2][(1, 4)]

# from openalea.cellcomplex.property_topomesh.property_topomesh_creation import vertex_topomesh
# topomesh = vertex_topomesh(elem[0])
# world.add(topomesh, 'detected_cell_vertex', colormap='Reds')

im_fname = '/data/Meristems/Carlos/PIN_maps/nuclei_images/qDII-PIN1-CLV3-PI-LD_E37_171113_sam07_t14/qDII-PIN1-CLV3-PI-LD_E37_171113_sam07_t14_PI_segmented.inr.gz'
im = imread(im_fname)

elem = cell_topological_elements_extraction(im[300:-300, 300:-300, :])
wall_coords = elem[2]
wall_edge_coords = elem[1]
wall_vertex_coords = elem[0]

import vplants.tissue_analysis.spatial_image_analysis
reload(vplants.tissue_analysis.spatial_image_analysis)
from vplants.tissue_analysis.spatial_image_analysis import find_pointset_median
from vplants.tissue_analysis.spatial_image_analysis import find_geometric_median

wmv = find_pointset_median(wall_coords, labels2exclude=None, return_id=False)
wemv = find_pointset_median(wall_edge_coords, labels2exclude=None, return_id=False)
wvmv = find_pointset_median(wall_vertex_coords, labels2exclude=None, return_id=False)


from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
spia = SpatialImageAnalysis(im, background=1)
spia.boundingbox()
spia.compute_all_topological_elements()
spia.get_cell_edge_medians()
