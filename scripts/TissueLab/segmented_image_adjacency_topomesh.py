import numpy as np

from openalea.image.serial.all import imread
from openalea.cellcomplex.property_topomesh.utils.tissue_analysis_tools import cell_vertex_extraction
from openalea.cellcomplex.property_topomesh.property_topomesh_creation import tetrahedra_topomesh
from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image

from openalea.oalab.colormap.colormap_def import load_colormaps

# filename = "/Users/gcerutti/Developpement/d-tk/gnomon-data/0hrs_plant1_seg_small.inr"
filename = "/data/Yassin/YR_01_ATLAS_iso/segmentations/t10_segmented_new_clean_2.tif"

background_adjacency = False

img = imread(filename)

if 'world' in dir():
    world.add(img,'segmented_image',colormap='glasbey',alphamap='constant',alpha=0.2)

cell_vertex = cell_vertex_extraction(img)
tetras = np.array(cell_vertex.keys())

img_graph = graph_from_image(img, spatio_temporal_properties=['barycenter','volume'], ignore_cells_at_stack_margins=True)
positions = img_graph.vertex_property('barycenter')

if background_adjacency:
    positions[1] = np.array([0,0,0],float)
else:
    tetras = tetras[tetras[:,0]!=1]

topomesh = tetrahedra_topomesh(tetras, positions)

if 'world' in dir():
    world.add(topomesh,'adjacency')
    world['adjacency_cells']['polydata_colormap'] = load_colormaps()['grey']
    world['adjacency_cells']['intensity_range'] = (-1,0)
    world['adjacency_cells']['polydata_alpha'] = 0.5
    world['adjacency']['coef_3'] = 0.5
    world['adjacency']['display_0'] = True
    world['adjacency_vertices']['polydata_colormap'] = load_colormaps()['glasbey']
    world['adjacency']['display_1'] = True
    world['adjacency_edges']['polydata_colormap'] = load_colormaps()['grey']
    world['adjacency_edges']['linewidth'] = 3
    world['adjacency']['display_2'] = True
    world['adjacency']['coef_2'] = 0.75
    world['adjacency_faces']['polydata_colormap'] = load_colormaps()['grey']
    world['adjacency_faces']['intensity_range'] = (-1,0)
