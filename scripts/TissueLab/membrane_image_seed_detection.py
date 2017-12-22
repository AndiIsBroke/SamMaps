# imports
import numpy as np
from timagetk.components import imread, imsave, SpatialImage
from timagetk.plugins import linear_filtering, morphology, h_transform, region_labeling
from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.oalab.colormap.colormap_def import load_colormaps

from openalea.tissue_nukem_3d.nuclei_mesh_tools import nuclei_layer

from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image

from timagetk.algorithms import linearfilter
from timagetk.plugins import morphology

import sys
sys.path.append('/home/marie/SamMaps/scripts/TissueLab/')
from equalization import z_slice_contrast_stretch
from slice_view import slice_view

def array_unique(array,return_index=False):
  _,unique_rows = np.unique(np.ascontiguousarray(array).view(np.dtype((np.void,array.dtype.itemsize * array.shape[1]))),return_index=True)
  if return_index:
    return array[unique_rows],unique_rows
  else:
    return array[unique_rows]

def seed_image_from_points(size, voxelsize, positions, point_radius=1.0, background_label=1):
    """
    Generate a SpatialImage of a given shape with labelled spherical regions around points
    """

    seed_img = background_label*np.ones(tuple(size),np.uint16)

    size = np.array(size)
    voxelsize = np.array(voxelsize)

    for p in positions.keys():
        image_neighborhood = np.array(np.ceil(point_radius/voxelsize),int)
        neighborhood_coords = np.mgrid[-image_neighborhood[0]:image_neighborhood[0]+1,-image_neighborhood[1]:image_neighborhood[1]+1,-image_neighborhood[2]:image_neighborhood[2]+1]
        neighborhood_coords = np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,0)))) + np.array(positions[p]/voxelsize,int)
        neighborhood_coords = np.minimum(np.maximum(neighborhood_coords,np.array([0,0,0])),size-1)
        neighborhood_coords = array_unique(neighborhood_coords)

        neighborhood_distance = np.linalg.norm(neighborhood_coords*voxelsize - positions[p],axis=1)
        neighborhood_coords = neighborhood_coords[neighborhood_distance<=point_radius]
        neighborhood_coords = tuple(np.transpose(neighborhood_coords))

        seed_img[neighborhood_coords] = p

    return SpatialImage(seed_img,voxelsize=list(voxelsize))

# Files's directories
#-----------------------

# image_dirname = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/nuclei_ground_truth_images/"
# image_dirname = "/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images"
image_dirname = "/home/marie/Marie/Lti6b/2017-12-01/"

# filename = 'DR5N_6.1_151124_sam01_z0.50_t00'
# filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t05'
filenames = ['Lti6b_xy0.156_z0.8_CH0_iso','Lti6b_xy0.156_z0.32_CH0_iso','Lti6b_xy0.156_z0.156_CH0_iso','Lti6b_xy0.156_z0.32_pinH0.34_CH0_iso','Lti6b_xy0.156_z0.80_pinH0.34_CH0_iso','Lti6b_xy0.313_z0.8_zoom0.5_CH0_iso']

#reference_name = "PI"
reference_name = "Lti6b"

microscope_orientation = +1

for filename in filenames :

    image_filename = image_dirname+filename+".inr"

    img = imread(image_filename)
    size = np.array(img.shape)
    voxelsize = np.array(img.voxelsize)

    x_sh, y_sh, z_sh = img.get_shape()
    # Display orthogonal view of ORIGINAL image:
    slice_view(img, x_sh/2, y_sh/2, z_sh/2, filename[:-8]+'\n original_image', image_filename[:-4] + ".png")
    world.add(img,"reference_image",colormap="invert_grey",voxelsize=microscope_orientation*voxelsize)

    # parameters
    pc_min, pc_max = 2, 99
    std_dev = 2.0
    morpho_radius = 3
    h_min = 170

    # intensity rescaling #
    img_eq = z_slice_contrast_stretch(img, pc_min=pc_min, pc_max=pc_max)

    title = filename[:-8]+"\n Contrast stretched (z slices): {}pc.-{}pc.".format(pc_min, pc_max)
    file_name = filename[:-4]+"_z_contrast_stretch_{}-{}pc.png".format(pc_min, pc_max)
    slice_view(img_eq, x_sh/2, y_sh/2, z_sh/2, title, image_dirname + file_name)
    img_eq = SpatialImage(img_eq,voxelsize=img.voxelsize)

    smooth_img = linear_filtering(img_eq, std_dev=std_dev, method='gaussian_smoothing')

    title = filename[:-8]+"\n Contrast stretched (z-slice): {}pc.-{}pc \n + Gaussian smoothing: std_dev:{}.".format(pc_min, pc_max, std_dev)
    file_name = filename[:-4]+"_z_contrast_stretch_{}-{}pc, gaussian smoothing_std_dev_{} .png".format(pc_min, pc_max,std_dev)
    slice_view(smooth_img, x_sh/2, y_sh/2, z_sh/2, title, image_dirname + file_name)

    asf_img = morphology(smooth_img, max_radius=morpho_radius, method='co_alternate_sequential_filter')
    title = filename[:-8]+"\n Contrast stretched (z-slice): {}pc.-{}pc \n + gaussian smoothing: std_dev:{} \n + Alternate sequential filter: morpho_radius:{}.".format(pc_min, pc_max, std_dev, morpho_radius)
    file_name = filename[:-4]+"_z_contrast_stretch_{}-{}pc, gaussian smoothing_std_dev_{}, co_alternate_sequential_filter: morpho_radius:{}.png".format(pc_min, pc_max, std_dev, morpho_radius)
    slice_view(smooth_img, x_sh/2, y_sh/2, z_sh/2, title, image_dirname + file_name)

    ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
    con_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')


    img_graph = graph_from_image(con_img,background=1,spatio_temporal_properties=['barycenter'],ignore_cells_at_stack_margins=False)
    print img_graph.nb_vertices()," Seeds detected"
    seed_positions = dict([(v,img_graph.vertex_property('barycenter')[v]) for v in img_graph.vertices()])
    seed_topomesh = vertex_topomesh(seed_positions)
    #world.add(seed_topomesh,'seeds')

    cell_layer = nuclei_layer(seed_positions, size, voxelsize)
    seed_topomesh.update_wisp_property('layer', 0, cell_layer)

    topomesh_file = image_dirname + filename+"_eq_seeds_topomesh.ply"
    save_ply_property_topomesh(seed_topomesh,topomesh_file,properties_to_save=dict([(0,['Lti6b']+['layer']),(1,[]),(2,[]),(3,[])]),color_faces=False)
    df = topomesh_to_dataframe(seed_topomesh,0)
    df.to_csv(image_dirname+filename+"_signal_data.csv")

# Correction :

seed_topomesh_filename = image_dirname + filename+"_eq_seeds_topomesh.ply"
seed_topomesh = read_ply_property_topomesh(seed_topomesh_filename)
## L1 cells only
L1_cells = np.array(list(seed_topomesh.wisps(0)))[seed_topomesh.wisp_property('layer',0).values()==1]
non_L1_cells = np.array(list(seed_topomesh.wisps(0)))[seed_topomesh.wisp_property('layer',0).values()!=1]
# from copy import deepcopy
# ## L1-seed :
# L1_seed_topomesh = deepcopy(seed_topomesh)
# L1_seed_cells = np.array(list(L1_seed_topomesh.wisps(0)))[L1_seed_topomesh.wisp_property('layer',0).values()==1]
# non_L1_seed_cells = [c for c in L1_seed_topomesh.wisps(0) if not c in L1_seed_cells]
# for c in non_L1_seed_cells:
#     L1_seed_topomesh.remove_wisp(0,c)
# for property_name in L1_seed_topomesh.wisp_property_names(0):
#     d_ppty = dict(zip(list(L1_seed_topomesh.wisps(0)), L1_seed_topomesh.wisp_property(property_name,0).values(list(L1_seed_topomesh.wisps(0)))))
#     L1_seed_topomesh.update_wisp_property(property_name,0,d_ppty)

world.add(seed_topomesh,"detected_seed")
world["detected_seed_vertices"]["polydata_colormap"] = load_colormaps()['jet']
world["detected_seed"]["property_name_0"] = "layer"
world["detected_seed_vertices"]["intensity_range"] = (-0.5,2.5)
world["detected_seed_vertices"]["display_colorbar"] = False




world.add(L1_corrected_topomesh,"L1_corrected_nuclei"+suffix)
world["L1_corrected_nuclei"]["property_name_0"] = 'layer'
world["L1_corrected_nuclei_vertices"]["polydata_colormap"] = load_colormaps()['Greens']


# evaluation = evaluate_nuclei_detection(seed_topomesh, eq_seed_topomesh, max_matching_distance=2.0, outlying_distance=4.0, max_distance=np.linalg.norm(size*voxelsize))
# print(evaluation)

# Create a seed image from the nuclei barycenters:
seed_img = seed_image_from_points(img.shape,img.voxelsize,seed_positions,background_label=0)

# Add the "background seed":
background_threshold = 2000.
smooth_img_bck = linearfilter(img, param_str_2 = '-x 0 -y 0 -z 0 -sigma 3.0')
background_img = (smooth_img_bck<background_threshold).astype(np.uint16)
for it in xrange(10):
    background_img = morphology(background_img, param_str_2 = '-operation erosion -iterations 10')
seed_img += background_img
seed_img = SpatialImage(seed_img,voxelsize=img.voxelsize)
world.add(seed_img,'seed_image',colormap='glasbey',alphamap='constant',bg_id=0)
segmented_filename = image_dirname+"_seeds.inr"
imsave(segmented_filename,seed_img)
