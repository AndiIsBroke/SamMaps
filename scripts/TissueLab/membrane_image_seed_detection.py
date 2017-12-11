# imports

from timagetk.components import imread, imsave, SpatialImage
from timagetk.plugins import linear_filtering, morphology, h_transform, region_labeling

from openalea.mesh.property_topomesh_creation import vertex_topomesh
from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image

# Files's directories
#-----------------------

dirname = "/home/marie/"

# image_dirname = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/nuclei_ground_truth_images/"
# image_dirname = "/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images"
image_dirname = dirname+"Carlos/nuclei_images"

# filename = 'DR5N_6.1_151124_sam01_z0.50_t00'
# filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t05'
filename = 'qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t00'

reference_name = "PI"
# reference_name = "Lti6b"

microscope_orientation = -1

image_filename = image_dirname+"/"+filename+"/"+filename+"_"+reference_name+".inr.gz"

# Original image
#------------------------------

img = imread(image_filename)
size = np.array(img.shape)
voxelsize = np.array(img.voxelsize)

world.add(img,"reference_image",colormap="invert_grey",voxelsize=microscope_orientation*voxelsize)

gaussian_sigma = 3.0
morpho_radius = 3
h_min = 2

smooth_img = linear_filtering(input_img, std_dev=gaussian_sigma, method='gaussian_smoothing')
asf_img = morphology(smooth_img, max_radius=morpho_radius, method='co_alternate_sequential_filter')
ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')
con_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')


img_graph = graph_from_image(con_img,background=1,spatio_temporal_properties=['barycenter'],ignore_cells_at_stack_margins=False)
print img_graph.nb_vertices()," Seeds detected"
seed_positions = dict([(v,img_graph.vertex_property('barycenter')[v]) for v in img_graph.vertices()])
seed_topomesh = vertex_topomesh(seed_positions)
world.add(seed_topomesh,'seeds')

