import numpy as np
import scipy.ndimage as nd
from scipy.cluster.vq import vq

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

from scipy.optimize import curve_fit
from scipy.stats import norm as normal_distribution
from scipy.stats import f_oneway

from vplants.image.serial.all import imread
from vplants.image.spatial_image import SpatialImage
from timagetk.components import SpatialImage as TissueImage
from timagetk.algorithms import resample_isotropic

from vplants.cellcomplex.property_topomesh.utils.array_tools import array_unique
from vplants.cellcomplex.property_topomesh.utils.implicit_surfaces import vtk_marching_cubes
from vplants.cellcomplex.property_topomesh.property_topomesh_creation import triangle_topomesh
from vplants.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property
from vplants.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_vertex_property_from_faces

import vplants.cellcomplex.property_topomesh.property_topomesh_composition
reload(vplants.cellcomplex.property_topomesh.property_topomesh_composition)
from vplants.cellcomplex.property_topomesh.property_topomesh_composition import append_topomesh

import vplants.cellcomplex.property_topomesh.utils.vtk_optimization_tools
reload(vplants.cellcomplex.property_topomesh.utils.vtk_optimization_tools)
from vplants.cellcomplex.property_topomesh.utils.vtk_optimization_tools import property_topomesh_vtk_smoothing_decimation

from vplants.cellcomplex.property_topomesh.property_topomesh_optimization import property_topomesh_vertices_deformation
from vplants.cellcomplex.property_topomesh.property_topomesh_optimization import property_topomesh_isotropic_remeshing

from vplants.cellcomplex.triangular_mesh import TriangularMesh
from vplants.container import array_dict

from vplants.tissue_nukem_3d.epidermal_maps import compute_local_2d_signal
from vplants.tissue_nukem_3d.epidermal_maps import nuclei_density_function

from vplants.tissue_analysis.property_spatial_image import PropertySpatialImage

from time import time as current_time
from copy import deepcopy

import os
import logging
logging.getLogger().setLevel(logging.INFO)

if "world" in dir():
    world.clear()

# filename = "E35_sam05_t14_small"
# filename = "E37_sam07_t14_small"
# filename = "E37_sam07_t05_small"
# filename = "E35_sam04_t00"

filenames = []
filenames += ["E37_sam07_t05_small"]
filenames += ["qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t00"]
filenames += ["qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t05"]
filenames += ["qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t10"]
filenames += ["qDII-PIN1-CLV3-PI-LD_E35_171110_sam04_t14"]
filenames += ["qDII-PIN1-CLV3-PI-LD_E37_171113_sam07_t00"]
filenames += ["qDII-PIN1-CLV3-PI-LD_E37_171113_sam07_t05"]
filenames += ["qDII-PIN1-CLV3-PI-LD_E37_171113_sam07_t10"]
filenames += ["qDII-PIN1-CLV3-PI-LD_E37_171113_sam07_t14"]
#filenames += ["PIN2-PI-LD_TEST_180509_root01_t00"]

# dirname = "/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images"
dirname = "/projects/SamMaps/nuclei_images"

microscope_orientation=-1
resampling_voxelsize = 0.5
target_edge_length = 1.

#all_walls = True
all_walls = False

for filename in filenames:

    pi_img_filename = dirname+"/"+filename+"/"+filename+"_PI.inr.gz"
    pi_img = imread(pi_img_filename)
    if "world" in dir():
        world.add(pi_img,'membrane_image',voxelsize=microscope_orientation*np.array(pi_img.voxelsize),colormap='Oranges')

    pin_img_filename = dirname+"/"+filename+"/"+filename+"_PIN1.inr.gz"
    #pin_img_filename = dirname+"/"+filename+"/"+filename+"_PIN2.inr.gz"
    pin_img = imread(pin_img_filename)
    if "world" in dir():
        world.add(pin_img,'pin1_image',voxelsize=microscope_orientation*np.array(pin_img.voxelsize),colormap='Greens')

    nuclei_img_filename = dirname+"/"+filename+"/"+filename+"_TagBFP.inr.gz"
    if os.path.exists(nuclei_img_filename):
        nuclei_img = imread(nuclei_img_filename)
        pi_img = np.maximum(0,pi_img.astype(np.int32) - nuclei_img.astype(np.int32))

    start_time = current_time()
    #img_filename = dirname+"/"+filename+"/"+filename+"_PI_seg.inr.gz"
    img_filename = dirname+"/"+filename+"/"+filename+"_PI_segmented.inr.gz"
    seg_img = imread(img_filename)
    if resampling_voxelsize is not None:
        seg_img = resample_isotropic(TissueImage(seg_img,voxelsize=seg_img.voxelsize),voxelsize=resampling_voxelsize,option='label')
    if "world" in dir():
        world.add(seg_img,'segmented_image',voxelsize=microscope_orientation*np.array(seg_img.voxelsize),colormap='glasbey',alphamap='constant')
    logging.info("--> Loading & resampling segmented image ["+str(current_time() - start_time)+" s]")


    start_time = current_time()
    p_img = PropertySpatialImage(seg_img,ignore_cells_at_stack_margins=False)
    p_img.compute_default_image_properties()
    if all_walls:
        p_img.update_image_property('layer',dict(zip(p_img.labels,[1 for l in p_img.labels])))
    # p_img.compute_cell_meshes(sub_factor=1)
    logging.info("--> Computing image cell layers ["+str(current_time() - start_time)+" s]")


    start_time = current_time()
    anticlinal_walls = np.concatenate([[(l,n) for n in p_img.image_graph.neighbors(l) if p_img.image_property('layer')[n]==1] for l in p_img.labels if p_img.image_property('layer')[l]==1])
    anticlinal_walls = array_unique(np.sort(anticlinal_walls))
    logging.info("--> Extracting L1 anticlinal walls ["+str(current_time() - start_time)+" s]")


    from vplants.cellcomplex.property_topomesh.property_topomesh_io import save_ply_property_topomesh
from vplants.cellcomplex.property_topomesh.property_topomesh_io import read_ply_property_topomesh
    marching_cubes_filename = dirname+"/"+filename+"/"+filename+"_marching_cubes_L1_anticlinal_walls.ply"
    from vplants.cellcomplex.property_topomesh.property_topomesh_extraction import cell_topomesh

    # if os.path.exists(marching_cubes_filename):
    if False:
        start_time = current_time()
        all_wall_topomesh = read_ply_property_topomesh(marching_cubes_filename)
        logging.info("--> Loading wall meshes ["+str(current_time() - start_time)+" s]")

        wall_meshes = {}
        for c in all_wall_topomesh.wisps(3):
            wall_topomesh = cell_topomesh(all_wall_topomesh,cells=[c],copy_properties=True)
            wall_meshes[tuple(wall_topomesh.wisp_property('cell_labels',3).values()[0])] = wall_topomesh

        wall_cells = {}
        for i_w, (left_label, right_label) in enumerate(anticlinal_walls):
            if (left_label,right_label) in wall_meshes:
                wall_cells[i_w+2] = (left_label,right_label)
    else:
        cell_meshes = {}
        for l in p_img.labels:
            if p_img.image_property('layer')[l]==1:
                start_time = current_time()
                cell_meshes[l] = vtk_marching_cubes(seg_img==l,smoothing=0,decimation=0)
                logging.info("  --> Marching cubes on cell "+str(l)+" ["+str(current_time() - start_time)+" s]")

        wall_meshes = {}
        for i_w, (left_label, right_label) in enumerate(anticlinal_walls):
            start_time = current_time()
            left_points, left_triangles = cell_meshes[left_label]
            right_points, right_triangles = cell_meshes[right_label]

            left_mesh = TriangularMesh()
            left_mesh.points = dict(zip(np.arange(len(left_points)),microscope_orientation*left_points*np.array(seg_img.voxelsize)))
            left_mesh.triangles = dict(zip(np.arange(len(left_triangles)),left_triangles))
            left_mesh.triangle_data = dict(zip(np.arange(len(left_triangles)),[left_label for t in xrange(len(left_triangles))]))
            # world.add(left_mesh,"left_mesh",colormap='glasbey')

            right_mesh = TriangularMesh()
            right_mesh.points = dict(zip(np.arange(len(right_points)),microscope_orientation*right_points*np.array(seg_img.voxelsize)))
            right_mesh.triangles = dict(zip(np.arange(len(right_triangles)),right_triangles))
            right_mesh.triangle_data = dict(zip(np.arange(len(right_triangles)),[right_label for t in xrange(len(right_triangles))]))
            # world.add(right_mesh,"right_mesh",colormap='glasbey')

            right_left_matching = vq(right_points*np.array(seg_img.voxelsize),left_points*np.array(seg_img.voxelsize))

            # right_mesh.point_data = dict(zip(np.arange(len(right_points)),np.isclose(right_left_matching[1],0,atol=0.1)))
            # world.add(right_mesh,"right_mesh",colormap='glasbey')

            matching_vertices = np.isclose(right_left_matching[1],0,atol=0.1)

            if matching_vertices.sum()>0:

                right_wall_vertices = np.arange(len(right_points))[matching_vertices]
                right_wall_triangles = right_triangles[np.all([[v in right_wall_vertices for v in t] for t in right_triangles],axis=1)]

                if len(right_wall_triangles)>0:
                    wall_topomesh = triangle_topomesh(right_wall_triangles,dict(zip(right_wall_vertices,(microscope_orientation*right_points*np.array(seg_img.voxelsize))[right_wall_vertices])))
                    wall_topomesh.update_wisp_property("wall_label",0,np.array([i_w+2 for v in wall_topomesh.wisps(0)]))
                    wall_meshes[(left_label,right_label)] = wall_topomesh
                    logging.info("  --> Matching wall mesh ("+str(left_label)+","+str(right_label)+") ["+str(current_time() - start_time)+" s]")

        wall_cells = {}
        for i_w, (left_label, right_label) in enumerate(anticlinal_walls):
            if (left_label,right_label) in wall_meshes:
                wall_cells[i_w+2] = (left_label,right_label)

        all_wall_topomesh = deepcopy(wall_meshes[tuple(anticlinal_walls[0])])
        for i_w, (left_label, right_label) in enumerate(anticlinal_walls[1:]):
            if (left_label,right_label) in wall_meshes:
                start_time = current_time()
                all_wall_topomesh,_ = append_topomesh(all_wall_topomesh,wall_meshes[(left_label,right_label)],properties_to_append={0:['wall_label'],1:[],2:[],3:[]})
                logging.info("  --> Appending wall mesh "+str(i_w+1)+"/"+str(len(anticlinal_walls))+" ("+str(left_label)+","+str(right_label)+") ["+str(current_time() - start_time)+" s]")
        all_wall_topomesh.update_wisp_property('cell_labels',3,dict(zip(list(all_wall_topomesh.wisps(3)),[wall_cells[all_wall_topomesh.wisp_property("wall_label",0)[list(all_wall_topomesh.borders(3,c,3))[0]]] for c in all_wall_topomesh.wisps(3)])))


        save_ply_property_topomesh(all_wall_topomesh,marching_cubes_filename,properties_to_save={0:['wall_label'],1:[],2:[],3:['cell_labels']})


    # world.add(all_wall_topomesh,"anticlinal_walls")

    smooth_filename = dirname+"/"+filename+"/"+filename+"_smooth_L1_anticlinal_walls.ply"

    # if os.path.exists(smooth_filename):
    if False:
        all_smooth_wall_topomesh = read_ply_property_topomesh(smooth_filename)

        smoothed_wall_meshes = {}
        for c in all_wall_topomesh.wisps(3):
            try:
                smooth_wall_topomesh = cell_topomesh(all_smooth_wall_topomesh,cells=[c],copy_properties=True)
                smoothed_wall_meshes[tuple(smooth_wall_topomesh.wisp_property('cell_labels',3).values()[0])] = smooth_wall_topomesh
            except:
                pass
    else:
        logging.getLogger().setLevel(logging.INFO)
        smoothed_wall_meshes = {}
        for i_w, (left_label, right_label) in enumerate(wall_meshes.keys()):
            start_time = current_time()
            wall_topomesh = wall_meshes[(left_label,right_label)]
            wall_label = wall_topomesh.wisp_property('wall_label',0).values().mean()
            # world.add(wall_topomesh,"wall")
            smooth_wall_topomesh = deepcopy(wall_topomesh)
            smooth_wall_topomesh = property_topomesh_isotropic_remeshing(smooth_wall_topomesh,maximal_length=np.min(seg_img.voxelsize),iterations=np.ceil(np.log2(np.max(seg_img.voxelsize)/np.min(seg_img.voxelsize))))
            # property_topomesh_vertices_deformation(smooth_wall_topomesh,omega_forces={'taubin_smothing':0.65},iterations=10,gaussian_sigma=0.5)
            compute_topomesh_property(smooth_wall_topomesh,'area',2)
            triangle_area = smooth_wall_topomesh.wisp_property('area',2).values().sum()/float(smooth_wall_topomesh.nb_wisps(2))
            decimation = np.ceil((np.power(target_edge_length,2)*np.sqrt(2.)/4.)/triangle_area)
            # print "Before decimation : ",smooth_wall_topomesh.nb_wisps(0)," Vertices, ",smooth_wall_topomesh.nb_wisps(2)," Triangles"
            smooth_wall_topomesh = property_topomesh_vtk_smoothing_decimation(smooth_wall_topomesh,smoothing=0,decimation=decimation)
            # print "After decimation  : ",smooth_wall_topomesh.nb_wisps(0)," Vertices, ",smooth_wall_topomesh.nb_wisps(2)," Triangles"
            # raw_input()
            if (smooth_wall_topomesh is not None) and (smooth_wall_topomesh.nb_wisps(2)>5):
                smooth_wall_topomesh = property_topomesh_isotropic_remeshing(smooth_wall_topomesh,maximal_length=target_edge_length,iterations=5)
                smooth_wall_topomesh.update_wisp_property("wall_label",0,np.array([wall_label for v in smooth_wall_topomesh.wisps(0)]))
                smoothed_wall_meshes[(left_label,right_label)] = smooth_wall_topomesh
                # print wall_topomesh.nb_wisps(2)," --> ",smooth_wall_topomesh.nb_wisps(2)," [",decimation,"]"
                # world.add(smooth_wall_topomesh,"smoothed_wall")
                logging.info("  --> Smoothing wall mesh "+str(i_w+1)+"/"+str(len(wall_meshes))+" ("+str(left_label)+","+str(right_label)+") ["+str(current_time() - start_time)+" s]")

        logging.getLogger().setLevel(logging.INFO)
        all_smooth_wall_topomesh = deepcopy(smoothed_wall_meshes[tuple(anticlinal_walls[0])])
        for left_label, right_label in anticlinal_walls[1:]:
            if (left_label,right_label) in smoothed_wall_meshes:
                start_time = current_time()
                logging.info("  --> Appending smooth wall mesh "+str(i_w+1)+"/"+str(len(anticlinal_walls)))
                all_smooth_wall_topomesh,_ = append_topomesh(all_smooth_wall_topomesh,smoothed_wall_meshes[(left_label,right_label)],properties_to_append={0:['wall_label'],1:[],2:[],3:[]})
                logging.info("  <-- Appending smooth wall mesh  ("+str(left_label)+","+str(right_label)+") ["+str(current_time() - start_time)+" s]")
        all_smooth_wall_topomesh.update_wisp_property('cell_labels',3,dict(zip(list(all_smooth_wall_topomesh.wisps(3)),[wall_cells[all_smooth_wall_topomesh.wisp_property("wall_label",0)[list(all_smooth_wall_topomesh.borders(3,c,3))[0]]] for c in all_smooth_wall_topomesh.wisps(3)])))

        save_ply_property_topomesh(all_smooth_wall_topomesh,smooth_filename,properties_to_save={0:['wall_label'],1:[],2:[],3:['cell_labels']})


    wall_areas = {}
    for i_w, (left_label, right_label) in enumerate(smoothed_wall_meshes.keys()):

        start_time = current_time()
        smooth_wall_topomesh = smoothed_wall_meshes[(left_label,right_label)]

        compute_topomesh_property(smooth_wall_topomesh,'normal',2,normal_method='orientation')
        normal_left_to_right = np.sign(np.dot(smooth_wall_topomesh.wisp_property('normal',2).values(),p_img.image_property('barycenter')[right_label]-p_img.image_property('barycenter')[left_label]))
        smooth_wall_topomesh.update_wisp_property('normal',2,array_dict(normal_left_to_right[:,np.newaxis]*smooth_wall_topomesh.wisp_property('normal',2).values(),keys=smooth_wall_topomesh.wisp_property('normal',2).keys()))
        compute_topomesh_vertex_property_from_faces(smooth_wall_topomesh,'normal',neighborhood=5.,adjacency_sigma=3.)
        # topomesh_property_gaussian_filtering(smooth_wall_topomesh,'normal',0,neighborhood=5,adjacency_sigma=3,distance_sigma=10.)

        compute_topomesh_property(smooth_wall_topomesh,'area',2)
        wall_areas[(left_label,right_label)] = smooth_wall_topomesh.wisp_property('area',2).values().sum()

        contour_edges = [e for e in smooth_wall_topomesh.wisps(1) if smooth_wall_topomesh.nb_regions(1,e)<2]
        smooth_wall_topomesh.update_wisp_property('contour',1,array_dict([e in contour_edges for e in smooth_wall_topomesh.wisps(1)],keys=list(smooth_wall_topomesh.wisps(1))))
        contour_vertices = [v for v in smooth_wall_topomesh.wisps(0) if np.any([e in contour_edges for e in smooth_wall_topomesh.regions(0,v)])]
        smooth_wall_topomesh.update_wisp_property('contour',0,array_dict([v in contour_vertices for v in smooth_wall_topomesh.wisps(0)],keys=list(smooth_wall_topomesh.wisps(0))))
        # smooth_wall_topomesh.update_wisp_property('normal',0,array_dict((1.-smooth_wall_topomesh.wisp_property('contour',0).values())[:,np.newaxis]*smooth_wall_topomesh.wisp_property('normal',0).values(),keys=smooth_wall_topomesh.wisp_property('normal',0).keys()))

        logging.info("  --> Computing wall normals "+str(i_w+1)+"/"+str(len(smoothed_wall_meshes))+" ("+str(left_label)+","+str(right_label)+") ["+str(current_time() - start_time)+" s]")


    if 'world' in dir():
        world.add(all_smooth_wall_topomesh,"anticlinal_smooth_walls")
        world["anticlinal_smooth_walls"]['property_name_0'] = 'normal'
        world["anticlinal_smooth_walls"]['display_0'] = True
        world["anticlinal_smooth_walls_vertices"]['polydata_alpha'] = 0.5
        world["anticlinal_smooth_walls_vertices"]['point_radius'] = 10
        # world["anticlinal_smooth_walls_faces"]['polydata_alpha'] = 0.5
        # world["anticlinal_smooth_walls_faces"]['point_radius'] = 10
        world.add(all_smooth_wall_topomesh,"reverse_anticlinal_smooth_walls")
        world["reverse_anticlinal_smooth_walls"]['display_3'] = False
        world["reverse_anticlinal_smooth_walls"]['property_name_0'] = 'normal'
        world["reverse_anticlinal_smooth_walls"]['display_0'] = True
        world["reverse_anticlinal_smooth_walls_vertices"]['polydata_alpha'] = 0.5
        world["reverse_anticlinal_smooth_walls_vertices"]['point_radius'] = -10
        # world["reverse_anticlinal_smooth_walls_faces"]['polydata_alpha'] = 0.5
        # world["reverse_anticlinal_smooth_walls_faces"]['point_radius'] = 10



    def normal_function(x, amplitude=1, loc=0, scale=1, offset=0):
        return offset+amplitude*normal_distribution.pdf(x,loc=loc,scale=scale)

    wall_sigma = 1.5
    line_sigma = 1.
    fitting_error_threshold = 10.

    pin1_distance = 0.6

    pin1_orientations = {}
    pin1_shifts = {}
    pin1_intensities = {}
    shift_pvalues = {}

    pin1_intensities_left = {}
    pin1_intensities_right = {}

    bbox_sizes = {}

    if not os.path.exists(dirname+"/"+filename+"/PIN1/"):
        os.makedirs(dirname+"/"+filename+"/PIN1/")

    # left_label, right_label =  wall_areas.keys()[np.argmax(wall_areas.values())]
    # left_label, right_label =  (4,16)
    for i_w, (left_label, right_label) in enumerate(smoothed_wall_meshes.keys()):

        start_time = current_time()
        logging.info("  --> Computing local PIN1 polarity "+str(i_w+1)+"/"+str(len(smoothed_wall_meshes))+" ("+str(left_label)+","+str(right_label)+")")

        # left_label, right_label =  (27,141)
        smooth_wall_topomesh = smoothed_wall_meshes[(left_label,right_label)]
        # world.add(smooth_wall_topomesh,"smoothed_wall")

        wall_bbox = []
        wall_bbox += [np.min([smooth_wall_topomesh.wisp_property('barycenter',0).values()+side*wall_sigma*smooth_wall_topomesh.wisp_property('normal',0).values() for side in [-1,1]],axis=(0,1))]
        wall_bbox += [np.max([smooth_wall_topomesh.wisp_property('barycenter',0).values()+side*wall_sigma*smooth_wall_topomesh.wisp_property('normal',0).values() for side in [-1,1]],axis=(0,1))]

        wall_bbox = np.sort(microscope_orientation*np.transpose(wall_bbox)/np.array(pi_img.voxelsize)[:,np.newaxis])
        wall_bbox[:,0] = np.minimum(np.maximum(0,np.floor(wall_bbox[:,0])),np.array(pi_img.shape)-1)
        wall_bbox[:,1] = np.minimum(np.maximum(0,np.ceil(wall_bbox[:,1])),np.array(pi_img.shape)-1)

        bbox_sizes[(left_label,right_label)] = int(np.prod(wall_bbox[:,1]-wall_bbox[:,0]))

        print "    --> Bounding box size ",left_label,",",right_label," : ",bbox_sizes[(left_label,right_label)]

        #if bbox_sizes[(left_label,right_label)]<30000:
        if True:

            wall_neighborhood_coords = np.mgrid[wall_bbox[0,0]:wall_bbox[0,1]+1,wall_bbox[1,0]:wall_bbox[1,1]+1,wall_bbox[2,0]:wall_bbox[2,1]+1]
            wall_neighborhood_coords = np.concatenate(np.concatenate(np.transpose(wall_neighborhood_coords,(1,2,3,0))))
            wall_neighborhood_coords = array_unique(wall_neighborhood_coords).astype(int)

            wall_neighborhood_points = microscope_orientation*wall_neighborhood_coords*np.array(pi_img.voxelsize)

            wall_neighborhood_coords = tuple(np.transpose(wall_neighborhood_coords))

            wall_neighborhood_pi = pi_img[wall_neighborhood_coords]
            wall_neighborhood_pin1 = pin_img[wall_neighborhood_coords]

            wall_points = smooth_wall_topomesh.wisp_property('barycenter',0).values()[True - smooth_wall_topomesh.wisp_property('contour',0).values()]
            wall_normal_vectors = smooth_wall_topomesh.wisp_property('normal',0).values()[True - smooth_wall_topomesh.wisp_property('contour',0).values()]

            # np.dot(wall_normal_vectors,microscope_orientation*(p_img.image_property('barycenter')[right_label]-p_img.image_property('barycenter')[left_label]))

            if len(wall_points)>0:

                wall_neighborhood_vectors = wall_neighborhood_points[np.newaxis,:] - wall_points[:,np.newaxis]

                wall_neighborhood_dot_products = np.einsum("...ij,...ij->...i",wall_neighborhood_vectors,wall_normal_vectors[:,np.newaxis])

                wall_neighborhood_projected_points = wall_points[:,np.newaxis] + wall_neighborhood_dot_products[:,:,np.newaxis]*wall_normal_vectors[:,np.newaxis]
                wall_neighborhood_projected_distances = np.linalg.norm(wall_neighborhood_projected_points-wall_neighborhood_points[np.newaxis,:],axis=2)


                figure = plt.figure(0)
                figure.clf()
                figure.patch.set_facecolor('w')

                pin1_pi_shifts = []
                pin1_amplitudes = []
                fitting_errors = []

                pin1_left_intensities = []
                pin1_right_intensities = []


                logging.info("  --> "+str(len(wall_neighborhood_dot_products))+" lines to consider")

                for i_line, (wall_distances, line_distances) in enumerate(zip(wall_neighborhood_dot_products,wall_neighborhood_projected_distances)):
                    # line_weights = (line_distances<3.*line_sigma)&(np.abs(wall_distances)<1.5*wall_sigma)

                    # line_weights = (np.exp(-np.power(line_distances,2)/np.power(line_sigma,2)))*(np.abs(wall_distances)<wall_sigma)
                    line_weights = (np.exp(-np.power(line_distances,2)/np.power(line_sigma,2)))*(np.abs(wall_distances)<wall_sigma)


                    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
                    p0 = [1., 0., 1., 0.]

                    try:
                        line_weights = (line_weights>0.1).astype(float)
                        pi_p,pi_cov = curve_fit(normal_function, wall_distances[line_weights>0.1], wall_neighborhood_pi[line_weights>0.1], p0=p0, maxfev=int(1e3))
                        # pi_p,_ = curve_fit(normal_function, wall_distances[line_weights>0], wall_neighborhood_pi[line_weights>0], p0=p0, sigma=line_weights[line_weights>0])
                        pi_amplitude, pi_loc, pi_scale, pi_offset = pi_p
                        pin1_p,pin1_cov = curve_fit(normal_function, wall_distances[line_weights>0.1], wall_neighborhood_pin1[line_weights>0.1], p0=p0, maxfev=int(1e3))
                        # pin1_p,_ = curve_fit(normal_function, wall_distances[line_weights>0], wall_neighborhood_pin1[line_weights>0], p0=p0, sigma=line_weights[line_weights>0])
                        pin1_amplitude, pin1_loc, pin1_scale, pin1_offset = pin1_p

                    except Exception as e:
                        #print e
                        logging.info("    --> "+str(i_line+1)+"/"+str(len(wall_neighborhood_dot_products))+" Impossible to fit gaussian on this line...")
                        # pass
                    else:
                        pin1_pi_shifts += [pin1_loc-pi_loc]
                        pin1_amplitudes += [normal_function(pin1_loc,*pin1_p)]

                        # np.power(normal_function(wall_distances[line_weights>0.1],*pi_p)-wall_neighborhood_pi[line_weights>0.1],2).mean()
                        pi_relative_error = float((np.abs(normal_function(wall_distances[line_weights>0.1],*pi_p)-wall_neighborhood_pi[line_weights>0.1])/normal_function(wall_distances[line_weights>0.1],*pi_p)).mean())
                        pin1_relative_error = float((np.abs(normal_function(wall_distances[line_weights>0.1],*pin1_p)-wall_neighborhood_pin1[line_weights>0.1])/normal_function(wall_distances[line_weights>0.1],*pin1_p)).mean())
                        fitting_errors += [(pi_relative_error,pin1_relative_error)]
                        logging.info("    --> "+str(i_line+1)+"/"+str(len(wall_neighborhood_dot_products))+" OK "+str(fitting_errors[-1]))

                        pi_wall_distances = wall_distances-pi_loc
                        left_pin1 = wall_neighborhood_pin1[(line_weights>0.1)&(pi_wall_distances>=0)&(np.abs(pi_wall_distances)< pin1_distance)].mean()
                        right_pin1 = wall_neighborhood_pin1[(line_weights>0.1)&(pi_wall_distances<=0)&(np.abs(pi_wall_distances)< pin1_distance)].mean()

                        pin1_left_intensities += [left_pin1]
                        pin1_right_intensities += [right_pin1]



                        wall_distance_range = np.linspace(-2.*wall_sigma,2.*wall_sigma,101)

                        figure = plt.figure(0)

                        if i_line == 0:
                            figure.add_subplot(4,1,1)
                            pi_colors = np.array([[0.6,0.6,0.6,1] for d in wall_distances])
                            pi_colors[:,3] = 0.5*line_weights
                            figure.gca().scatter(wall_distances[line_weights>0.1],wall_neighborhood_pi[line_weights>0.1],color=pi_colors)
                            figure.gca().plot(wall_distance_range,normal_function(wall_distance_range,*pi_p),color=[0.6,0.6,0.6],linewidth=2,alpha=1)
                            figure.gca().plot([pi_loc,pi_loc],[0,65535],color=[0.6,0.6,0.6],alpha=0.5)

                            pin1_colors = np.array([[0.1,0.8,0.2,1] for d in wall_distances])
                            pin1_colors[:,3] = 0.5*line_weights
                            figure.gca().scatter(wall_distances[line_weights>0.1],wall_neighborhood_pin1[line_weights>0.1],color=pin1_colors)
                            figure.gca().plot(wall_distance_range,normal_function(wall_distance_range,*pin1_p),color=[0.1,0.8,0.2],linewidth=2,alpha=1)
                            figure.gca().plot([pin1_loc,pin1_loc],[0,65535],color=[0.1,0.8,0.2],alpha=0.5)


                            figure.gca().fill_between([pi_loc+pin1_distance,pi_loc],[0,0],[left_pin1,left_pin1],color=[0.1,0.8,0.2],alpha=0.5*left_pin1/65535.)
                            figure.gca().fill_between([pi_loc-pin1_distance,pi_loc],[0,0],[right_pin1,right_pin1],color=[0.1,0.8,0.2],alpha=0.5*left_pin1/65535.)



                        figure.add_subplot(4,1,2)
                        # pi_colors = np.array([[0.6,0.6,0.6,1] for d in wall_distances])
                        # pi_colors[:,3] = line_weights
                        # figure.gca().scatter(wall_distances,wall_neighborhood_pi,color=pi_colors)
                        figure.gca().plot(wall_distance_range,normal_function(wall_distance_range,*pi_p),color=[0.6,0.6,0.6],linewidth=1,alpha=0.1)
                        figure.gca().plot([pi_loc,pi_loc],[0,65535],color=[0.6,0.6,0.6],alpha=0.1)

                        # pin1_colors = np.array([[0.1,0.8,0.2,1] for d in wall_distances])
                        # pin1_colors[:,3] = line_weights
                        # figure.gca().scatter(wall_distances,wall_neighborhood_pin1,color=pin1_colors)
                        figure.gca().plot(wall_distance_range,normal_function(wall_distance_range,*pin1_p),color=[0.1,0.8,0.2],linewidth=1,alpha=0.1)
                        figure.gca().plot([pin1_loc,pin1_loc],[0,65535],color=[0.1,0.8,0.2],alpha=0.1)

                        figure.add_subplot(4,1,3)

                        #if np.max([pi_relative_error,pin1_relative_error])<0.5:
                        if np.max([pi_relative_error,pin1_relative_error])<fitting_error_threshold:
                            # pi_colors = np.array([[0.6,0.6,0.6,1] for d in wall_distances])
                            # pi_colors[:,3] = line_weights
                            # figure.gca().scatter(wall_distances,wall_neighborhood_pi,color=pi_colors)
                            figure.gca().plot(wall_distance_range-pi_loc,normal_function(wall_distance_range,*pi_p),color=[0.6,0.6,0.6],linewidth=1,alpha=0.1)

                            # pin1_colors = np.array([[0.1,0.8,0.2,1] for d in wall_distances])
                            # pin1_colors[:,3] = line_weights
                            # figure.gca().scatter(wall_distances,wall_neighborhood_pin1,color=pin1_colors)
                            #figure.gca().plot(wall_distance_range-pi_loc,normal_function(wall_distance_range,*pin1_p),color=[0.1,0.8,0.2],linewidth=1,alpha=0.1)

                            figure.gca().plot([0,0],[0,65535],color=[0.6,0.6,0.6],alpha=0.1)
                            figure.gca().plot([pin1_loc-pi_loc,pin1_loc-pi_loc],[0,65535],color=[0.1,0.8,0.2],alpha=0.1)
                            figure.gca().plot([pin1_distance,0],[left_pin1,left_pin1],color=[0.1,0.8,0.2],alpha=0.1)
                            figure.gca().plot([-pin1_distance,0],[right_pin1,right_pin1],color=[0.1,0.8,0.2],alpha=0.1)

                #if len(pin1_pi_shifts)>0:
                if len(pin1_left_intensities)>0:

                    pin1_pi_shifts = np.array(pin1_pi_shifts)
                    pin1_amplitudes = np.array(pin1_amplitudes)
                    pin1_left_intensities = np.array(pin1_left_intensities)
                    pin1_right_intensities = np.array(pin1_right_intensities)

                    shift_validity = np.ones_like(pin1_pi_shifts).astype(bool)
                    shift_validity = shift_validity & (np.max(fitting_errors,axis=1)<fitting_error_threshold)
                    shift_validity = shift_validity & (pin1_amplitudes<np.iinfo(pin_img.dtype).max)
                    #pin1_pi_shifts = pin1_pi_shifts[shift_validity]
                    #pin1_amplitudes = pin1_amplitudes[shift_validity]

                    pi_validity =  np.ones_like(pin1_pi_shifts).astype(bool)
                    pi_validity = pi_validity & (np.array(fitting_errors)[:,0]<fitting_error_threshold)
                    pin1_pi_shifts = pin1_pi_shifts[pi_validity]
                    pin1_amplitudes = pin1_amplitudes[pi_validity]
                    pin1_left_intensities = pin1_left_intensities[pi_validity]
                    pin1_right_intensities = pin1_right_intensities[pi_validity]

                    ##if len(pin1_pi_shifts)>0:
                    if len(pin1_left_intensities)>0:
                        # anova = f_oneway(pin1_pi_shifts,normal_distribution.rvs(0,np.std(pin1_pi_shifts),size=len(pin1_pi_shifts)))
                        # if anova.pvalue < 1e-5:
                        # if anova.pvalue < 1e-3:
                        #     pin1_orientation = -np.sign(np.median(pin1_pi_shifts))
                        # elif anova.pvalue < 1e-2:
                        #      pin1_orientation = -np.sign(np.median(pin1_pi_shifts))/2.
                        # elif anova.pvalue < 5e-2:
                        #     pin1_orientation = -np.sign(np.median(pin1_pi_shifts))/4.
                        # else:
                        #     pin1_orientation = 0

                        anova = f_oneway(pin1_left_intensities,pin1_right_intensities)
                        if anova.pvalue < 1e-3:
                            pin1_orientation = -np.sign(np.median(pin1_left_intensities-pin1_right_intensities))
                        elif anova.pvalue < 1e-2:
                            pin1_orientation = -np.sign(np.median(pin1_left_intensities-pin1_right_intensities))/2.
                        elif anova.pvalue < 5e-2:
                            pin1_orientation = -np.sign(np.median(pin1_left_intensities-pin1_right_intensities))/4.
                        else:
                            pin1_orientation = 0


                        pin1_orientations[(left_label,right_label)] = pin1_orientation
                        pin1_shifts[(left_label,right_label)] = np.median(pin1_pi_shifts)
                        pin1_intensities[(left_label,right_label)] = np.median(pin1_amplitudes)

                        pin1_intensities_left[(left_label,right_label)]  = np.median(pin1_left_intensities)
                        pin1_intensities_right[(left_label,right_label)]  = np.median(pin1_right_intensities)

                        figure = plt.figure(0)

                        figure.add_subplot(4,1,1)
                        figure.gca().set_xlim(-1.5*wall_sigma,1.5*wall_sigma)
                        figure.gca().set_xticklabels([])
                        figure.gca().set_ylim(0,65535)
                        figure.gca().set_ylabel("Signal intensity",size=14)

                        figure.add_subplot(4,1,2)
                        figure.gca().set_xlim(-1.5*wall_sigma,1.5*wall_sigma)
                        figure.gca().set_xticklabels([])
                        figure.gca().set_ylim(0,65535)
                        figure.gca().set_ylabel("Signal intensity",size=14)

                        figure.add_subplot(4,1,3)
                        figure.gca().set_xlim(-1.5*wall_sigma,1.5*wall_sigma)
                        figure.gca().set_xticklabels([])
                        figure.gca().set_ylim(0,65535)
                        figure.gca().set_ylabel("Signal intensity",size=14)

                        figure.add_subplot(4,1,4)
                        figure.gca().text(-1,0.5,str(right_label),size=18)
                        figure.gca().text(1,0.5,str(left_label),size=18)

                        figure.gca().arrow(-pin1_orientation*0.6,0.5,pin1_orientation,0,head_width=0.1,head_length=0.1,color=[0.1,0.8,0.2],linewidth=3,alpha=0.5)

                        #if pin1_orientations[(left_label,right_label)]!=0:
                            #figure.gca().fill_between([0,pin1_shifts[(left_label,right_label)]],[0,0],[pin1_intensities[(left_label,right_label)]/float(pin_img.max()),pin1_intensities[(left_label,right_label)]/float(pin_img.max())],color=[0.1,0.8,0.2],alpha=np.abs(pin1_orientations[(left_label,right_label)]))

                        figure.gca().fill_between([0,pin1_distance],[0,0],[pin1_intensities_left[(left_label,right_label)]/float(pin_img.max()),pin1_intensities_left[(left_label,right_label)]/float(pin_img.max())],color=[0.1,0.8,0.2],alpha=0.5*np.abs(pin1_orientations[(left_label,right_label)]))
                        figure.gca().fill_between([0,-pin1_distance],[0,0],[pin1_intensities_right[(left_label,right_label)]/float(pin_img.max()),pin1_intensities_right[(left_label,right_label)]/float(pin_img.max())],color=[0.1,0.8,0.2],alpha=0.5*np.abs(pin1_orientations[(left_label,right_label)]))


                        figure.gca().set_xlim(-1.5*wall_sigma,1.5*wall_sigma)
                        figure.gca().set_xlabel("Distance to the wall ($\mu m$)",size=14)
                        figure.gca().set_ylim(0,1)
                        figure.gca().set_yticks([])

                        figure.set_size_inches(10,10)
                        figure.tight_layout()
                        figure.savefig(dirname+"/"+filename+"/PIN1/"+filename+"_L1_"+str(left_label)+"_to_"+str(right_label)+"_PIN1_polarity.png")

                        logging.info("  <-- Computing local PIN1 polarity ("+str(left_label)+","+str(right_label)+") ["+str(current_time() - start_time)+" s]")



    X = p_img.image_property('barycenter').values()[:,0][p_img.image_property('layer').values()==1]
    Y = p_img.image_property('barycenter').values()[:,1][p_img.image_property('layer').values()==1]
    Z = p_img.image_property('barycenter').values()[:,2][p_img.image_property('layer').values()==1]

    xx,yy = np.meshgrid(np.linspace(0,(pi_img.shape[0]-1)*pi_img.voxelsize[0],pi_img.shape[0]),np.linspace(0,(pi_img.shape[1]-1)*pi_img.voxelsize[1],pi_img.shape[1]))
    # extent = xx.max(),xx.min(),yy.min(),yy.max()
    extent = xx.min(),xx.max(),yy.max(),yy.min()
    zz = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),Z)
    coords = (np.transpose([xx,yy,zz],(1,2,0)))/np.array(pi_img.voxelsize)
    for k in xrange(3):
        coords[:,:,k] = np.maximum(np.minimum(coords[:,:,k],pi_img.shape[k]-1),0)
    coords[np.isnan(coords)]=0
    coords = coords.astype(int)
    coords = tuple(np.transpose(np.concatenate(coords)))


    wall_centers={}
    wall_normals={}
    wall_pin1_vectors={}
    for left_label, right_label in pin1_orientations.keys():
        smooth_wall_topomesh = smoothed_wall_meshes[(left_label,right_label)]
        pin1_orientation = pin1_orientations[(left_label,right_label)]

        wall_points = smooth_wall_topomesh.wisp_property('barycenter',0).values()[True - smooth_wall_topomesh.wisp_property('contour',0).values()]
        wall_normal_vectors = smooth_wall_topomesh.wisp_property('normal',0).values()[True - smooth_wall_topomesh.wisp_property('contour',0).values()]

        wall_centers[(left_label,right_label)] = wall_points.mean(axis=0)
        wall_normals[(left_label,right_label)] = wall_normal_vectors.mean(axis=0)

    for left_label, right_label in pin1_orientations.keys():
        pin1_orientations[(right_label,left_label)] = -pin1_orientations[(left_label,right_label)]
        wall_centers[(right_label,left_label)] = wall_centers[(left_label,right_label)]
        wall_normals[(right_label,left_label)] = -wall_normals[(left_label,right_label)]
        wall_areas[(right_label,left_label)] = wall_areas[(left_label,right_label)]
        pin1_intensities[(right_label,left_label)] = pin1_intensities[(left_label,right_label)]
        pin1_intensities_left[(right_label,left_label)] = pin1_intensities_right[(left_label,right_label)]
        pin1_intensities_right[(right_label,left_label)] = pin1_intensities_left[(left_label,right_label)]
        pin1_shifts[(right_label,left_label)] = -pin1_shifts[(left_label,right_label)]

    wall_cells = np.array(wall_normals.keys())

    for left_label, right_label in wall_cells:
        wall_pin1_vectors[(left_label,right_label)] = pin1_orientations[(left_label,right_label)]*wall_normals[(left_label,right_label)]

    import pandas as pd

    wall_pin1_data = pd.DataFrame()

    wall_pin1_data['left_label'] = wall_cells[:,0]
    wall_pin1_data['right_label'] = wall_cells[:,1]
    wall_pin1_data['wall_area'] = [wall_areas[(l,r)] for (l,r) in wall_cells]
    for k,dim in enumerate(['x','y','z']):
        wall_pin1_data['wall_center_'+dim] = [wall_centers[(l,r)][k] for (l,r) in wall_cells]
    for k,dim in enumerate(['x','y','z']):
        wall_pin1_data['wall_normal_'+dim] = [wall_normals[(l,r)][k] for (l,r) in wall_cells]
    wall_pin1_data['pin1_orientation'] = [pin1_orientations[(l,r)] for (l,r) in wall_cells]
    wall_pin1_data['pin1_signal'] = [pin1_intensities[(l,r)] for (l,r) in wall_cells]
    #wall_pin1_data['pin1_shift'] = [pin1_shifts[(l,r)] for (l,r) in wall_cells]
    wall_pin1_data['pin1_signal_left'] = [pin1_intensities_left[(l,r)] for (l,r) in wall_cells]
    wall_pin1_data['pin1_signal_right'] = [pin1_intensities_right[(l,r)] for (l,r) in wall_cells]

    wall_pin1_data.to_csv(dirname+"/"+filename+"/"+filename+"_anticlinal_L1_wall_PIN1_data.csv",index=False)


    cell_centers = dict(zip(np.unique(wall_cells),[p_img.image_property('barycenter')[c][:2] for c in np.unique(wall_cells)]))
    wall_weights = dict(zip([(l,r) for (l,r) in wall_cells],[pin1_intensities[(l,r)]*wall_areas[(l,r)] for (l,r) in wall_cells]))
    cell_pin1_vectors = dict(zip(np.unique(wall_cells),np.transpose([nd.sum(np.array([wall_weights[(l,r)]*wall_pin1_vectors[(l,r)][k] for (l,r) in wall_cells]),wall_cells[:,0],index=np.unique(wall_cells)) for k in xrange(2)])/nd.sum(np.array([wall_weights[(l,r)] for (l,r) in wall_cells]),wall_cells[:,0],index=np.unique(wall_cells))[:,np.newaxis]))


    from vplants.cellcomplex.property_topomesh.utils.matplotlib_tools import mpl_draw_topomesh

    all_smooth_wall_topomesh.update_wisp_property('barycenter',0,array_dict(microscope_orientation*all_smooth_wall_topomesh.wisp_property('barycenter',0).values(),all_smooth_wall_topomesh.wisp_property('barycenter',0).keys()))

    figure = plt.figure(0)
    figure.clf()
    figure.patch.set_facecolor('w')

    mpl_draw_topomesh(all_smooth_wall_topomesh, figure, degree=1, color='k', alpha=0.2, linewidth=1)
    figure.gca().imshow(pi_img[coords].reshape(xx.shape),cmap="Oranges",vmin=0,vmax=50000,extent=extent,interpolation='none',alpha=1)
    # figure.gca().imshow(pin_img[coords].reshape(xx.shape),cmap="Greens",vmin=10000,vmax=60000,extent=extent,interpolation='none',alpha=1)

    figure.gca().axis('off')
    figure.set_size_inches(32,32)
    figure.tight_layout()
    figure.savefig(dirname+"/"+filename+"/"+filename+"_L1_wall_meshes.png")


    figure = plt.figure(1)
    figure.clf()
    figure.patch.set_facecolor('w')

    # mpl_draw_topomesh(all_smooth_wall_topomesh, figure, degree=1, color='k', alpha=0.2, linewidth=1)
    figure.gca().imshow(pi_img[coords].reshape(xx.shape),cmap="Oranges",vmin=0,vmax=50000,extent=extent,interpolation='none',alpha=1)
    figure.gca().imshow(pin_img[coords].reshape(xx.shape),cmap="Greens",vmin=10000,vmax=60000,extent=extent,interpolation='none',alpha=0.66)

    for (left_label, right_label), pin1_orientation in pin1_orientations.items():
        intensity = pin1_intensities[(left_label,right_label)]
        shift = pin1_shifts[(left_label,right_label)]
        wall_center = microscope_orientation*wall_centers[(left_label,right_label)]
        wall_normal = microscope_orientation*wall_normals[(left_label,right_label)]
        if pin1_orientation:
            pin1_vector = np.sign(pin1_orientation)*wall_normal
            # pin1_color = np.array([0.1,0.8,0.2])*np.abs(pin1_orientation) + np.array([1.,1.,1.])*(1.-np.abs(pin1_orientation))
            # pin1_edgecolor = np.array([0.0,0.0,0.0])*np.abs(pin1_orientation) + np.array([0.8,0.2,0.2])*(1.-np.abs(pin1_orientation))
            # if np.abs(pin1_orientation)==1:
            #     pin1_color = 'chartreuse'
            # elif np.abs(pin1_orientation)==0.5:
            #     pin1_color = 'gold'
            # elif np.abs(pin1_orientation)==0.25:
            #     pin1_color = 'peru'
            pin1_color = mpl.cm.ScalarMappable(cmap='Greens',norm=Normalize(vmin=0,vmax=65535)).to_rgba(intensity)

            figure.gca().arrow(wall_center[0]-0.75*pin1_vector[0],wall_center[1]-0.75*pin1_vector[1],pin1_vector[0],pin1_vector[1],head_width=np.sqrt(np.abs(shift)/0.2),head_length=np.sqrt(np.abs(shift)/0.2),edgecolor='k',facecolor=pin1_color,linewidth=1,alpha=np.abs(pin1_orientation))


    for label in cell_pin1_vectors.keys():
        cell_center = cell_centers[label]
        # cell_pin1_vector = 2.*microscope_orientation*cell_pin1_vectors[label]
        figure.gca().text(cell_center[0],cell_center[1],str(label),zorder=10,size=12,horizontalalignment='center',verticalalignment='top')
        # if np.all(cell_pin1_vector):
            # figure.gca().scatter(microscope_orientation*p_img.image_property('barycenter')[l][0],microscope_orientation*p_img.image_property('barycenter')[l][1])
            # figure.gca().arrow(cell_center[0]-0.75*cell_pin1_vector[0],cell_center[1]-0.75*cell_pin1_vector[1],cell_pin1_vector[0],cell_pin1_vector[1],head_width=0.5,head_length=0.5,edgecolor='k',facecolor='k',linewidth=2,alpha=0.25)

    figure.gca().axis('off')
    figure.set_size_inches(50,50)
    figure.tight_layout()
    figure.savefig(dirname+"/"+filename+"/"+filename+"_L1_PIN1_polarities.png")

    figure = plt.figure(2)
    figure.clf()
    figure.patch.set_facecolor('w')

    # mpl_draw_topomesh(all_smooth_wall_topomesh, figure, degree=1, color='k', alpha=0.2, linewidth=1)
    # figure.gca().imshow(pi_img[coords].reshape(xx.shape),cmap="Oranges",vmin=0,vmax=50000,extent=extent)
    figure.gca().imshow(pin_img[coords].reshape(xx.shape),cmap="Greens",vmin=10000,vmax=60000,extent=extent,interpolation='none',alpha=1)

    for label in cell_pin1_vectors.keys():
        cell_center = cell_centers[label]
        cell_pin1_vector = 2.*microscope_orientation*cell_pin1_vectors[label]
        # figure.gca().text(cell_center[0],cell_center[1],str(label),zorder=10,size=12,horizontalalignment='center',verticalalignment='top')
        if np.all(cell_pin1_vector):
            # figure.gca().scatter(microscope_orientation*p_img.image_property('barycenter')[l][0],microscope_orientation*p_img.image_property('barycenter')[l][1])
            figure.gca().arrow(cell_center[0]-0.75*cell_pin1_vector[0],cell_center[1]-0.75*cell_pin1_vector[1],cell_pin1_vector[0],cell_pin1_vector[1],head_width=0.5,head_length=0.5,edgecolor='k',facecolor='k',linewidth=2,alpha=0.25)

    figure.gca().axis('off')
    figure.set_size_inches(32,32)
    figure.tight_layout()
    figure.savefig(dirname+"/"+filename+"/"+filename+"_L1_PIN1_cell_polarities.png")
