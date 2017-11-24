import numpy as np
import pandas as pd

import openalea.container
from openalea.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image
from openalea.tissue_nukem_3d.nuclei_image_topomesh import nuclei_image_topomesh
from openalea.tissue_nukem_3d.nuclei_detection import compute_fluorescence_ratios
from openalea.tissue_nukem_3d.nuclei_mesh_tools import nuclei_layer

from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.mesh.utils.pandas_tools import topomesh_to_dataframe

from openalea.image.serial.all import imsave
from openalea.image.spatial_image import SpatialImage

from openalea.oalab.colormap.colormap_def import load_colormaps

from openalea.container import array_dict

import os

# world.clear()
from skimage import exposure
def equalize_adapthist(img):
    # Adaptive Equalization
    return np.array(exposure.equalize_adapthist(img, clip_limit=0.03)*(2**16)).astype(np.uint16)

def sl_equalize_adapthist(img):
    # Slice by slice equalization
    sh = img.get_shape()
    return np.array([equalize_adapthist(img[:,:,n]) for n in range(0, sh[2])]).transpose([1,2,0])

def contrast_stretch(img, pc_min=2, pc_max=99):
    # Contrast stretching
    pcmin = np.percentile(img, pc_min)
    pcmax = np.percentile(img, pc_max)
    return exposure.rescale_intensity(img, in_range=(pcmin, pcmax))

def sl_contrast_stretch(img, pc_min=2, pc_max=99):
    # Slice by slice contrast stretching
    sh = img.get_shape()
    return np.array([contrast_stretch(img[:,:,n], pc_min, pc_max) for n in range(0, sh[2])]).transpose([1,2,0])

filename = 'qDII-CLV3-PIN1-PI-E35-LD-SAM4-T0.czi'

# Carlos
# dirname = "/home/carlos/"
# image_dirname = "/media/carlos/DONNEES/Documents/CNRS/SamMaps/nuclei_images"
# microscopy_dirname = "/media/carlos/DONNEES/Documents/CNRS/Microscopy/LSM710/20171110 MS-E35 LD qDII-CLV3-PIN1-PI/"

# Marie
dirname = "/home/marie/"
image_dirname = dirname+"Carlos/nuclei_images"
microscopy_dirname = dirname+"Carlos/qDII-CLV3-PIN1-PI-E35-LD/SAM4/"

nomenclature_file = dirname + "SamMaps/nomenclature.csv"
nomenclature_data = pd.read_csv(nomenclature_file,sep=';')[:-1]
nomenclature_names = dict(zip(nomenclature_data['Name'],nomenclature_data['Nomenclature Name']))

reference_name = 'TagBFP'
channel_names = ['DIIV','PIN1','PI','TagBFP','CLV3']
signal_names = channel_names
compute_ratios = [n in ['DIIV'] for n in signal_names]
microscope_orientation = -1

image_filename = microscopy_dirname+"/RAW/"+filename
image_dict = read_czi_image(image_filename,channel_names=channel_names)

no_organ_filename = microscopy_dirname+"/TIF-No-organs/"+filename[:-4]+"-No-organs.tif"
if os.path.exists(no_organ_filename):
    no_organ_dict = read_tiff_image(no_organ_filename,channel_names=channel_names)
    voxelsize = image_dict[reference_name].voxelsize
    for channel in channel_names:
        image_dict[channel] = SpatialImage(no_organ_dict[channel],voxelsize=voxelsize)

# detection step
for rescaling in [False, True]:
    reference_img = image_dict[reference_name]
    suffix = ""
    if rescaling:
        reference_img = sl_equalize_adapthist(reference_img)
        suffix += "_AdaptHistEq"
    # world.add(reference_img,'nuclei_image',colormap='invert_grey',voxelsize=microscope_orientation*np.array(image_dict[reference_name].voxelsize))
    # world['nuclei_image']['intensity_range'] = (2000,20000)

    if 'PI' in channel_names:
        pi_img = image_dict['PI']
        # world.add(pi_img,'membrane_image',colormap='Reds',voxelsize=microscope_orientation*np.array(image_dict[reference_name].voxelsize))
        # world['membrane_image']['intensity_range'] = (5000,30000)

    if not os.path.exists(image_dirname+"/"+nomenclature_names[filename]):
        os.makedirs(image_dirname+"/"+nomenclature_names[filename])

    for i_channel, channel_name in enumerate(channel_names):
        # raw_img_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_"+channel_name+"_raw.inr.gz"
        img_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_"+channel_name+".inr.gz"
        imsave(img_file,image_dict[channel_name])

    topomesh_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_nuclei_detection_topomesh{}.ply".format(suffix)
    if os.path.exists(topomesh_file):
        topomesh = read_ply_property_topomesh(topomesh_file)
    else:
        # topomesh, surface_topomesh = nuclei_image_topomesh(image_dict,threshold=1000,reference_name=reference_name,microscope_orientation=microscope_orientation,signal_names=signal_names,compute_ratios=compute_ratios,subsampling=4,return_surface=True)
        topomesh = nuclei_image_topomesh(image_dict,threshold=1000,reference_name=reference_name,microscope_orientation=microscope_orientation,signal_names=signal_names,compute_ratios=compute_ratios,subsampling=4,surface_subsampling=6)
        save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,signal_names+['layer']),(1,[]),(2,[]),(3,[])]),color_faces=False)


# Correction step:
for rescaling in [False, True]:
    reference_img = image_dict[reference_name]
    suffix = ""
    if rescaling:
        reference_img = sl_equalize_adapthist(reference_img)
        suffix += "_AdaptHistEq"

    topomesh_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_nuclei_detection_topomesh{}.ply".format(suffix)
    topomesh = read_ply_property_topomesh(topomesh_file)

    world.add(reference_img,'nuclei_image'+suffix,colormap='invert_grey',voxelsize=microscope_orientation*np.array(image_dict[reference_name].voxelsize))
    world['nuclei_image']['intensity_range'] = (2000,20000)

    L1_cells = np.array(list(topomesh.wisps(0)))[topomesh.wisp_property('layer',0).values()==1]
    L1_topomesh = vertex_topomesh(topomesh.wisp_property('barycenter',0).values(L1_cells))

    non_L1_cells = np.array(list(topomesh.wisps(0)))[topomesh.wisp_property('layer',0).values()!=1]

    world.add(L1_topomesh,"detected_nuclei")
    world["detected_nuclei_vertices"]["polydata_colormap"] = load_colormaps()['jet']
    world["detected_nuclei_vertices"]["display_colorbar"] = False
    world['nuclei_image']['x_plane_position'] = 0
    world['nuclei_image']['y_plane_position'] = 0
    world['nuclei_image']['z_plane_position'] = 10
    world['nuclei_image']['cut_planes_alpha'] = 0.5
    # End of automatic detection

    # Start of post-edition
    edited_L1_points = np.array(world["detected_nuclei_vertices"].data.points.values())
    non_L1_points = topomesh.wisp_property('barycenter',0).values(non_L1_cells)

    edited_positions = dict(zip(np.arange(len(edited_L1_points)+len(non_L1_points))+2,microscope_orientation*np.concatenate([edited_L1_points,non_L1_points])))
    edited_layer = dict(zip(np.arange(len(edited_L1_points)+len(non_L1_points))+2,np.concatenate([np.ones_like(edited_L1_points[:,0]),np.zeros_like(non_L1_points[:,0])])))

    signal_values = {}
    for signal_name, compute_ratio in zip(signal_names,compute_ratios):
        signal_img = image_dict[signal_name]

        ratio_img = reference_img if compute_ratio else np.ones_like(reference_img)
        signal_values[signal_name] = compute_fluorescence_ratios(ratio_img,signal_img,edited_positions)


    edited_positions = array_dict(np.array(edited_positions.values())*microscope_orientation,edited_positions.keys()).to_dict()

    edited_topomesh = vertex_topomesh(edited_positions)
    for signal_name in signal_names:
        edited_topomesh.update_wisp_property(signal_name,0,signal_values[signal_name])

    edited_topomesh.update_wisp_property('layer',0,edited_layer)

    topomesh_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_nuclei_detection_topomesh_corrected{}.ply".format(suffix)
    save_ply_property_topomesh(edited_topomesh,topomesh_file,properties_to_save=dict([(0,signal_names+['layer']),(1,[]),(2,[]),(3,[])]),color_faces=False)

    df = topomesh_to_dataframe(edited_topomesh,0)
    df.to_csv(image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_signal_data_corrected{}.csv".format(suffix))
