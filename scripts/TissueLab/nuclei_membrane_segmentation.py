import numpy as np
import pandas as pd

# from openalea.image.serial.all import imread
# from openalea.image.serial.all import imsave
# from openalea.image.spatial_image import SpatialImage

try:
    from timagetk.util import data_path
    from timagetk.io import imread
    from timagetk.io import imsave
    from timagetk.components import SpatialImage
    from timagetk.algorithms import linearfilter
    from timagetk.algorithms import morpho
    from timagetk.plugins import linear_filtering
    from timagetk.plugins import morphology
    from timagetk.plugins import h_transform
    from timagetk.plugins import region_labeling
    from timagetk.plugins import segmentation
    from timagetk.plugins.segmentation import seeded_watershed
    from timagetk.plugins import labels_post_processing
except ImportError:
    raise ImportError('Import Error')

from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.mesh.property_topomesh_io import read_ply_property_topomesh
from openalea.mesh.utils.pandas_tools import topomesh_to_dataframe

from openalea.oalab.colormap.colormap_def import load_colormaps

from openalea.container import array_dict
from timagetk.algorithms import isometric_resampling

import os
import warnings


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



# ---------------------------
# Guillaume
# ---------------------------
# dirname = "/Users/gcerutti/Desktop/WorkVP/"
# image_dirname = dirname+"SamMaps/nuclei_images"
# microscopy_dirname = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/microscopy/20170226_qDII-CLV3-DR5/"
# filename = 'qDII-CLV3-DR5-PI-LD-SAM11-T0.czi'

# ---------------------------
# Carlos
# ---------------------------
# dirname = "/home/carlos/"
# image_dirname = "/media/carlos/DONNEES/Documents/CNRS/SamMaps/nuclei_images"
# microscopy_dirname = "/media/carlos/DONNEES/Documents/CNRS/Microscopy/LSM710/20171110 MS-E35 LD qDII-CLV3-PIN1-PI/"
# filename = 'qDII-CLV3-PIN1-PI-E35-LD-SAM4-T5.czi'

# ---------------------------
# Marie
# ---------------------------
dirname = "/home/marie"
image_dirname = "/home/marie/Carlos/nuclei_images"
microscopy_dirname = dirname+"/Carlos/qDII-CLV3-PIN1-PI-E35-LD/SAM4/"
filename = 'qDII-CLV3-PIN1-PI-E35-LD-SAM4-T0.czi'

nomenclature_file = dirname + "/SamMaps/nomenclature.csv"
nomenclature_data = pd.read_csv(nomenclature_file,sep=';')[:-1]
nomenclature_names = dict(zip(nomenclature_data['Name'],nomenclature_data['Nomenclature Name']))

reference_name = 'TagBFP'
membrane_name = 'PI'
microscope_orientation = -1

membrane_image_filename = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_"+membrane_name+".inr.gz"
membrane_img = imread(membrane_image_filename)

mask_filename = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_mask.inr.gz"
mask_img = imread(mask_filename)
membrane_img[mask_img == 0] = 0

# world.add(membrane_img,'membrane_image',colormap='invert_grey')
#world['membrane_image']['intensity_range'] = (5000,30000)

topomesh_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_nuclei_detection_topomesh_corrected.ply"
topomesh = read_ply_property_topomesh(topomesh_file)
positions = topomesh.wisp_property('barycenter',0)
positions = array_dict(microscope_orientation*positions.values(),positions.keys())

# Create a seed image fro the nuclei barycenters:
seed_img = seed_image_from_points(membrane_img.shape,membrane_img.voxelsize,positions,background_label=0)

# Add the "background seed":
background_threshold = 2000.
smooth_img_bck = linearfilter(membrane_img, param_str_2 = '-x 0 -y 0 -z 0 -sigma 3.0')
background_img = (smooth_img_bck<background_threshold).astype(np.uint16)
for it in xrange(10):
    background_img = morphology(background_img, param_str_2 = '-operation erosion -iterations 10')
seed_img += background_img
seed_img = SpatialImage(seed_img,voxelsize=membrane_img.voxelsize)
#world.add(seed_img,'seed_image',colormap='glasbey',alphamap='constant',bg_id=0)
segmented_filename = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_corrected_nuclei_seed.inr"
imsave(segmented_filename,seed_img)

seed_img = isometric_resampling(seed_img, option='label')

std_dev = 2.0
membrane_img = isometric_resampling(membrane_img)
vxs = membrane_img.voxelsize

try:
    from equalization import z_slice_contrast_stretch
    from slice_view import slice_view

except:
    import sys
    sys.path.append('/home/marie/SamMaps/scripts/TissueLab/')
    from equalization import z_slice_contrast_stretch
    from slice_view import slice_view


rescaling = True
suffix = ""
if rescaling:
    membrane_img = z_slice_contrast_stretch(membrane_img, pc_max=99)
    membrane_img = SpatialImage(membrane_img,voxelsize=vxs)
    suffix += "_contrast_stretch"

smooth_img = linear_filtering(membrane_img, std_dev=std_dev,
                                  method='gaussian_smoothing')
smooth_img = SpatialImage(smooth_img, voxelsize=vxs)

x_sh, y_sh, z_sh = smooth_img.shape
pc_min=2
pc_max=99
title = "Contrast stretched (z-slice): {}pc.-{}pc + gaussian smoothing: std_dev:{}.".format(pc_min, pc_max, std_dev)
filename_smooth = filename[:-4]+"_z_contrast_stretch_{}-{}pc, gaussian smoothing_std_dev_{}.png".format(pc_min, pc_max,std_dev)
slice_view(smooth_img, x_sh/2, y_sh/2, z_sh/2, title, microscopy_dirname + filename_smooth)

# world.add(smooth_img,'membrane_image_adapthist_smooth',colormap='invert_grey')
# world['membrane_image_adapthist_smooth']['intensity_range'] = (5000,30000)

seg_img = seeded_watershed(smooth_img, seed_img, control='most')
segmented_filename = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_corrected_nuclei_seg{}.inr".format(suffix)
imsave(segmented_filename,seg_img)
# world.add(seg_img,'corrected_nuclei_segmented_image'+suffix,colormap='glasbey',alphamap='constant')
