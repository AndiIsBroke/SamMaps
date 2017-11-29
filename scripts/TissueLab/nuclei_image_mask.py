import numpy as np
import pandas as pd

import openalea.container
from openalea.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image, read_tiff_image
from scipy.misc import imread as imread_2d

from openalea.image.serial.all import imsave
from openalea.image.spatial_image import SpatialImage

import os

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

image_filename = microscopy_dirname+"/RAW/"+filename
image_dict = read_czi_image(image_filename,channel_names=channel_names)

no_organ_filename = microscopy_dirname+"/TIF-No-organs/"+filename[:-4]+"-No-organs.tif"
if os.path.exists(no_organ_filename):
    no_organ_dict = read_tiff_image(no_organ_filename,channel_names=channel_names)

    image_mask = (image_dict[reference_name] == no_organ_dict[reference_name])
    voxelsize = image_dict[reference_name].voxelsize
    img_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_mask.inr.gz"
    imsave(img_file,SpatialImage((255*image_mask).astype(np.uint8),voxelsize=voxelsize))

png_filename =  microscopy_dirname+"/max_projection_intensity&masks/"+filename[:-4]+"_CH2_iso_MIP6000.png"
mask_png_filename =  microscopy_dirname+"/max_projection_intensity&masks/"+filename[:-4]+"_CH2_iso_MIP6000mask.png"

if os.path.exists(mask_png_filename):
    png_img = imread_2d(png_filename)
    mask_png_img = imread_2d(mask_png_filename)
    mask_png_img = (mask_png_img == png_img)&(png_img != 0)

    image_mask = np.tile(mask_png_img[:,:,np.newaxis],(1,1,image_dict[reference_name].shape[2]))

    img_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_projection_mask.inr.gz"
    imsave(img_file,SpatialImage((255*image_mask).astype(np.uint8),voxelsize=voxelsize))
