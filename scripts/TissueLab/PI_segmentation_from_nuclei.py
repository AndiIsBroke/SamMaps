import os
import numpy as np
import pandas as pd

from openalea.container import array_dict
from openalea.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image, read_tiff_image
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.oalab.colormap.colormap_def import load_colormaps

world.clear()

filename = 'qDII-CLV3-PIN1-PI-E35-LD-SAM4-T0.czi'

# # Carlos
# dirname = "/home/carlos/"
# image_dirname = "/media/carlos/DONNEES/Documents/CNRS/SamMaps/nuclei_images/"
# microscopy_dirname = "/media/carlos/DONNEES/Documents/CNRS/Microscopy/LSM710/20171110 MS-E35 LD qDII-CLV3-PIN1-PI/"

# Jo
dirname = "/data/Meristems/Carlos/"
image_dirname = dirname+"PIN_maps/nuclei_images/"
microscopy_dirname = dirname+"PIN_maps/microscopy/20171110 MS-E35 LD qDII-CLV3-PIN1-PI/"

# - Read NOMENCLATURE file defining naming conventions:
nomenclature_file = dirname + "SamMaps/nomenclature.csv"
nomenclature_data = pd.read_csv(nomenclature_file, sep=';')[:-1]
nomenclature_names = dict(zip(nomenclature_data['Name'], nomenclature_data['Nomenclature Name']))

# - Define CZI channel names, the microscope orientation and nuclei and membrane channel names:
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
nuclei_ch_name = 'TagBFP'
membrane_ch_name = 'PI'

# - Define and read CZI file for defined channel names:
img_fname = microscopy_dirname+"RAW/"+filename
img_dict = read_czi_image(img_fname, channel_names=channel_names)

# - If masks have been defined, we use them to crop the ROI:
no_organ_filename = microscopy_dirname+"TIF-No-organs/"+filename[:-4]+"-No-organs.tif"
if os.path.exists(no_organ_filename):
    no_organ_dict = read_tiff_image(no_organ_filename, channel_names=channel_names)
    voxelsize = img_dict[membrane_ch_name].voxelsize
    for channel in channel_names:
        img_dict[channel] = SpatialImage(no_organ_dict[channel], voxelsize=voxelsize)

# - Defines the "membrane image" used for cell-based segmentation:
memb_img = img_dict[membrane_ch_name]
vxs = microscope_orientation * np.array(memb_img.voxelsize)
world.add(memb_img,'{}_channel'.format(membrane_ch_name), colormap='invert_grey', voxelsize=vxs)
world['{}_channel'.format(membrane_ch_name)]['intensity_range'] = (5000, 2**16)

# - Create (if necessary) the folder structure according to NOMENCLATURE:
if not os.path.exists(image_dirname + nomenclature_names[filename]):
    os.makedirs(image_dirname + nomenclature_names[filename])

# - Create (if necessary) the (compressed) inr image of each defined channel:
for n_ch, ch_name in enumerate(channel_names):
    # raw_img_file = image_dirname+"/"+nomenclature_names[filename]+"/"+nomenclature_names[filename]+"_"+ch_name+"_raw.inr.gz"
    img_file = image_dirname + nomenclature_names[filename] + "/" + nomenclature_names[filename] + "_" + ch_name + ".inr.gz"
    if not os.path.exists(img_file):
        imsave(img_file, img_dict[ch_name])
