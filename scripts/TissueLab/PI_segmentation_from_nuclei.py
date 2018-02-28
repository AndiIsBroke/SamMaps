import os
import numpy as np
import pandas as pd
import scipy.ndimage as nd

from openalea.container import array_dict
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.oalab.colormap.colormap_def import load_colormaps
from openalea.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image, read_tiff_image
from openalea.tissue_nukem_3d.nuclei_segmentation import seed_image_from_points

from timagetk.algorithms import isometric_resampling
from timagetk.components import SpatialImage
from timagetk.components import imsave
from timagetk.plugins import linear_filtering, morphology, h_transform, region_labeling, segmentation, registration

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
    dirname = "/data/Meristems/Carlos/PIN_maps/"
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
    dirname = "/projects/SamMaps/"
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/TissueLab/')

from equalization import z_slice_equalize_adapthist
from nomenclature import get_nomenclature_name
from nomenclature import get_nomenclature_channel_fname
from nomenclature import get_nomenclature_segmentation_name

# world.clear()

import time as t
start = t.time()

# filename = 'qDII-CLV3-PIN1-PI-E35-LD-SAM4-T0.czi'
filename = sys.argv[1]
try:
    sys.argv[2] == 'force'
except:
    force = False
else:
    force = True

raw_czi_path, raw_czi_fname = os.path.split(filename)

# # Carlos:
# dirname = "/home/carlos/DONNEES/Documents/CNRS/"
# image_dirname = dirname + "SamMaps/nuclei_images/"
# raw_czi_path = dirname + "Microscopy/LSM710/20171110 MS-E35 LD qDII-CLV3-PIN1-PI/"
# filename = raw_czi_path + filename

# Jo:
dirname = "/data/Meristems/Carlos/"
image_dirname = dirname+"PIN_maps/nuclei_images/"

# - NOMENCLATURE file defining naming conventions:
nomenclature_file = dirname + "SamMaps/nomenclature.csv"

# - Define CZI channel names, the microscope orientation and nuclei and membrane channel names:
channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
microscope_orientation = -1  # inverted microscope!
nuclei_ch_name = 'TagBFP'
membrane_ch_name = 'PI'

# - Defines segmented filename:
seg_path_suffix, seg_img_fname = get_nomenclature_segmentation_name(raw_czi_fname, nomenclature_file, membrane_ch_name)
if os.path.exists(image_dirname + seg_path_suffix + seg_img_fname) and not force:
    print "A segmentation file '{}' already exists, aborting now.".format(seg_img_fname)
    sys.exit(0)

# - Define and read CZI file for defined channel names:
czi_img = read_czi_image(filename, channel_names=channel_names)

# - Defines the "membrane image" used for cell-based segmentation:
memb_img = czi_img[membrane_ch_name] - czi_img['CLV3']
memb_img[czi_img[membrane_ch_name] <= czi_img['CLV3']] = 0
# substract the 'CLV3' signal from the 'PI' since it might have leaked:
vxs = np.array(memb_img.voxelsize)
memb_img = SpatialImage(memb_img, voxelsize=vxs)
# - Display the "membrane image":
# world.add(memb_img,'{}_channel'.format(membrane_ch_name), colormap='invert_grey', voxelsize=microscope_orientation*vxs)
# world['{}_channel'.format(membrane_ch_name)]['intensity_range'] = (-1, 2**16)

nom_names = get_nomenclature_name(nomenclature_file)
# - Read CSV file containing barycenter positions:
signal_file = image_dirname + nom_names[raw_czi_fname] + '/' + nom_names[raw_czi_fname] + "_signal_data.csv"
signal_data = pd.read_csv(signal_file, sep=',')[:-1]
x, y, z = signal_data.center_x, signal_data.center_y, signal_data.center_z
bary_dict = {k: np.array([x[k], y[k], z[k]])*microscope_orientation for k in x.keys()}


# - Adaptative Histogram Equalization:
print "Adaptative Histogram Equalization...",
memb_img = z_slice_equalize_adapthist(memb_img)

# - Performs isometric resampling of MEMBRANE image:
memb_img = isometric_resampling(memb_img)
iso_vxs = np.array(memb_img.get_voxelsize())
iso_shape = memb_img.get_shape()
# - Display the isometric version of the "equalized membrane image":
# world.add(memb_img,'{}_channel_equalized_isometric'.format(membrane_ch_name), colormap='invert_grey', voxelsize=microscope_orientation*iso_vxs)
# world['{}_channel_equalized_isometric'.format(membrane_ch_name)]['intensity_range'] = (-1, 2**16)

background = 1
# - If 0 or background (usually 1) is a key in the list of barycenters, we change
# them since they are "special values":
unauthorized_labels = [0, background]
for l in unauthorized_labels:
    if l in bary_dict.keys():
        bary_dict[np.max(bary_dict.keys())+1] = bary_dict.pop(l)


# - NUCLEI positions edition:
# -- Create a vertex_topomesh from seeds positions:
# detected_topomesh = vertex_topomesh({k: v * mo for k, v in bary_dict.items()})
# world.add(detected_topomesh, "seeds_topomesh")
# world["seeds_topomesh_vertices"]["polydata_colormap"] = load_colormaps()['jet']
# world["seeds_topomesh_vertices"]["display_colorbar"] = False
# world['{}_channel_equalized_isometric'.format(membrane_ch_name)]['x_plane_position'] = 0
# world['{}_channel_equalized_isometric'.format(membrane_ch_name)]['y_plane_position'] = 0
# world['{}_channel_equalized_isometric'.format(membrane_ch_name)]['z_plane_position'] = 10
# world['{}_channel_equalized_isometric'.format(membrane_ch_name)]['cut_planes_alpha'] = 0.5


# - Create a seed image from the list of NUCLEI barycenters:
# -- Construct the seed image with a 'zero' background, since "true background" will be added later:
seed_img = seed_image_from_points(iso_shape, iso_vxs, bary_dict, 1.0, 0)

# -- Add background position:
# --- Detect background from membrane intensity image:
background_threshold = 2000.
smooth_img_bck = linear_filtering(memb_img, std_dev=3.0, method='gaussian_smoothing')
background_img = (smooth_img_bck < background_threshold).astype(np.uint16)
for it in xrange(15):
    background_img = morphology(background_img, param_str_2 = '-operation erosion -iterations 10')

# --- Detect small regions defined as background and remove them:
connected_background_components, n_components = nd.label(background_img)
components_area = nd.sum(np.ones_like(connected_background_components), connected_background_components, index=np.arange(n_components)+1)
largest_component = (np.arange(n_components)+1)[np.argmax(components_area)]
background_img = (connected_background_components == largest_component).astype(np.uint16)

# --- Finaly add the background and make a SpatialImage:
seed_img[background_img==background] = background
seed_img = SpatialImage(seed_img, voxelsize=iso_vxs)

del smooth_img_bck, background_img
seeds = np.unique(seed_img)
nb_seeds = len(seeds)

# world.add(seed_img, "seed_image", colormap="glasbey", alphamap="constant", voxelsize=microscope_orientation*iso_vxs, bg_id=background)


# - Performs automatic seeded watershed using previously created seed image:
# -- Performs Gaussian smoothing:
std_dev = 1.0
smooth_img = linear_filtering(memb_img, std_dev=std_dev, method='gaussian_smoothing')
# -- Performs the seeded watershed segmentation:
print "\n# - Seeded watershed using seed EXPERT seed positions..."
seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed')
imsave(image_dirname + seg_img_fname, seg_im)

labels = np.unique(seg_im)
nb_labels = len(labels)
print "Found {} labels out of {} seeds!".format(nb_labels, nb_seeds)

# world.add(seg_im, "seg_image", colormap="glasbey", alphamap="constant",voxelsize=microscope_orientation*iso_vxs, bg_id=background)

end = int(np.ceil(t.time() - start))
print "It took {}min {}s to performs all operations on {}!".format(end/60, end%60, raw_czi_fname)
