import numpy as np
import scipy.ndimage as nd
from scipy.cluster.vq import vq
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
from matplotlib import patches as patch


from openalea.image.serial.all import imread
from openalea.image.spatial_image import SpatialImage
from timagetk.components import SpatialImage as TissueImage
from timagetk.algorithms import resample_isotropic

from openalea.cellcomplex.property_topomesh.property_topomesh_io import read_ply_property_topomesh
from openalea.cellcomplex.property_topomesh.property_topomesh_creation import vertex_topomesh

from openalea.container import array_dict

from vplants.tissue_nukem_3d.epidermal_maps import compute_local_2d_signal
from vplants.tissue_nukem_3d.epidermal_maps import nuclei_density_function

from vplants.tissue_analysis.property_spatial_image import PropertySpatialImage

from time import time as current_time
from copy import deepcopy

import os
import logging
logging.getLogger().setLevel(logging.INFO)

# -- Channel name for the 'membrane labelling' image:
DEF_MEMB_CH = 'PI'
# -- Channel name for the 'membrane-targetted signal' image:
DEF_SIG_CH = 'PIN1'
# -- Channel name for the 'clearing' image:
DEF_CLEAR_CH = 'TagBFP'

# PARAMETERS:
# -----------
import argparse
parser = argparse.ArgumentParser(description='Performs quantification of membrane localized signal.')
# positional arguments:
parser.add_argument('membrane_im', type=str,
                    help="file containing the 'membrane labelling' channel.")
parser.add_argument('signal_im', type=str,
                    help="file containing the 'membrane-targetted signal of interest'.")
parser.add_argument('segmented_im', type=str,
                    help="segmented image corresponding to the 'membrane labelling' channel")
parser.add_argument('wall_quantif_csv', type=str,
                    help="CSV file corresponding to wall-based quantification of signal image")

# optional arguments:
parser.add_argument('--clearing_im', type=str, default=None,
                    help="image used to clear the , None by default")
parser.add_argument('--clearing_ch_name', type=str, default=DEF_CLEAR_CH,
                    help="channel name for the 'clearing' channel, '{}' by default".format(DEF_CLEAR_CH))

parser.add_argument('--membrane_ch_name', type=str, default=DEF_MEMB_CH,
                    help="channel name for the 'membrane labelling' channel, '{}' by default".format(DEF_MEMB_CH))
parser.add_argument('--signal_ch_name', type=str, default=DEF_SIG_CH,
                    help="channel name for the 'membrane-targetted signal of interest', '{}' by default".format(DEF_SIG_CH))
parser.add_argument('--bounding_box', type=int, nargs='+', default=None,
                    help="if given, used to crop the image around these voxel coordinates, use '0' for the beginning and '-1' for the end")
parser.add_argument('--resampling_voxelsize', type=int, default=None,
                    help="if given, resampling factor used on segemented image")

# - Clear the world ???
if "world" in dir():
    world.clear()


args = parser.parse_args()
img_dict = {}
# - Variables definition from mandatory arguments parsing:
# -- Membrane labelling signal image:
memb_im_fname = args.membrane_im
print "\n\n# - Reading membrane labelling signal image file {}...".format(memb_im_fname)
img_dict[args.membrane_ch_name] =  = read_image(memb_im_fname)
print "Done."
# -- Membrane-targetted signal image:
sig_im_fname = args.signal_im
print "\n\n# - Reading membrane-targetted signal image file {}...".format(sig_im_fname)
img_dict[args.signal_ch_name] = read_image(sig_im_fname)
print "Done."
# -- Segmented images:
seg_im_fname = args.segmented_im
print "\n\n# - Reading segmented image file {}...".format(seg_im_fname)
seg_im = read_image(seg_im_fname)
print "Done."
# -- CSV filename:
csv_data = args.wall_quantif_csv

# - Variables definition from optional arguments parsing:
# -- Channel names:
memb_ch = args.membrane_ch_name
sig_ch = args.signal_ch_name

# -- Clearing channel if any:
if args.clearing_im is not None:
    img_dict[args.clearing_ch_name] = read_image(args.clearing_im)
    pi_img = TissueImage(np.maximum(0, img_dict[memb_ch].astype(np.int32)-img_dict[args.clearing_ch_name].astype(np.int32)).astype(np.uint16), voxelsize=img_dict[memb_ch].voxelsize)
else:
    pi_img = TissueImage(img_dict[memb_ch].astype(np.uint16), voxelsize=img_dict[memb_ch].voxelsize)

pin_img = img_dict[sig_ch]

# -- Resampling factor:
resampling_voxelsize = args.resampling_voxelsize
try:
    assert resampling_voxelsize > 1
except AssertionError:
    raise ValueError("Optional parameter 'resampling_voxelsize' should be greater than 1, got {}!".format(resampling_voxelsize))

# -- Other variables, not handled by argument parser (yet):
microscope_orientation=-1
target_edge_length = 1.
all_walls = False
# all_walls = True

if "world" in dir():
    # -- Add cleared membrane labelled image:
    world.add(pi_img, 'cleared_membrane_image', voxelsize=microscope_orientation*np.array(pi_img.voxelsize), colormap='Oranges')
    # -- Add original membrane labelled image:
    world.add(img_dict[memb_ch], 'membrane_image', voxelsize=microscope_orientation*np.array(img_dict[memb_ch].voxelsize), colormap='Reds')
    # -- Add original membrane-targetted signal image:
    world.add(pin_img, 'membrane-targetted_signal_image', voxelsize=microscope_orientation*np.array(pin_img.voxelsize), colormap='Greens')

# - Loading & resampling segmented image:
start_time = current_time()
if resampling_voxelsize is not None:
    seg_img = resample_isotropic(TissueImage(seg_img, voxelsize=seg_img.voxelsize), voxelsize=resampling_voxelsize, option='label')
if "world" in dir():
    world.add(seg_img, 'segmented_image', voxelsize=microscope_orientation*np.array(seg_img.voxelsize), colormap='glasbey', alphamap='constant')
logging.info("--> Loading & resampling segmented image ["+str(current_time() - start_time)+" s]")

# - Computing cellular features:
start_time = current_time()
p_img = PropertySpatialImage(seg_img, ignore_cells_at_stack_margins=False)
p_img.compute_default_image_properties()
if all_walls:
    p_img.update_image_property('layer', dict(zip(p_img.labels,[1 for l in p_img.labels])))
# p_img.compute_cell_meshes(sub_factor=1)
logging.info("--> Computing image cell layers ["+str(current_time() - start_time)+" s]")

# - Load CSV file:
wall_csv_data = pd.read_csv(csv_data)
# -- get list of wall defining labelpairs:
wall_cells = wall_csv_data[['right_label','left_label']].values
# -- symmetric labelpairs dict (ie. dict[i,j]==dict[j,i]):
wall_areas = dict(zip([(l,r) for (l,r) in wall_cells], wall_csv_data['wall_area'].values))
wall_centers = dict(zip([(l,r) for (l,r) in wall_cells], wall_csv_data[['wall_center_'+dim for dim in ['x','y','z']]].values))
wall_normals = dict(zip([(l,r) for (l,r) in wall_cells], wall_csv_data[['wall_normal_'+dim for dim in ['x','y','z']]].values))
# -- oriented labelpairs dict (ie. dict[i,j]==dict[j,i]):
pin1_orientations = dict(zip([(l,r) for (l,r) in wall_cells], wall_csv_data[sig_ch+'_orientation'].values))
pin1_intensities = dict(zip([(l,r) for (l,r) in wall_cells], wall_csv_data[sig_ch+'_signal'].values))
# pin1_shifts = dict(zip([(l,r) for (l,r) in wall_cells],wall_csv_data['pin1_shift'].values))
pin1_intensities_left = dict(zip([(l,r) for (l,r) in wall_cells], wall_csv_data[sig_ch+'_signal_left'].values))
pin1_intensities_right = dict(zip([(l,r) for (l,r) in wall_cells] wall_csv_data[sig_ch+'_signal_right'].values))

wall_pin1_vectors={}
for left_label, right_label in wall_cells:
    wall_pin1_vectors[(left_label, right_label)] = pin1_orientations[(left_label, right_label)] * wall_normals[(left_label, right_label)]


# cell_centers = dict(zip(np.unique(wall_cells),[p_img.image_property('barycenter')[c][:2] for c in np.unique(wall_cells)]))
cell_centers = dict(zip(np.unique(wall_cells), np.transpose([nd.sum(wall_csv_data['wall_center_'+dim].values ,wall_cells[:,0], index=np.unique(wall_cells)) for dim in ['x','y']])/nd.sum(np.ones_like(wall_cells[:,0]),wall_cells[:,0],index=np.unique(wall_cells))[:,np.newaxis]))

weights = np.ones_like(wall_cells[:,0]).astype(float)
weights *= np.array([wall_areas[(l,r)] for (l,r) in wall_cells])
# weights *= np.array([pin1_intensities[(l,r)] for (l,r) in wall_cells])
weights *= np.array([np.abs(pin1_intensities_left[(l,r)]-pin1_intensities_right[(l,r)]) for (l,r) in wall_cells])

wall_weights = dict(zip([(l,r) for (l,r) in wall_cells],weights))


cell_pin1_vectors = dict(zip(np.unique(wall_cells),np.transpose([nd.sum(np.array([wall_weights[(l,r)]*wall_pin1_vectors[(l,r)][k] for (l,r) in wall_cells]),wall_cells[:,0],index=np.unique(wall_cells)) for k in xrange(2)])/nd.sum(np.array([wall_weights[(l,r)] for (l,r) in wall_cells]),wall_cells[:,0],index=np.unique(wall_cells))[:,np.newaxis]))

cell_labels = p_img.image_property('barycenter').keys()[p_img.image_property('layer').values()==1]
X = microscope_orientation*p_img.image_property('barycenter').values()[:,0][p_img.image_property('layer').values()==1]
Y = microscope_orientation*p_img.image_property('barycenter').values()[:,1][p_img.image_property('layer').values()==1]
Z = microscope_orientation*p_img.image_property('barycenter').values()[:,2][p_img.image_property('layer').values()==1]

# xx,yy = np.meshgrid(np.linspace(0,(pi_img.shape[0]-1)*pi_img.voxelsize[0],pi_img.shape[0]),np.linspace(0,(pi_img.shape[1]-1)*pi_img.voxelsize[1],pi_img.shape[1]))
xx,yy = np.meshgrid(np.linspace(0,(pi_img.shape[0]-1)*microscope_orientation*pi_img.voxelsize[0],pi_img.shape[0]/2),np.linspace(0,(pi_img.shape[1]-1)*microscope_orientation*pi_img.voxelsize[1],pi_img.shape[1]/2))
extent = xx.max(),xx.min(),yy.min(),yy.max()
# extent = xx.min(),xx.max(),yy.max(),yy.min()
zz = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),Z)
coords = (np.transpose([xx,yy,zz],(1,2,0)))/(microscope_orientation*np.array(pi_img.voxelsize))
for k in xrange(3):
    coords[:,:,k] = np.maximum(np.minimum(coords[:,:,k],pi_img.shape[k]-1),0)
coords[np.isnan(coords)]=0
coords = coords.astype(int)
coords = tuple(np.transpose(np.concatenate(coords)))


seg_cell_topomesh = vertex_topomesh(dict(zip(range(len(X)),np.transpose([X,Y,Z]))))

# signal_filename = dirname+"/"+filename+"/"+filename+"_signal_data.csv"
signal_filename = ""
if os.path.exists(signal_filename):
    signal_data = pd.read_csv(signal_filename)
else:
    signal_data = None

if signal_data is not None:
    cell_radius = 7.5
    density_k = 0.55
    signal_X = signal_data[signal_data['layer']==1]['center_x'].values
    signal_Y = signal_data[signal_data['layer']==1]['center_y'].values
    signal_Z = signal_data[signal_data['layer']==1]['center_z'].values

    signal_cell_topomesh = vertex_topomesh(dict(zip(range(len(signal_X)),np.transpose([signal_X,signal_Y,signal_Z]))))

    auxin_map = compute_local_2d_signal(np.transpose([signal_X,signal_Y]),np.transpose([xx,yy],(1,2,0)),signal_data[signal_data['layer']==1]['DIIV'],cell_radius=cell_radius,density_k=density_k)
    density_map = nuclei_density_function(dict(zip(range(len(signal_X)),np.transpose([signal_X,signal_Y,np.zeros_like(signal_X)]))),cell_radius=cell_radius,k=density_k)(xx,yy,np.zeros_like(xx))
    confidence_map = nd.gaussian_filter(density_map,sigma=2.0)


from openalea.cellcomplex.property_topomesh.utils.matplotlib_tools import mpl_draw_topomesh

# all_smooth_wall_topomesh.update_wisp_property('barycenter',0,array_dict(microscope_orientation*all_smooth_wall_topomesh.wisp_property('barycenter',0).values(),all_smooth_wall_topomesh.wisp_property('barycenter',0).keys()))
all_smooth_wall_topomesh = read_ply_property_topomesh("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_smooth_L1_anticlinal_walls.ply")


import openalea.tissue_nukem_3d.utils.signal_luts
reload(openalea.tissue_nukem_3d.utils.signal_luts)
from vplants.tissue_nukem_3d.utils.signal_luts import *

figure = plt.figure(0)
figure.clf()
figure.patch.set_facecolor('w')

mpl_draw_topomesh(all_smooth_wall_topomesh, figure, degree=1, color='k', alpha=0.2, linewidth=1)
figure.gca().imshow(pi_img[coords].reshape(xx.shape),cmap="Oranges",vmin=0,vmax=50000,extent=extent,interpolation='none',alpha=1)
# figure.gca().imshow(pin_img[coords].reshape(xx.shape),cmap="Greens",vmin=10000,vmax=60000,extent=extent,interpolation='none',alpha=1)

if signal_data is not None:
    figure.gca().scatter(signal_X,signal_Y,color='r')
    # figure.gca().scatter(X,Y,color='b')

for cell_x,cell_y,label in zip(X,Y,cell_labels):
    cell_center = [cell_x,cell_y]
    # cell_pin1_vector = 2.*microscope_orientation*cell_pin1_vectors[label]
    figure.gca().text(cell_center[0],cell_center[1],str(label),zorder=10,size=12,color='b',horizontalalignment='center',verticalalignment='top')
    # if np.all(cell_pin1_vector):

figure.gca().scatter(np.array(cell_centers.values())[:,0],np.array(cell_centers.values())[:,1],color='m')
figure.gca().scatter(np.array(wall_centers.values())[:,0],np.array(wall_centers.values())[:,1],marker='s',color='m')

figure.gca().set_xlim(extent[0],extent[1])
figure.gca().set_ylim(extent[2],extent[3])

figure.gca().axis('off')
figure.set_size_inches(32,32)
figure.tight_layout()
figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_L1_wall_meshes.png")


figure = plt.figure(1)
figure.clf()
figure.patch.set_facecolor('w')
# figure.gca().imshow(pin_img[coords].reshape(xx.shape),cmap="green_hot",vmin=5000,vmax=60000,extent=extent,interpolation='none',alpha=1)
# figure.gca().imshow(np.log(pin_img[coords]+1).reshape(xx.shape),cmap="green_hot",vmin=10,vmax=12,extent=extent,interpolation='none',alpha=1)
figure.gca().imshow(np.exp(pin_img[coords]/(65535.)).reshape(xx.shape),cmap="green_hot",vmin=1.2,vmax=2.5,extent=extent,interpolation='none',alpha=1)
#
figure.gca().axis('off')
figure.set_size_inches(32,32)
figure.tight_layout()
# figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_L1_PIN1.png")
figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_L1_exp_PIN1.png")


figure = plt.figure(1)
figure.clf()
figure.patch.set_facecolor('w')

# mpl_draw_topomesh(all_smooth_wall_topomesh, figure, degree=1, color='k', alpha=0.2, linewidth=1)
figure.gca().imshow(pi_img[coords].reshape(xx.shape),cmap="Oranges",vmin=0,vmax=50000,extent=extent,interpolation='none',alpha=1)
figure.gca().imshow(pin_img[coords].reshape(xx.shape),cmap="Greens",vmin=10000,vmax=60000,extent=extent,interpolation='none',alpha=0.66)
# figure.gca().imshow(pin_img[coords].reshape(xx.shape),cmap="green_hot",vmin=0,vmax=60000,extent=extent,interpolation='none',alpha=1)

for (left_label, right_label), pin1_orientation in pin1_orientations.items():
    intensity = pin1_intensities[(left_label,right_label)]
    # shift = pin1_shifts[(left_label,right_label)]
    wall_center = wall_centers[(left_label,right_label)]
    wall_normal = wall_normals[(left_label,right_label)]
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

        figure.gca().arrow(wall_center[0]-0.75*pin1_vector[0],wall_center[1]-0.75*pin1_vector[1],pin1_vector[0],pin1_vector[1],head_width=1,head_length=1,edgecolor='k',facecolor=pin1_color,linewidth=1,alpha=np.abs(pin1_orientation))


for label in cell_pin1_vectors.keys():
    cell_center = cell_centers[label]
    # cell_pin1_vector = 2.*microscope_orientation*cell_pin1_vectors[label]
    figure.gca().text(cell_center[0],cell_center[1],str(label),zorder=10,size=12,horizontalalignment='center',verticalalignment='top')
    # if np.all(cell_pin1_vector):
        # figure.gca().scatter(microscope_orientation*p_img.image_property('barycenter')[l][0],microscope_orientation*p_img.image_property('barycenter')[l][1])
        # figure.gca().arrow(cell_center[0]-0.75*cell_pin1_vector[0],cell_center[1]-0.75*cell_pin1_vector[1],cell_pin1_vector[0],cell_pin1_vector[1],head_width=0.5,head_length=0.5,edgecolor='k',facecolor='k',linewidth=2,alpha=0.25)

figure.gca().axis('off')
# figure.set_size_inches(50,50)
figure.set_size_inches(32,32)
figure.tight_layout()
figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_L1_PIN1_polarities.png")


figure = plt.figure(2)
figure.clf()
figure.patch.set_facecolor('w')

# mpl_draw_topomesh(all_smooth_wall_topomesh, figure, degree=1, color='k', alpha=0.2, linewidth=1)
# figure.gca().imshow(pi_img[coords].reshape(xx.shape),cmap="Oranges",vmin=0,vmax=50000,extent=extent)
figure.gca().imshow(pin_img[coords].reshape(xx.shape),cmap="Greens",vmin=10000,vmax=60000,extent=extent,interpolation='none',alpha=1)

for label in cell_pin1_vectors.keys():
    cell_center = cell_centers[label]
    cell_pin1_vector = 2.*cell_pin1_vectors[label]
    # figure.gca().text(cell_center[0],cell_center[1],str(label),zorder=10,size=12,horizontalalignment='center',verticalalignment='top')
    if np.all(cell_pin1_vector):
        # figure.gca().scatter(microscope_orientation*p_img.image_property('barycenter')[l][0],microscope_orientation*p_img.image_property('barycenter')[l][1])
        figure.gca().arrow(cell_center[0]-0.75*cell_pin1_vector[0],cell_center[1]-0.75*cell_pin1_vector[1],cell_pin1_vector[0],cell_pin1_vector[1],head_width=0.5,head_length=0.5,edgecolor='k',facecolor='k',linewidth=2,alpha=0.25)

figure.gca().axis('off')
figure.set_size_inches(32,32)
figure.tight_layout()
figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_L1_PIN1_cell_polarities.png")


if signal_data is not None:
    figure = plt.figure(3)
    figure.clf()
    figure.patch.set_facecolor('w')

    figure.gca().imshow(img_dict['DIIV'][coords].reshape(xx.shape),cmap="Greens",vmin=10000,vmax=60000,extent=extent,interpolation='none',alpha=0)

    figure.gca().pcolormesh(xx,yy,auxin_map,cmap='lemon_hot',alpha=1,antialiased=True,vmin=0,vmax=0.5)
    figure.gca().contour(xx,yy,auxin_map,np.linspace(0,1,101),cmap='gray',alpha=0.1,linewidths=1,antialiased=True,vmin=-1,vmax=0)
    for a in xrange(16):
        figure.gca().contourf(xx,yy,confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)

    for label in cell_pin1_vectors.keys():
        cell_center = cell_centers[label]
        cell_pin1_vector = 3.*cell_pin1_vectors[label]
        # figure.gca().text(cell_center[0],cell_center[1],str(label),zorder=10,size=12,horizontalalignment='center',verticalalignment='top')
        if np.all(cell_pin1_vector):
            # figure.gca().scatter(microscope_orientation*p_img.image_property('barycenter')[l][0],microscope_orientation*p_img.image_property('barycenter')[l][1])
            figure.gca().arrow(cell_center[0]-0.75*cell_pin1_vector[0],cell_center[1]-0.75*cell_pin1_vector[1],cell_pin1_vector[0],cell_pin1_vector[1],head_width=0.5,head_length=0.5,edgecolor='r',facecolor='r',linewidth=4,alpha=1)

    figure.gca().axis('off')
    figure.set_size_inches(32,32)
    figure.tight_layout()
    figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_L1_Auxin_PIN1_polarities.png")




    aligned_signal_data = pd.read_csv(dirname+"/"+filename+"/"+filename+"_aligned_L1_normalized_signal_data.csv")

    signal_X = aligned_signal_data['center_x'].values
    signal_Y = aligned_signal_data['center_y'].values
    signal_Z = aligned_signal_data['center_z'].values

    aligned_signal_X = aligned_signal_data['aligned_x'].values
    aligned_signal_Y = aligned_signal_data['aligned_y'].values
    aligned_signal_Z = aligned_signal_data['aligned_z'].values


    from openalea.cellcomplex.property_topomesh.property_topomesh_creation import vertex_topomesh
    signal_cell_topomesh = vertex_topomesh(dict(zip(range(len(signal_X)),np.transpose([signal_X,signal_Y,signal_Z]))))


    from openalea.image.registration.registration import pts2transfo

    T = pts2transfo(np.transpose([signal_X,signal_Y,signal_Z]),np.transpose([aligned_signal_X,aligned_signal_Y,aligned_signal_Z]))

    cell_points = np.transpose([X,Y,Z,np.ones_like(X)])
    transformed_cell_points = np.transpose(np.dot(T,cell_points.T))


    wall_center_X =  np.array([wall_centers[(l,r)] for (l,r) in wall_cells])[:,0]
    wall_center_Y =  np.array([wall_centers[(l,r)] for (l,r) in wall_cells])[:,1]
    wall_center_Z =  np.array([wall_centers[(l,r)] for (l,r) in wall_cells])[:,2]
    wall_center_points = np.transpose([wall_center_X,wall_center_Y,wall_center_Z,np.ones_like(wall_center_X)])
    transformed_wall_center_points = np.transpose(np.dot(T,wall_center_points.T))

    aligned_wall_centers = dict(zip([(l,r) for (l,r) in wall_cells],transformed_wall_center_points[:,:3]))

    R = T[:3,:3]

    wall_normal_vectors = np.array([wall_normals[(l,r)] for (l,r) in wall_cells])
    transformed_wall_normal_vectors = np.transpose(np.dot(R,wall_normal_vectors.T))

    aligned_wall_normals = dict(zip([(l,r) for (l,r) in wall_cells],transformed_wall_normal_vectors))

    aligned_wall_pin1_vectors={}
    for left_label, right_label in wall_cells:
        aligned_wall_pin1_vectors[(left_label,right_label)] = pin1_orientations[(left_label,right_label)]*aligned_wall_normals[(left_label,right_label)]


    # cell_centers = dict(zip(np.unique(wall_cells),[p_img.image_property('barycenter')[c][:2] for c in np.unique(wall_cells)]))
    aligned_cell_centers = dict(zip(np.unique(wall_cells),np.transpose([nd.sum([aligned_wall_centers[(l,r)][i_dim] for (l,r) in wall_cells],wall_cells[:,0],index=np.unique(wall_cells)) for i_dim,dim in enumerate(['x','y'])])/nd.sum(np.ones_like(wall_cells[:,0]),wall_cells[:,0],index=np.unique(wall_cells))[:,np.newaxis]))
    # wall_weights = dict(zip([(l,r) for (l,r) in wall_cells],[pin1_intensities[(l,r)]*wall_areas[(l,r)] for (l,r) in wall_cells]))
    aligned_cell_pin1_vectors = dict(zip(np.unique(wall_cells),np.transpose([nd.sum(np.array([wall_weights[(l,r)]*aligned_wall_pin1_vectors[(l,r)][k] for (l,r) in wall_cells]),wall_cells[:,0],index=np.unique(wall_cells)) for k in xrange(2)])/nd.sum(np.array([wall_weights[(l,r)] for (l,r) in wall_cells]),wall_cells[:,0],index=np.unique(wall_cells))[:,np.newaxis]))


    r_max = 100
    aligned_xx, aligned_yy = np.meshgrid(np.linspace(-r_max,r_max,r_max+1),np.linspace(-r_max,r_max,r_max+1))

    aligned_auxin_map = compute_local_2d_signal(np.transpose([aligned_signal_X,aligned_signal_Y]),np.transpose([aligned_xx,aligned_yy],(1,2,0)),aligned_signal_data['Auxin'],cell_radius=cell_radius,density_k=density_k)
    aligned_density_map = nuclei_density_function(dict(zip(range(len(aligned_signal_X)),np.transpose([aligned_signal_X,aligned_signal_Y,np.zeros_like(signal_X)]))),cell_radius=cell_radius,k=density_k)(aligned_xx,aligned_yy,np.zeros_like(aligned_xx))
    aligned_confidence_map = nd.gaussian_filter(aligned_density_map,sigma=2.0)

    aligned_pin1_map = compute_local_2d_signal(np.array([aligned_wall_centers[(l,r)][:2] for (l,r) in wall_cells]),np.transpose([aligned_xx,aligned_yy],(1,2,0)),np.array([pin1_intensities[(l,r)] for (l,r) in wall_cells]),cell_radius=cell_radius,density_k=density_k)


    pin1_thetas = np.radians(np.linspace(-180,179,360))[::4]
    theta_sigma = np.pi/6.
    aligned_pin1_histograms = dict(zip(np.unique(wall_cells),[np.zeros_like(pin1_thetas) for c in np.unique(wall_cells)]))
    for l,r in wall_cells:
        pin1_vector = aligned_wall_pin1_vectors[(l,r)]
        if np.linalg.norm(pin1_vector)>0:
            pin1_theta = np.sign(pin1_vector[1])*np.arccos(pin1_vector[0]/np.linalg.norm(pin1_vector))
            if (pin1_vector[1]==0) and (pin1_vector[0]<0) : pin1_theta=-np.pi

            theta_weights = np.sum([np.exp(-np.power(pin1_thetas-pin1_theta+k*np.pi,2.)/(2*np.power(theta_sigma,2.))) for k in [-2,0,2]],axis=0)/(theta_sigma*np.sqrt(2*np.pi))
            aligned_pin1_histograms[l] += wall_weights[(l,r)]*theta_weights

    cell_wall_weights = dict(zip(np.unique(wall_cells),nd.sum(np.array([wall_weights[(l,r)] for (l,r) in wall_cells]),wall_cells[:,0],index=np.unique(wall_cells))))
    cell_wall_areas = dict(zip(np.unique(wall_cells),nd.sum(np.array([wall_areas[(l,r)] for (l,r) in wall_cells]),wall_cells[:,0],index=np.unique(wall_cells))))

    for l in np.unique(wall_cells):
        aligned_pin1_histograms[l] = aligned_pin1_histograms[l]/cell_wall_weights[l]
        # aligned_pin1_histograms[l] = aligned_pin1_histograms[l]/cell_wall_areas[l]

    aligned_cell_auxin = compute_local_2d_signal(np.transpose([aligned_signal_X,aligned_signal_Y]),np.array(aligned_cell_centers.values()),aligned_signal_data['Auxin'],cell_radius=cell_radius,density_k=density_k)


    figure = plt.figure(4)
    figure.clf()
    figure.patch.set_facecolor('w')

    scale_factor = 5.
    # scale_factor = 1/5000.

    # figure.gca().pcolormesh(aligned_xx,aligned_yy,aligned_auxin_map,cmap='lemon_hot_r',alpha=0.5,antialiased=True,shading='gouraud',vmin=0.5,vmax=1)
    # figure.gca().scatter(np.array(aligned_cell_centers.values())[:,0],np.array(aligned_cell_centers.values())[:,1],c='g',s=40,linewidth=0,zorder=5)
    figure.gca().scatter(np.array(aligned_cell_centers.values())[:,0],np.array(aligned_cell_centers.values())[:,1],c=aligned_cell_auxin,s=120,cmap='lemon_hot_r',linewidth=1,edgecolor='g',vmin=0.5,vmax=1,zorder=1)
    # figure.gca().scatter(aligned_signal_X,aligned_signal_Y,c=aligned_signal_data['Auxin'].values,s=100,cmap='lemon_hot_r',linewidth=0,vmin=0.5,vmax=1,zorder=1)

    for l in np.unique(wall_cells):
        histo_x = aligned_cell_centers[l][0] + scale_factor*aligned_pin1_histograms[l]*np.cos(pin1_thetas)
        histo_y = aligned_cell_centers[l][1] + scale_factor*aligned_pin1_histograms[l]*np.sin(pin1_thetas)
        figure.gca().plot(histo_x,histo_y,color='g',linewidth=2)

    c = patch.Circle([0,0],radius=28,edgecolor='m',facecolor='none',linewidth=3)
    figure.gca().add_patch(c)

    figure.gca().set_xlim(-r_max,r_max)
    figure.gca().set_ylim(-r_max,r_max)

    figure.gca().axis('off')
    figure.set_size_inches(20,20)
    figure.tight_layout()
    # figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_aligned_L1_Auxin_wall_PIN1_polarities.png")
    figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_aligned_L1_cell_PIN1_polarity_histogram.png")


    for l in np.unique(wall_cells):
        aligned_pin1_histograms[l] = aligned_pin1_histograms[l]*cell_wall_weights[l]/cell_wall_areas[l]

    figure = plt.figure(4)
    figure.clf()
    figure.patch.set_facecolor('w')

    # scale_factor = 5.
    scale_factor = 1/1000.

    # figure.gca().scatter(np.array(aligned_cell_centers.values())[:,0],np.array(aligned_cell_centers.values())[:,1],c='g',s=40,linewidth=0,zorder=1)
    figure.gca().scatter(np.array(aligned_cell_centers.values())[:,0],np.array(aligned_cell_centers.values())[:,1],c=aligned_cell_auxin,s=120,cmap='lemon_hot_r',linewidth=1,edgecolor='g',vmin=0.5,vmax=1,zorder=1)
    # figure.gca().scatter(aligned_signal_X,aligned_signal_Y,c=aligned_signal_data['Auxin'].values,s=200,cmap='lemon_hot_r',linewidth=0,vmin=0.5,vmax=1,zorder=1)

    for l in np.unique(wall_cells):
        histo_x = aligned_cell_centers[l][0] + scale_factor*aligned_pin1_histograms[l]*np.cos(pin1_thetas)
        histo_y = aligned_cell_centers[l][1] + scale_factor*aligned_pin1_histograms[l]*np.sin(pin1_thetas)
        figure.gca().plot(histo_x,histo_y,color='g',linewidth=2)

    c = patch.Circle([0,0],radius=28,edgecolor='m',facecolor='none',linewidth=3)
    figure.gca().add_patch(c)

    figure.gca().set_xlim(-r_max,r_max)
    figure.gca().set_ylim(-r_max,r_max)

    figure.gca().axis('off')
    figure.set_size_inches(20,20)
    figure.tight_layout()
    # figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_aligned_L1_Auxin_wall_PIN1_polarities.png")
    figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_aligned_L1_cell_PIN1_intensity_polarity_histogram.png")





    figure = plt.figure(4)
    figure.clf()
    figure.patch.set_facecolor('w')

    figure.gca().scatter(aligned_signal_X,aligned_signal_Y,c=aligned_signal_data['Auxin'].values,s=200,cmap='lemon_hot_r',linewidth=0,vmin=0.5,vmax=1,zorder=1)
    # mpl_draw_topomesh(all_smooth_wall_topomesh, figure, degree=1, color='k', alpha=0.2, linewidth=1)
    # figure.gca().imshow(pi_img[coords].reshape(xx.shape),cmap="Oranges",vmin=0,vmax=50000,extent=extent,interpolation='none',alpha=0)

    for (left_label, right_label), pin1_orientation in pin1_orientations.items():
        intensity = pin1_intensities[(left_label,right_label)]
        # shift = pin1_shifts[(left_label,right_label)]
        wall_center = aligned_wall_centers[(left_label,right_label)]
        wall_normal = aligned_wall_normals[(left_label,right_label)]
        if pin1_orientation:
            pin1_vector = np.sign(pin1_orientation)*wall_normal
            pin1_color = mpl.cm.ScalarMappable(cmap='Greens',norm=Normalize(vmin=0,vmax=65535)).to_rgba(intensity)

            figure.gca().arrow(wall_center[0]-0.75*pin1_vector[0],wall_center[1]-0.75*pin1_vector[1],pin1_vector[0],pin1_vector[1],head_width=1,head_length=1,edgecolor='k',facecolor=pin1_color,linewidth=1,alpha=np.abs(pin1_orientation))

    figure.gca().set_xlim(-r_max,r_max)
    figure.gca().set_ylim(-r_max,r_max)

    figure.gca().axis('off')
    figure.set_size_inches(20,20)
    figure.tight_layout()
    figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_aligned_L1_Auxin_wall_PIN1_polarities.png")



    figure = plt.figure(5)
    figure.clf()
    figure.patch.set_facecolor('w')

    figure.gca().pcolormesh(aligned_xx,aligned_yy,aligned_auxin_map,cmap='lemon_hot_r',alpha=1,antialiased=True,shading='gouraud',vmin=0.5,vmax=1)
    figure.gca().contour(aligned_xx,aligned_yy,aligned_auxin_map,np.linspace(0,1,101),cmap='gray',alpha=0.1,linewidths=1,antialiased=True,vmin=-1,vmax=0)
    for a in xrange(16):
        figure.gca().contourf(aligned_xx,aligned_yy,aligned_confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)

    for label in aligned_cell_pin1_vectors.keys():
        cell_center = aligned_cell_centers[label]
        cell_pin1_vector = 3.*aligned_cell_pin1_vectors[label]
        # figure.gca().text(cell_center[0],cell_center[1],str(label),zorder=10,size=12,horizontalalignment='center',verticalalignment='top')
        if np.all(cell_pin1_vector):
            # figure.gca().scatter(microscope_orientation*p_img.image_property('barycenter')[l][0],microscope_orientation*p_img.image_property('barycenter')[l][1])
            figure.gca().arrow(cell_center[0]-0.75*cell_pin1_vector[0],cell_center[1]-0.75*cell_pin1_vector[1],cell_pin1_vector[0],cell_pin1_vector[1],head_width=0.5,head_length=0.5,edgecolor='r',facecolor='none',linewidth=4,alpha=1)

    c = patch.Circle([0,0],radius=28,edgecolor='m',facecolor='none',linewidth=3)
    figure.gca().add_patch(c)

    figure.gca().axis('off')
    figure.set_size_inches(20,20)
    figure.tight_layout()
    figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_aligned_L1_Auxin_map_PIN1_polarities.png")


    figure = plt.figure(6)
    figure.clf()
    figure.patch.set_facecolor('w')

    figure.add_subplot(1,2,1)
    figure.gca().pcolormesh(aligned_xx,aligned_yy,aligned_pin1_map,cmap='green_hot',alpha=1,antialiased=True,shading='gouraud',vmin=20000,vmax=65000)
    figure.gca().contour(aligned_xx,aligned_yy,aligned_pin1_map,np.linspace(0,70000,51),cmap='gray',alpha=0.5,linewidths=1,antialiased=True,vmin=-1,vmax=0)

    figure.add_subplot(1,2,2)
    figure.gca().pcolormesh(aligned_xx,aligned_yy,aligned_auxin_map,cmap='lemon_hot_r',alpha=1,antialiased=True,shading='gouraud',vmin=0.5,vmax=1)
    figure.gca().contour(aligned_xx,aligned_yy,aligned_auxin_map,np.linspace(0,1,101),cmap='gray',alpha=0.5,linewidths=1,antialiased=True,vmin=-1,vmax=0)

    for i in [1,2]:
        figure.add_subplot(1,2,i)
        for a in xrange(16):
            figure.gca().contourf(aligned_xx,aligned_yy,aligned_confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)

        CS = figure.gca().contour(aligned_xx, aligned_yy, np.linalg.norm([aligned_xx,aligned_yy],axis=0),np.linspace(0,r_max,r_max/10+1),cmap='Greys',vmin=-1,vmax=0,alpha=0.15)
        figure.gca().clabel(CS, inline=1, fontsize=10,alpha=0.1)

        c = patch.Circle([0,0],radius=28,edgecolor='m',facecolor='none',linewidth=3)
        figure.gca().add_patch(c)
        figure.gca().axis('off')

    figure.set_size_inches(40,20)
    figure.tight_layout()
    figure.savefig("/Users/gcerutti/Desktop/WorkVP/SamMaps/nuclei_images/"+filename+"/"+filename+"_aligned_L1_PIN1_map.png")
