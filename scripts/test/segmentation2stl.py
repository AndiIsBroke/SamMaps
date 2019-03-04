from mayavi import mlab
from timagetk.components import SpatialImage

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
    dirname = "/data/Meristems/Carlos/PIN_maps/"
elif platform.uname()[1] == "calculus":
    SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
    dirname = "/projects/SamMaps/"
else:
    raise ValueError("Unknown custom path to 'SamMaps' for this system...")
sys.path.append(SamMaps_dir+'/scripts/lib/')

from segmentation_pipeline import read_image
from nomenclature import splitext_zip
from openalea.cellcomplex.property_topomesh.utils.image_tools import image_to_vtk_polydata

# fname = '/data/Meristems/Bihai/20180523_WT_Ler_SD+LD_1/20180523 WT_Ler SD+LD_1#-T0-seg-iso-adpat_eq-h_min9.inr'
fname = '/data/Meristems/Carlos/PIN_maps/nuclei_images/qDII-PIN1-CLV3-PI-LD_E37_171113_sam07_t14/qDII-PIN1-CLV3-PI-LD_E37_171113_sam07_t14_PI_raw_segmented.inr.gz'
im = read_image(fname)

import tissue_printer
reload(tissue_printer)
from tissue_printer import *


# - Performs vtkDiscreteMarchingCubes on labelled image (with all labels!)
# dmc = vtk_dmc(im, mesh_fineness=3.0)
# write_stl(bin_dmc, splitext_zip(fname)[0]+'.stl')


# - Performs vtkDiscreteMarchingCubes on inside/outside masked image:
back_id = 1
# -- Convert the labelled image into inside/outside mask image:
mask = np.array(im != back_id, dtype="uint8")
bin_im = SpatialImage(mask, voxelsize=im.get_voxelsize())
# -- Run vtkDiscreteMarchingCubes:
bin_dmc = vtk_dmc(bin_im)
# from openalea.cellcomplex.property_topomesh.utils.image_tools import image_to_vtk_polydata
# bin_dmc = image_to_vtk_polydata(bin_im, mesh_fineness=5.0)
# from openalea.cellcomplex.property_topomesh.utils.image_tools import image_to_vtk_cell_polydata
# bin_dmc = image_to_vtk_cell_polydata(bin_im, mesh_fineness=3.0)

stl_fname = splitext_zip(fname)[0]+'_binary.stl'
write_stl(bin_dmc, stl_fname)

from vplants.tissue_analysis.image2vtk import mlab_vtkSurface_viewer
mlab_vtkSurface_viewer(bin_dmc)
