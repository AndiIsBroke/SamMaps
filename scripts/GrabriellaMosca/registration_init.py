from timagetk.components import SpatialImage

from timagetk.wrapping import BalTrsf

from timagetk.io import imread
from timagetk.io import imsave

from timagetk.plugins import registration
from timagetk.plugins import morphology

raw_ref_img = imread("/data/GabriellaMosca/EM_C_140/EM_C_140- C=0.tif")
raw_float_img = imread("/data/GabriellaMosca/EM_C_214/EM_C_214 C=0.tif")

ref_img = imread("/data/GabriellaMosca/EM_C_140/EM_C_140-masked.tif")
float_img = imread("/data/GabriellaMosca/EM_C_214/EM_C_214-masked.tif")

init_trsf = BalTrsf()
init_trsf.read('/data/GabriellaMosca/T140_214.trsf')

trsf_def, img_def = registration(float_img, ref_img, method='deformable', init_trsf=init_trsf)
imsave("/data/GabriellaMosca/EM_C_214/EM_C_214-masked_deformable.tif", img_def)

from timagetk.visu.mplt import grayscale_imshow
img2plot = [float_img, ref_img, img_rig, ref_img]
img_titles = ["t0", "t1", "Registered t0 on t1", "t1"]
grayscale_imshow(img2plot, "Effect of rigid registration", img_titles, vmin=0, vmax=255, max_per_line=2)


img2plot = [float_img.get_z_slice(40), ref_img.get_z_slice(40), img_rig.get_z_slice(40), ref_img.get_z_slice(40)]
img_titles = ["t0", "t1", "Registered t0 on t1", "t1"]
grayscale_imshow(img2plot, "Effect of rigid registration", img_titles, vmin=0, vmax=255, max_per_line=2)
