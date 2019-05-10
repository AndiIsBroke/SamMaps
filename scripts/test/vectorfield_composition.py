import numpy as np
from timagetk.components import SpatialImage
from timagetk.algorithms.trsf import compose_trsf
from timagetk.wrapping import BalTrsf
from timagetk.wrapping.bal_trsf import TRSF_TYPE_DICT
from timagetk.wrapping.bal_trsf import TRSF_UNIT_DICT
trsf_type = 'VECTORFIELD_3D'
trsf_unit = 'REAL_UNIT'

arr2 = np.zeros((30, 30, 15))
im2 = SpatialImage(arr2, origin=[0,0,0], voxelsize=[1.,1.,2.])

if isinstance(trsf_type, str):
    trsf_type = TRSF_TYPE_DICT[trsf_type]
if isinstance(trsf_unit, str):
    trsf_unit = TRSF_UNIT_DICT[trsf_unit]

t01 = BalTrsf(trsf_type=trsf_type, trsf_unit=trsf_unit)
t01.read('/home/jonathan/Projects/TissueAnalysis/timagetk/timagetk/share/data/vf3d_id_20_20_10.trsf')
t12 = BalTrsf(trsf_type=trsf_type, trsf_unit=trsf_unit)
t12.read('/home/jonathan/Projects/TissueAnalysis/timagetk/timagetk/share/data/vf3d_id_30_30_15.trsf')

t02 = compose_trsf([t01, t12], template_img=im2)
