# Vectorfield parameters:
# -----------------------
pm_propagation='skiz'
vpd='20'
fd='40'
fs='10'
et='lts'

# Images and landmarks paths:
# ---------------------------
path='/data/Yassin/YR_01_ATLAS_iso/registered_sequence'
t_ref='10'
t_float='0'

# (no need to change following lines):
# ------------------------------------
ref_im=$path/'t'$t_ref'.tif'
ref_seg_im=$path/'t'$t_ref'_segmented.tif'

float_im=$path/t$t_float'.tif'
float_seg_im=$path/t$t_float'_segmented.tif'

ref_ldmk=$path/'landmarks-t'$t_ref'_relab_t'$t_float'.txt'
float_ldmk=$path/'landmarks-t'$t_float'.txt'

# OUTPUT filenames (no need to change this):
# ------------------------------------------
# - Rigid registration:
res_trsf_rig=$path/'trsf-rigid-t'$t_ref'-t'$t_float'.trsf'
res_trsf_rig_inv=$path/'trsf-rigid_inv-t'$t_ref'-t'$t_float'.trsf'
float_rig_ldmk=$path/'landmarks-rigid-t'$t_float'.txt'
# - Affine registration:
res_trsf_aff=$path/'trsf-affine-t'$t_ref'-t'$t_float'.trsf'
res_trsf_aff_inv=$path/'trsf-affine_inv-t'$t_ref'-t'$t_float'.trsf'
float_aff_ldmk=$path/'landmarks-affine-t'$t_float'.txt'
# - Vectorfield registration:
res_trsf_vf=$path/'trsf-vf_'$pm_propagation'-t'$t_ref'-t'$t_float'.trsf'
res_trsf_vf_af=$path/'trsf-aff_vf_'$pm_propagation'-t'$t_ref'-t'$t_float'.trsf'
# - Final transformation estimated with blockmatching
res_trsf=$path/'trsf-bm_'$pm_propagation'-t'$t_ref'-t'$t_float'.trsf'
# - Result float images
res_float_im=$path/'t'$t_float'-on-t'$t_ref'.tif'
res_float_seg_im=$path/'t'$t_float'-on-t'$t_ref'_segmented.tif'

cd ~/Projects/Morpheme/morpheme-privat/vt/build/bin

### Rigid registration of 'floating' landmarks onto 'reference' landmarks, thus returning a tranformation oriented from 'reference' toward 'floating':
./pointmatching -reference $ref_ldmk -floating $float_ldmk -res-trsf $res_trsf_rig -trsf-type 'rigid'

### Inverting the affine tranformation which will then be oriented from 'floating' toward 'reference':
./invTrsf $res_trsf_rig $res_trsf_rig_inv

### Applying this (inverted) affine transformation to 'floating' landmarks:
./applyTrsfToPoints $float_ldmk $float_rig_ldmk -trsf $res_trsf_rig_inv

echo "ipython /data/scripts_python/morpheme_registration/quiver3d_visu.py $ref_ldmk $float_rig_ldmk"
# To run in another shell with "conda activate tissueanalysis-dev"

### Affine registration of 'floating' landmarks onto 'reference' landmarks, thus returning a tranformation oriented from 'reference' toward 'floating':
./pointmatching -reference $ref_ldmk -floating $float_ldmk -res-trsf $res_trsf_aff -trsf-type 'affine'

### Inverting the affine tranformation which will then be oriented from 'floating' toward 'reference':
./invTrsf $res_trsf_aff $res_trsf_aff_inv

### Applying this (inverted) affine transformation to 'floating' landmarks:
./applyTrsfToPoints $float_ldmk $float_aff_ldmk -trsf $res_trsf_aff_inv

echo "ipython /data/scripts_python/morpheme_registration/quiver3d_visu.py $ref_ldmk $float_aff_ldmk"


### Non-linear registration of (affine) 'floating' landmarks onto 'reference' landmarks:
./pointmatching -reference $ref_ldmk -floating $float_aff_ldmk -res-trsf $res_trsf_vf -trsf-type 'vectorfield' -template $ref_im -vector-propagation-distance $vpd -fading-distance $fd -vector-propagation-type $pm_propagation -fluid-sigma $fs -estimator-type $et

# Compose transformations 'affine' with 'non-linear' to init blockmatching with:
./composeTrsf -res $res_trsf_vf_af -trsfs $res_trsf_aff $res_trsf_vf -template $ref_im

# Run blockmatching using '-initial-transformation' param:
./blockmatching -floating $float_im -reference $ref_im \
  -initial-transformation $res_trsf_vf_af -composition-with-initial \
  -res-trsf $res_trsf -trsf-type 'vectorfield' -time

# Apply the transformation to float intensity image:
./applyTrsf $float_im $res_float_im -template $ref_im -trsf $res_trsf
echo "$res_float_im"

# Apply the transformation to float segmented image:
./applyTrsf $float_seg_im $res_float_seg_im -nearest -template $ref_im -trsf $res_trsf
echo "$res_float_seg_im"
