from timagetk.components import SpatialImage

from timagetk.io import imread
from timagetk.io import imsave

from timagetk.plugins import registration
from timagetk.plugins import morphology

raw_float_img = imread("/data/GabriellaMosca/EM_C_140/EM_C_140- C=0.tif")
raw_ref_img = imread("/data/GabriellaMosca/EM_C_214/EM_C_214 C=0.tif")

mask_seg_float_img = imread("/data/GabriellaMosca/EM_C_140/C3-EM_C_C2_relabeledCropped.tif")
mask_seg_float_img = mask_seg_float_img.revert_axis('y')
mask_seg_float_img = mask_seg_float_img.astype('uint8')
imsave("/data/GabriellaMosca/EM_C_140/EM_C_140-segmented.tif", mask_seg_float_img)

mask_seg_ref_img = imread("/data/GabriellaMosca/EM_C_214/EM_C_214_ C=1_relabeledAndCropped.tif")
mask_seg_ref_img = mask_seg_ref_img.revert_axis('y')
mask_seg_ref_img = mask_seg_ref_img.astype('uint8')
imsave("/data/GabriellaMosca/EM_C_214/EM_C_214-segmented.tif", mask_seg_ref_img)


################################################################################
# - Apply morphological filters:

from scipy import ndimage

POSS_METHODS = ["erosion", "dilation", "opening", "closing"]
DEFAULT_METHOD = 0

DEF_ITERS = 1
DEF_CONNECT = 18

def mask_morphology(image, method=None, **kwargs):
    """Binary morphology plugin.

    Valid ``method`` input are:

      - erosion
      - dilation
      - opening
      - closing

    Parameters
    ----------
    image : SpatialImage
        input image to modify, should be a 'grayscale' (intensity) image
    method : str, optional
        used method, by default 'erosion'

    Other Parameters
    ----------------
    iterations : int, optional
        number of time to apply the morphological operation, default is 1
    connectivity : int, optional
        use it to override the default 'sphere' parameter for the structuring
        element, equivalent to 'connectivity=18'

    Returns
    -------
    SpatialImage
        image and metadata

    Raises
    ------
    TypeError
        if the given ``image`` is not a ``SpatialImage``
    NotImplementedError
        if the given ``method`` is not defined in the list of possible methods ``POSS_METHODS``

    Example
    -------
    >>> from timagetk.util import data_path
    >>> from timagetk.io import imread
    >>> from timagetk.plugins import morphology
    >>> image_path = data_path('time_0_cut.inr')
    >>> image = imread(image_path)
    >>> dilation_image = morphology(image, radius=2, iterations=2, method='dilation')
    >>> oc_asf_image = morphology(image, max_radius=3, method='oc_alternate_sequential_filter')
    """
    # - Assert the 'image' is a SpatialImage instance:
    _input_img_check(image)
    # - Set method if None and check it is a valid method:
    method = _method_check(method, POSS_METHODS, DEFAULT_METHOD)

    iterations = kwargs.get('iterations', DEF_ITERS)
    connectivity = kwargs.get('connectivity', DEF_CONNECT)
    if method == 'erosion':
        return morphology_erosion(image, iterations, connectivity)
    elif method == 'dilation':
        return morphology_dilation(image, iterations, connectivity)
    elif method == 'opening':
        return morphology_opening(image, iterations, connectivity)
    elif method == 'closing':
        return morphology_closing(image, iterations, connectivity)
    else:
        msg = "The required method '{}' is not implemented!"
        raise NotImplementedError(msg.format(method))

def binary_erosion(image, iterations=DEF_ITERS, connectivity=DEF_CONNECT):
    """Morpholocial erosion on binary image.

    Parameters
    ----------
    image : SpatialImage
        input image to transform
    iterations : int, optional
        number of iterations to performs with structuring element, default is 1
    connectivity : int, optional
        use it to override the default 'sphere' parameter for the structuring
        element, equivalent to 'connectivity=18'

    Returns
    -------
    SpatialImage
        transformed image with its metadata
    """
    ori = image.origin
    vxs = image.voxelsize
    md = image.metadata
    struct = ndimage.generate_binary_structure(3, connectivity)
    out_img = ndimage.binary_erosion(image.get_array(), structure=struct, iterations=iterations)
    return SpatialImage(out_img, origin=ori, voxelsize=vxs, metadata=md)

def binary_dilation(image, iterations=DEF_ITERS, connectivity=DEF_CONNECT):
    """Morpholocial dilation on binary image.

    Parameters
    ----------
    image : SpatialImage
        input image to transform
    iterations : int, optional
        number of iterations to performs with structuring element, default is 1
    connectivity : int, optional
        use it to override the default 'sphere' parameter for the structuring
        element, equivalent to 'connectivity=18'

    Returns
    -------
    SpatialImage
        transformed image with its metadata
    """
    ori = image.origin
    vxs = image.voxelsize
    md = image.metadata
    struct = ndimage.generate_binary_structure(3, connectivity)
    out_img = ndimage.binary_dilation(image.get_array(), structure=struct, iterations=iterations)
    return SpatialImage(out_img, origin=ori, voxelsize=vxs, metadata=md)


################################################################################

# Re-create masks:
def get_mask(img):
    return SpatialImage(img.get_array() != 0, voxelsize=img.voxelsize, dtype="uint8")

float_mask = get_mask(mask_seg_float_img)
ref_mask = get_mask(mask_seg_ref_img)

float_mask = binary_erosion(float_mask, iterations=2)
float_mask = binary_dilation(float_mask, iterations=10)

ref_mask = binary_erosion(ref_mask, iterations=2)
ref_mask = binary_dilation(ref_mask, iterations=10)

# imsave("/data/GabriellaMosca/EM_C_140/EM_C_140-mask.tif", float_mask.astype('uint8'))
# imsave("/data/GabriellaMosca/EM_C_140/EM_C_214-mask.tif", ref_mask.astype('uint8'))


def apply_mask(img, mask):
    return SpatialImage(img.get_array() * mask, origin=img.origin, voxelsize=img.voxelsize, metadata=img.metadata)

float_img = apply_mask(raw_float_img, float_mask)
imsave("/data/GabriellaMosca/EM_C_140/EM_C_140-masked.tif", float_img)

ref_img = apply_mask(raw_ref_img, ref_mask)
imsave("/data/GabriellaMosca/EM_C_214/EM_C_214-masked.tif", ref_img)


trsf_rig, img_rig = registration(float_img, ref_img, method='rigid')
imsave("/data/GabriellaMosca/EM_C_140/EM_C_140-masked_rigid.tif", img_rig)

trsf_aff, img_aff = registration(float_img, ref_img, method='affine')
imsave("/data/GabriellaMosca/EM_C_140/EM_C_140-masked_affine.tif", img_aff)

trsf_def, img_def = registration(float_img, ref_img, method='deformable')
imsave("/data/GabriellaMosca/EM_C_140/EM_C_140-masked_deformable.tif", img_def)

from timagetk.visu.mplt import grayscale_imshow
img2plot = [float_img, ref_img, img_rig, ref_img]
img_titles = ["t0", "t1", "Registered t0 on t1", "t1"]
grayscale_imshow(img2plot, "Effect of rigid registration", img_titles, vmin=0, vmax=255, max_per_line=2)


img2plot = [float_img.get_z_slice(40), ref_img.get_z_slice(40), img_rig.get_z_slice(40), ref_img.get_z_slice(40)]
img_titles = ["t0", "t1", "Registered t0 on t1", "t1"]
grayscale_imshow(img2plot, "Effect of rigid registration", img_titles, vmin=0, vmax=255, max_per_line=2)
