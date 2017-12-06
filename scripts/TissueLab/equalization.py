from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

def equalize_adapthist(img, clip_limit=None):
    # Adaptive Equalization
    if clip_limit is None:
        clip_limit = 0.01
    return np.array(exposure.equalize_adapthist(img, clip_limit=clip_limit)*(2**16-1)).astype(np.uint16)
# [O.O1, 0.05, 0.1, 0.2, 0.5, 0.8]

def sl_equalize_adapthist(img, clip_limit=None):
    # Slice by slice equalization
    sh = img.shape
    return np.array([equalize_adapthist(img[:,:,n], clip_limit) for n in range(0, sh[2])]).transpose([1,2,0])

def contrast_stretch(img, pc_min=2, pc_max=99):
    # Contrast stretching
    pcmin = np.percentile(img, pc_min)
    pcmax = np.percentile(img, pc_max)
    return exposure.rescale_intensity(img, in_range=(pcmin, pcmax))

def sl_contrast_stretch(img, pc_min=2, pc_max=99):
    # Slice by slice contrast stretching
    sh = img.shape
    return np.array([contrast_stretch(img[:,:,n], pc_min, pc_max) for n in range(0, sh[2])]).transpose([1,2,0])


def slice_view(img, x_slice=None, y_slice=None, z_slice=None, cbar=False, title="", fig_name=""):
    """
    Matplotlib representation of an image slice.
    Slice numbering starts at 1, not 0 (like indexing).

    Parameters
    ----------
    img : np.array|SpatialImage
        image from which to extract the slice
    """
    import matplotlib.pyplot as plt

    try:
        assert x_slice is not None or y_slice is not None or z_slice is not None
    except:
        raise ValueError("Provide at least one x, y or z slice to extract!")
    x_sl, y_sl, z_sl = None, None, None

    if x_slice is not None:
        x_sl = img[x_slice-1, :, :]
    if y_slice is not None:
        y_sl = img[:, y_slice-1, :]
    if z_slice is not None:
        z_sl = img[:, :, z_slice-1]

    if sum([sl is not None for sl in [x_sl, y_sl, z_sl]]) == 1:
        plt.figure()
        plt.imshow(z_sl, 'gray', vmin=0, vmax=2**16-1)
        plt.title(title)
        if fig_name != "":
            plt.savefig(fig_name)
    return
