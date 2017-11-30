from skimage import exposure
import numpy as np

def equalize_adapthist(img):
    # Adaptive Equalization
    return np.array(exposure.equalize_adapthist(img, clip_limit=0.03)*(2**16)).astype(np.uint16)

def sl_equalize_adapthist(img):
    # Slice by slice equalization
    sh = img.shape
    return np.array([equalize_adapthist(img[:,:,n]) for n in range(0, sh[2])]).transpose([1,2,0])

def contrast_stretch(img, pc_min=2, pc_max=99):
    # Contrast stretching
    pcmin = np.percentile(img, pc_min)
    pcmax = np.percentile(img, pc_max)
    return exposure.rescale_intensity(img, in_range=(pcmin, pcmax))

def sl_contrast_stretch(img, pc_min=2, pc_max=99):
    # Slice by slice contrast stretching
    sh = img.shape
    return np.array([contrast_stretch(img[:,:,n], pc_min, pc_max) for n in range(0, sh[2])]).transpose([1,2,0])
