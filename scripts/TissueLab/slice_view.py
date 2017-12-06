from matplotlib import gridspec
import matplotlib.pyplot as plt

from skimage import exposure


def type_to_range(img):
    """
    Returns the minimum and maximum values of a dtype according to image.
    """
    try:
        assert hasattr(img, 'dtype')
    except:
        raise ValueError("Input 'img' has no attribute 'dtype', please check!")

    if img.dtype == 'uint8':
        return 0, 2**8-1
    elif img.dtype == 'uint16':
        return 0, 2**16-1
    else:
        raise NotImplementedError("Does not know what to do with such type: '{}'!".format(img.dtype))


def slice_n_hist(image, title="", img_title="", figname=""):
    """
    Display a 2D image with value histogram and cummulative histogram.

    Parameters
    ----------
    image : np.array or SpatialImage
        2D image to represent
    title : str, optional
        If provided (default is empty), add this string of characters as title
    img_title : str, optional
        If provided (default is empty), add this string of characters as title
    fig_name : str, optional
        If provided (default is empty), the image will be saved under this filename.
    """
    # TODO: make use of 'skimage.exposure.histogram' and 'skimage.exposure.cumulative_distribution' ?!
    try:
        assert image.ndim == 2
    except:
        raise ValueError("Input `image` should be 2D")
    mini, maxi = type_to_range(image)
    # Initialise figure:
    plt.figure()
    plt.suptitle(title)
    gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,1])
    # Display 2D image:
    ax = plt.subplot(gs[:,0])
    plt.imshow(image, 'gray', vmin=mini, vmax=maxi)
    plt.axis('off')
    plt.title(img_title)
    # Plot intensity histogram
    ax = plt.subplot(gs[0,1])
    plt.title('Intensity histogram')
    plt.hist(image.flatten(), bins=256, range=(mini, maxi+1), normed=True)
    # exposure.histogram(image)
    # Plot intensity cumulative histogram
    ax = plt.subplot(gs[1,1])
    plt.title('Cumumative histogram')
    plt.hist(image.flatten(), bins=256, range=(mini, maxi+1), cumulative=True,
             histtype='step', normed=True)
    # exposure.cumulative_distribution(image)
    if figname != "":
        plt.savefig(figname)

    return


def slice_view(img, x_slice=None, y_slice=None, z_slice=None, title="", fig_name="", cmap='gray'):
    """
    Matplotlib representation of an image slice.
    Slice numbering starts at 1, not 0 (like indexing).
    Note that at least one of the '*_slice' parameter should be given, and all
    three can be given for an orthogonal representation of the stack.

    Parameters
    ----------
    img : np.array or SpatialImage
        Image from which to extract the slice
    x_slice : int
        Value defining the slice to represent in x direction.
    y_slice : int
        Value defining the slice to represent in y direction.
    z_slice : int
        Value defining the slice to represent in z direction.
    title : str, optional
        If provided (default is empty), add this string of characters as title
    fig_name : str, optional
        If provided (default is empty), the image will be saved under this filename.
    """
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
    # If only one slice is required, display it "alone":
    if sum([sl is not None for sl in [x_sl, y_sl, z_sl]]) == 1:
        sl = [s for s in [x_sl, y_sl, z_sl] if s is not None][0]
        mini, maxi = type_to_range(sl)
        plt.figure()
        plt.imshow(sl, cmap, vmin=mini, vmax=maxi)
        plt.title(title)
        if fig_name != "":
            plt.savefig(fig_name)
    # If three slices are required, display them "orthogonaly":
    elif sum([sl is not None for sl in [x_sl, y_sl, z_sl]]) == 3:
        x_sh, y_sh, z_sh = img.shape
        mini, maxi = type_to_range(img)
        plt.figure()
        gs = gridspec.GridSpec(2, 2, width_ratios=[x_sh, z_sh], height_ratios=[y_sh, z_sh])
        # plot z_slice:
        ax = plt.subplot(gs[0,0])
        plt.plot([x_slice, x_slice], [0, y_sh], color='yellow')
        plt.plot([0, x_sh], [y_slice, y_slice], color='yellow')
        plt.imshow(z_sl, cmap, vmin=mini, vmax=maxi)
        plt.axis('off')
        plt.title('z-slice {}/{}'.format(z_slice, z_sh))
        # plot y_slice
        ax = plt.subplot(gs[0,1])
        plt.plot([z_slice, z_slice], [0, y_sh], color='yellow')
        plt.imshow(y_sl, cmap, vmin=mini, vmax=maxi)
        plt.axis('off')
        plt.title('y-slice {}/{}'.format(y_slice, y_sh))
        # plot x_slice
        ax = plt.subplot(gs[1,0])
        plt.plot([0, x_sh], [z_slice, z_slice], color='yellow')
        plt.imshow(x_sl.T, cmap, vmin=mini, vmax=maxi)
        plt.axis('off')
        plt.title('x-slice {}/{}'.format(x_slice, x_sh))
        # Add suptitle:
        plt.suptitle(title)
        if fig_name != "":
            plt.savefig(fig_name)
    else:
        print "You should not be here !!"
        pass
    return
