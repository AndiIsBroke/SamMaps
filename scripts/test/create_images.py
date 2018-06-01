# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon - INRIA
#
#       File author(s): Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
################################################################################

"""
Library to create artificial segmented and intensity images.
"""

import numpy as np

from timagetk.components import SpatialImage


def create_two_label_image(labA=2, labB=3, x=30, y=30, z=10, vxs=(1., 1., 1.)):
    """
    Create a labelled image (SpatialImage) with two labels splitting it in two
    equal parts (along the x-axis).
    The image shape is given by 'x', 'y' & 'z'.
    Use an unsigned 8-bit encoding.

    Parameters
    ----------
    labA : int, optional
        label A
    labB : int, optional
        label B
    x : int, optional
        shape along the x dimension
    y : int, optional
        shape along the y dimension
    z : int, optional
        shape along the z dimension
    vxs : tuple|list|np.array
        voxelsize of the returned SpatialImage

    Returns
    -------
    im : SpatialImage
        the labelled image
    """
    i = int(x/2.)
    arr = np.zeros((x, y, z), dtype=np.uint8)
    arr[0:i, :, :] = labA
    arr[i:, :, :] = labB
    im = SpatialImage(arr, voxelsize=vxs, origin=(0, 0, 0))
    return im


def linear_decrease(x, a, b):
    """
    Returns 'y', the result of the linear decreasing function:
       y = a * x + b
    """
    return a * x + b


def signal2dist(n_values, decrease_func='linear'):
    """
    Compute the signal values, from max value to 0 according to a decreas
    function.

    Parameters
    ----------
    n_values : int
        number of intensity values to compute (equal to the required distance)
    decrease_func : str, optional
        name of a function computing the signal decrease as a function of the
        distance to the membrane
    """
    max_value = 255  # 'uint8' case
    min_value = 0
    signal = [min_value] * (n_values + 1)
    distance = range(n_values)

    if decrease_func == 'linear':
        a = - max_value / float(n_values)
        b = 255
        for d in distance:
            signal[d] = linear_decrease(d, a, b)

    return signal


def create_two_sided_intensity_image(x=30, y=30, z=10, vxs=(1., 1., 1.), max_sig_dist=8., signal_dist_func='linear'):
    """
    Create a two sided intensity image with a separation splitting the image in
    two equal parts (along the x-axis).
    The image shape is given by 'x', 'y' & 'z'.
    Use an unsigned 8-bit encoding.

    Parameters
    ----------
    x : int, optional
        shape along the x dimension
    y : int, optional
        shape along the y dimension
    z : int, optional
        shape along the z dimension
    vxs : tuple|list|np.array
        voxelsize of the returned SpatialImage
    max_sig_dist : float
        max distance to the membrane, in real units (ie. depend on 'vxs'), at
        which there is no more signal (signal intensity = 0)
    signal_dist_func : str
        string mathing the name of function computing the signal decrease as a
        function of the distance to the membrane (middle of the image here.)
    """
    i = int(x/2.)
    arr = np.zeros((x, y, z), dtype=np.uint8)
    # - Define the decreasing signal values according to membrane distance
    vox_dist = int(max_sig_dist/vxs[0])  # we split the image along the x axis, hence the x-voxelsize!
    try:
        assert vox_dist < i
    except AssertionError:
        raise ValueError("Parameters 'max_sig_dist' is too big (ie. at {}, but should be < {})".format(max_sig_dist, i*vxs[0]))

    decrease_sig2dist = signal2dist(vox_dist, signal_dist_func)

    # - Add these values to the intensity image:
    for d in range(vox_dist):
        arr[(i-1) - d, :, :] = arr[i + d, :, :] = decrease_sig2dist[d]

    return SpatialImage(arr, voxelsize=vxs, origin=(0, 0, 0))


def create_left_sided_intensity_image(x=30, y=30, z=10, vxs=(1., 1., 1.), max_sig_dist=8., signal_dist_func='linear'):
    """
    Create a left sided intensity image with a separation splitting the image in
    two equal parts (along the x-axis).
    The image shape is given by 'x', 'y' & 'z'.
    Use an unsigned 8-bit encoding.

    Parameters
    ----------
    x : int, optional
        shape along the x dimension
    y : int, optional
        shape along the y dimension
    z : int, optional
        shape along the z dimension
    vxs : tuple|list|np.array
        voxelsize of the returned SpatialImage
    max_sig_dist : float
        max distance to the membrane, in real units (ie. depend on 'vxs'), at
        which there is no more signal (signal intensity = 0)
    signal_dist_func : str
        string mathing the name of function computing the signal decrease as a
        function of the distance to the membrane (middle of the image here.)
    """
    i = int(x/2.)
    arr = np.zeros((x, y, z), dtype=np.uint8)
    # - Define the decreasing signal values according to membrane distance
    vox_dist = int(max_sig_dist/vxs[0])  # we split the image along the x axis, hence the x-voxelsize!
    try:
        assert vox_dist < i
    except AssertionError:
        raise ValueError("Parameters 'max_sig_dist' is too big (ie. at {}, but should be < {})".format(max_sig_dist, i*vxs[0]))

    decrease_sig2dist = signal2dist(vox_dist, signal_dist_func)

    # - Add these values to the intensity image:
    for d in range(vox_dist):
        arr[(i-1) - d, :, :] = decrease_sig2dist[d]

    return SpatialImage(arr, voxelsize=vxs, origin=(0, 0, 0))


def create_right_sided_intensity_image(x=30, y=30, z=10, vxs=(1., 1., 1.), max_sig_dist=8., signal_dist_func='linear'):
    """
    Create a right sided intensity image with a separation splitting the image in
    two equal parts (along the x-axis).
    The image shape is given by 'x', 'y' & 'z'.
    Use an unsigned 8-bit encoding.

    Parameters
    ----------
    x : int, optional
        shape along the x dimension
    y : int, optional
        shape along the y dimension
    z : int, optional
        shape along the z dimension
    vxs : tuple|list|np.array
        voxelsize of the returned SpatialImage
    max_sig_dist : float
        max distance to the membrane, in real units (ie. depend on 'vxs'), at
        which there is no more signal (signal intensity = 0)
    signal_dist_func : str
        string mathing the name of function computing the signal decrease as a
        function of the distance to the membrane (middle of the image here.)
    """
    i = int(x/2.)
    arr = np.zeros((x, y, z), dtype=np.uint8)
    # - Define the decreasing signal values according to membrane distance
    vox_dist = int(max_sig_dist/vxs[0])  # we split the image along the x axis, hence the x-voxelsize!
    try:
        assert vox_dist < i
    except AssertionError:
        raise ValueError("Parameters 'max_sig_dist' is too big (ie. at {}, but should be < {})".format(max_sig_dist, i*vxs[0]))

    decrease_sig2dist = signal2dist(vox_dist, signal_dist_func)

    # - Add these values to the intensity image:
    for d in range(vox_dist):
        arr[i + d, :, :] = decrease_sig2dist[d]

    return SpatialImage(arr, voxelsize=vxs, origin=(0, 0, 0))
