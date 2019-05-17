from os.path import join

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    base_dir = '/data/Meristems/Bihai/20180616_3-global_della_Ler_LD/'
elif platform.uname()[1] == "calculus":
    base_dir = '/projects/SamGrowth/microscopy/20180616_global_della_Ler_LD'
else:
    raise ValueError("Unknown custom path to 'base_dir' for this system...")

fname = '20180616 3-global della_Ler_LD +++.lsm'

from timagetk.io import imread
im = imread(join(base_dir, fname))

from timagetk.plugins import linear_filtering
from timagetk.algorithms.exposure import z_slice_contrast_stretch
from timagetk.algorithms.exposure import z_slice_equalize_adapthist
s = 0.6

smooth = linear_filtering(im, method="gaussian_smoothing", sigma=s, real=True)

stretch = z_slice_contrast_stretch(im)
smooth_str = z_slice_contrast_stretch(linear_filtering(im, method="gaussian_smoothing", sigma=s, real=True))
str_smooth = linear_filtering(z_slice_contrast_stretch(im), method="gaussian_smoothing", sigma=s, real=True)

eq = z_slice_equalize_adapthist(im)
smooth_eq = z_slice_equalize_adapthist(linear_filtering(im, method="gaussian_smoothing", sigma=s, real=True))
eq_smooth = linear_filtering(z_slice_equalize_adapthist(im), method="gaussian_smoothing", sigma=s, real=True)

from timagetk.visu.mplt import image_n_hist
sh = im.shape
xs, ys = int(sh[0]/2.), int(sh[1]/3.)

img_list = [im, stretch, smooth, smooth_str, str_smooth]
img_title = ['Original', "Stretched", "Gaussian_smoothing", "Smooth + Stretching", "Stretching + Smooth"]
image_n_hist([img.get_array()[xs, ys:ys*2, :] for img in img_list], title="x_slice{}".format(xs), img_title=img_title, aspect_ratio=im.extent[0]/im.extent[1])

img_list = [im, eq, smooth, smooth_eq, eq_smooth]
img_title = ['Original', "Equalize", "Gaussian_smoothing", "Smooth + Equalize", "Equalize + Smooth"]
image_n_hist([img.get_array()[xs, ys:ys*2, :] for img in img_list], title="x_slice{}".format(xs), img_title=img_title, aspect_ratio=im.extent[0]/im.extent[1])


image_n_hist(smooth_str.get_array()[xs, :, :], figname=join(base_dir, "smooth-str-x_slice{}.png".format(xs)))
image_n_hist(str_smooth.get_array()[xs, :, :], figname=join(base_dir, "str-smooth-x_slice{}.png".format(xs)))

from timagetk.visu.mplt import grayscale_imshow

grayscale_imshow([im, iso_im, st_im])
