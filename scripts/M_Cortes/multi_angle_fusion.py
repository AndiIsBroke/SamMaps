from timagetk.io import imread, imsave
from timagetk.plugins.fusion import fusion_on_first
from timagetk.visu.mplt import grayscale_imshow, stack_browser, channels_stack_browser
from timagetk.algorithms.resample import isometric_resampling
from timagetk.plugins.projection import projection

root_dir = '/projects/lsfm/multi_view/20190430_M_lti6b-gfp'
filenames = ['20190430_multiangle_top.tif', '20190430_multiangle_0.tif', '20190430_multiangle_45.tif', '20190430_multiangle_90.tif', '20190430_multiangle_135.tif', '20190430_multiangle_180.tif', '20190430_multiangle_-45.tif', '20190430_multiangle_-90.tif', '20190430_multiangle_-135.tif']

proj_list = []
for fname in filenames:
    img = imread(root_dir+"/"+fname)
    print "{}: shape={}, voxelsize={}".format(img.filename, img.shape, img.voxelsize)
    proj_list.append(projection(img, method='contour'))

grayscale_imshow(proj_list, range='auto', title='20190430_M_lti6b-gfp', subplot_titles=filenames, figname=root_dir+"/"+"contour_projections.png")


# filenames = ['20190430_multiangle_top.tif', '20190430_multiangle_45.tif']
# list_img = [imread(root_dir+"/"+im) for im in filenames]

# res_trsf, res_imgs = fusion_on_first(list_img, registration_method='rigid')

# from timagetk.plugins.cropping import max_percentile_profiles
# max_percentile_profiles(top_im, n_jobs=1)
#
#
# from timagetk.plugins.cropping import threshold_max_percentile_widget
# crop = threshold_max_percentile_widget(top_im, n_jobs=1)
#
# list_img = [threshold_max_percentile_widget(imread(root_dir+"/"+im)) for im in filenames]
#
# list_img = [isometric_resampling(imread(root_dir+"/"+im), method='max', option='linear') for im in filenames]
#
#
# channels_stack_browser(res_imgs, channel_names=[f for f in list_files], colors=['red', 'green', 'blue'], title="Fusion on first ({}-{})".format(reg_method, av_method))
