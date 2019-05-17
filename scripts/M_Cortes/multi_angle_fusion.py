from timagetk.io import imread, imsave
from timagetk.plugins.fusion import fusion_on_first
from timagetk.visu.mplt import stack_browser, stack_browser_RGBA
from timagetk.algorithms.resample import isometric_resampling

root_dir = '/projects/lsfm/multi_view/20190430_M_lti6b-gfp'
# list_fname = ['20190430_multiangle_top.tif', '20190430_multiangle_45.tif', '20190430_multiangle_90.tif']
list_fname = ['20190430_multiangle_top.tif', '20190430_multiangle_45.tif']
list_img = [imread(root_dir+"/"+im) for im in list_fname]
# top_im = imread(root_dir+"/"+list_fname[0])

res_trsf, res_imgs = fusion_on_first(list_img, registration_method='rigid')

# from timagetk.plugins.cropping import max_percentile_profiles
# max_percentile_profiles(top_im, n_jobs=1)
#
#
# from timagetk.plugins.cropping import threshold_max_percentile_widget
# crop = threshold_max_percentile_widget(top_im, n_jobs=1)
#
# list_img = [threshold_max_percentile_widget(imread(root_dir+"/"+im)) for im in list_fname]
#
# list_img = [isometric_resampling(imread(root_dir+"/"+im), method='max', option='linear') for im in list_fname]
#
#
# stack_browser_RGBA(res_imgs, channel_names=[f for f in list_files], colors=['red', 'green', 'blue'], title="Fusion on first ({}-{})".format(reg_method, av_method))
