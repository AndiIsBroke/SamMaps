from timagetk.io import imread, imsave
from timagetk.plugins.registration import z_stack_registration, apply_z_stack_trsf
# dir = '/projects/SamMaps/SuperResolution/LSM Airyscan/'
dir = '/data/Meristems/Carlos/SuperResolution/'

base_fname = 'SAM1-gfp-pi-stack-LSM800-Airyscan Processing-standard_{}.tif'
ref_channel = 'PI_SR'
float_channels = ['PIN_SR', 'PI_conf', 'PIN_conf']
image = imread(dir + base_fname.format(ref_channel))

reg_method = 'rigid'
reg_stack, z_stack_trsf = z_stack_registration(image, method=reg_method, get_trsf=True)

# - Have a look at the z-stack registration results on PI channel:
from timagetk.visu.mplt import grayscale_imshow
mid_y = int(image.get_y_dim()/2.)
grayscale_imshow([image.get_y_slice(mid_y), reg_stack.get_y_slice(mid_y)], title="Z-stack {} registration - y-slice {}/{}".format(reg_method, mid_y, image.get_y_dim()), subplot_titles=['Original', 'Registered'], range=[0, 10000])

grayscale_imshow([image, reg_stack], title="Z-stack {} registration - Contour projection".format(reg_method), subplot_titles=['Original', 'Registered'], range=[0, 10000])

from timagetk.visu.mplt import stack_browser
stack_browser(reg_stack, title="{} registered z-stack".format(reg_method))

from timagetk.visu.mplt import stack_browser_RGBA
stack_browser_RGBA([image, reg_stack], channel_names=['Original', 'Registered'], colors=['red', 'green'], title="Z-stack {} registration".format(reg_method))

imsave(dir + base_fname.format(ref_channel+"_z-stack_reg"), reg_stack)

# - Apply this registration to the other channels:
for float_channel in float_channels:
    image = imread(dir + base_fname.format(float_channel))
    reg_image = apply_z_stack_trsf(image, z_stack_trsf)
    imsave(dir + base_fname.format(float_channel+"_z-stack_reg"), reg_image)
