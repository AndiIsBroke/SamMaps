import argparse
from timagetk.components import SpatialImage, imread, imsave

import sys, platform
if platform.uname()[1] == "RDP-M7520-JL":
    sys.path.append('/data/Meristems/Carlos/SamMaps/scripts/TissueLab/')
elif platform.uname()[1] == "calculus":
    sys.path.append('/projects/SamMaps/scripts/SamMaps_git/scripts/TissueLab/')
else:
    raise ValueError("Unknown custom path to 'SamMaps/scripts/TissueLab/' for this system...")

from nomenclature import splitext_zip

################################################################################
### Examples :
################################################################################
#$ python inr2tif.py inr_img_1.inr inr_img_2.inr

def main():
    # - Argument parsing:
    parser = argparse.ArgumentParser(description="Convert inr images to tif format.")
    # -- MANDATORY arguments:
    parser.add_argument('inr', type=list, nargs='+',
                        help='Image or list of images to convert.')
    input_args = parser.parse_args()

    # Check "images" (list) to convert to *.inr:
    img_list = [''.join(fname) for fname in input_args.inr]
    print img_list
    try:
        assert img_list != []
    except:
        raise TypeError("Could not understand the list of provided filenames.")

    # Performs czi to inr convertion:
    for input_img in img_list:
        tif_fname = splitext_zip(input_img)[0] + '.tif'
        imsave(tif_fname, imread(input_img))

if __name__ == '__main__':
    main()
