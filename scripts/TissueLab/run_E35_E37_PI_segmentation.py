import os

E35_dir = '/data/Meristems/Carlos/PIN_maps/microscopy/20171110 MS-E35 LD qDII-CLV3-PIN1-PI/RAW/'
E35_fnames = os.listdir(E35_dir)
E35_fnames = [E35_dir + f for f in E35_fnames if f.find('SAM4')!=-1 or f.find('SAM6')!=-1]

E37_dir = '/data/Meristems/Carlos/PIN_maps/microscopy/20171113 MS-E37 LD qDII-CLV3-PIN1-PI/RAW/'
E37_fnames = os.listdir(E37_dir)
E37_fnames = [E37_dir + f for f in E37_fnames]

czi_fnames = E35_fnames + E37_fnames

for czi in czi_fnames:
    %run -t /data/Meristems/Carlos/SamMaps/scripts/TissueLab/PI_segmentation_from_nuclei.py '$czi'


