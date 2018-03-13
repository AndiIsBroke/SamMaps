for XP, SAM in [("E35", 4), ("E35", 6), ("E37", 5) , ("E37", 7)]:
    %run PI_segmentation_from_nuclei.py $XP $SAM


for XP, SAM in [("E35", 4), ("E35", 6), ("E37", 5) , ("E37", 7)]:
    %run registration_on_last_timepoint.py $XP $SAM 'rigid'




import os

PIN_channel = True
PIN_func_fname = "PIN_polarity"
sampling_distance2wall = 0.6


for XP, SAM in [("E35", 4), ("E35", 6), ("E37", 5) , ("E37", 7)]:
    for tp in [0, 5, 10, 14]:
        %run PIN_quantif.py $XP $SAM $tp $sampling_distance2wall



for XP, SAM in [("E35", 4), ("E35", 6), ("E37", 5) , ("E37", 7)]:
    for tp in [0, 5, 10, 14]:
        %run PIN_quantif_vector_map.py $XP $SAM $tp $PIN_func_fname $sampling_distance2wall



for XP, SAM in [("E35", 4), ("E35", 6), ("E37", 5) , ("E37", 7)]:
    import sys, platform
    if platform.uname()[1] == "RDP-M7520-JL":
        SamMaps_dir = '/data/Meristems/Carlos/SamMaps/'
        dirname = "/data/Meristems/Carlos/PIN_maps/"
    elif platform.uname()[1] == "calculus":
        SamMaps_dir = '/projects/SamMaps/scripts/SamMaps_git/'
        dirname = "/projects/SamMaps/"
    else:
        raise ValueError("Unknown custom path to 'SamMaps' for this system...")
    sys.path.append(SamMaps_dir+'/scripts/TissueLab/')

    from nomenclature import splitext_zip
    from nomenclature import get_res_img_fname
    from nomenclature import get_nomenclature_channel_fname

    image_dirname = dirname + "nuclei_images/"
    nomenclature_file = SamMaps_dir + "nomenclature.csv"
    base_fname = "qDII-CLV3-PIN1-PI-{}-LD-SAM{}".format(XP, SAM)
    czi_fname = base_fname + "-T{}.czi"
    time_steps = [0, 5, 10, 14]

    ext = '.inr.gz'
    cmd = "convert -delay 100"
    suffix = '{}-D_{}-{}scaled_arrowheads'.format(PIN_func_fname[3:], sampling_distance2wall, "PIN_signal-" if PIN_channel else "")
    for tp in time_steps:
        ref_czi_fname = czi_fname.format(tp)
        signal_path, signal_fname = get_nomenclature_channel_fname(ref_czi_fname, nomenclature_file, "PIN1")
        if tp != time_steps[-1]:
            # Get RIDIG registered on last time-point filename:
            signal_fname = get_res_img_fname(signal_fname, time_steps[-1], tp, 'rigid')
        png_name = image_dirname + splitext_zip(signal_fname)[0] + suffix + '.png'
        cmd += " {}".format(png_name)

    cmd += " {}.gif".format(image_dirname + base_fname + suffix)
    print cmd

    os.system(cmd)

