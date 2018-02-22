import pandas as pd
from os.path import splitext

def get_nomenclature_name(nomenclature_file, sep=','):
    """
    Return the nomenclature names from a csv file.
    """
    n_data = pd.read_csv(nomenclature_file, sep=sep)[:-1]
    n_names = dict(zip(n_data['Name'], n_data['Nomenclature Name']))
    return n_names


def get_nomenclature_channel_fname(czi_fname, nomenclature_file, channel_name, ext='.inr.gz'):
    """
    Return
    """
    # - Read NOMENCLATURE file defining naming conventions:
    n_names = get_nomenclature_name(nomenclature_file)
    return n_names[czi_fname] + "/" + n_names[czi_fname] +  "_" + channel_name + ext


def splitext_zip(fname):
    """
    Returns filename and extension of fname
    Unsensitive to 'gz' or 'zip' extensions
    """
    base_fname, ext = splitext(fname)
    if ext == '.gz' or ext == '.zip':
        base_fname, ext2 = splitext(base_fname)
        ext = ''.join([ext2, ext])
    return base_fname, ext
