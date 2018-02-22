import pandas as pd
from os.path import splitext

def get_nomenclature_name(nomenclature_file, sep=','):
    """
    Return a dictionary relating czi filenames to nomenclature names.
    Ussually from a csv file, separators can be set.

    Parameters
    ----------
    nomenclature_file : str
        the path to the nomenclature file to use
    sep : str, optional
        separator used in nomencalture file to separate data
    """
    n_data = pd.read_csv(nomenclature_file, sep=sep)[:-1]
    n_names = dict(zip(n_data['Name'], n_data['Nomenclature Name']))
    return n_names


def get_nomenclature_channel_fname(czi_fname, nomenclature_file, channel_name, ext='.inr.gz'):
    """
    Return the filename associated to the channel of interest for a known czi
    file and according to the nomenclature.

    Parameters
    ----------
    czi_fname : str
        the path & filename of the original CZI image
    nomenclature_file : str
        the path  & filename of the nomenclature file to use
    channel_name : str
        the name of the channel you want to access
    ext : str, optional
        the extension of the file containing the channel of interest
    """
    # - Read NOMENCLATURE file defining naming conventions:
    n_names = get_nomenclature_name(nomenclature_file)
    return n_names[czi_fname] + "/" + n_names[czi_fname] +  "_" + channel_name + ext

def get_nomenclature_segmentation_name(czi_fname, nomenclature_file, channel_name='PI', ext='.inr.gz'):
    """
    Return the filename associated to the channel of interest for a known czi
    file and according to the nomenclature.

    Parameters
    ----------
    czi_fname : str
        the path & filename of the original CZI image
    nomenclature_file : str
        the path  & filename of the nomenclature file to use
    channel_name : str
        the name of the channel you want to access
    ext : str, optional
        the extension of the file containing the channel of interest
    """
    # - Read NOMENCLATURE file defining naming conventions:
    n_names = get_nomenclature_name(nomenclature_file)
    return n_names[czi_fname] + "/" + n_names[czi_fname] + "_" + channel_name + "_segmented" + ext

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
