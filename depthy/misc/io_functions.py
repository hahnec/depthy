import os, sys
import imageio
import numpy as np
from typing import Union
import tkinter as tk
import re

from .normalizer import Normalizer

GEN_IMG_EXTS = ('bmp', 'png', 'tiff', 'tif', 'jpeg', 'jpg')


def load_img_file(file_path: str = None) -> np.ndarray:
    """
    Load image file to numpy array.

    :param file_path: absolute path of file name
    :return: image as numpy array
    """

    file_type = file_path.split('.')[-1]

    if any(file_type in ext for ext in GEN_IMG_EXTS):
        try:
            img = imageio.imread(uri=file_path, format=file_type)
        except OSError:
            # support load of truncated images
            from PIL import ImageFile, Image
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(file_path)
        except AttributeError:
            raise TypeError('File type %s not recognized' % file_type)
    else:
        raise TypeError('File type %s not recognized' % file_type)

    # normalize (convert to numpy array)
    img = Normalizer(img).type_norm()

    return img


def load_lf_arr(fnames: Union[np.ndarray, list] = None) -> np.ndarray:
    """
    Load light-field images from a list of file names (absolute path) in ascending order and reshape the array to
    equally sized views. Reshaping of light-field dimensions is based on :math:`\\sqrt{N}` where :math:`N` is the total
    number of provided file names.

    :param fnames: list of file names (full absolute path)
    :return: numpy array of light-field images
    """

    # sort file names by removing non-numeric characters
    fnames = sorted(fnames, key=lambda x: int(re.sub("[^0-9]", "", os.path.basename(x).split(".")[0])))

    # load file names
    img_list = [load_img_file(fname) for fname in fnames]

    # light-field dimension reshaping
    lf_dim = len(img_list) ** .5
    assert lf_dim % 2 == 1, 'Reshaping image list to light-field array failed due to size %s.' % len(img_list)
    lf_img_arr = np.array(img_list).reshape((int(lf_dim), int(lf_dim)) + np.shape(img_list)[-3:])

    return lf_img_arr


def select_file(init_dir: str = None, title: str = '', root: tk.Tk = None):
    """ get file path from tkinter dialog """

    # consider initial directory if provided
    init_dir = os.path.expanduser('~/') if not init_dir else init_dir

    # import tkinter while considering Python version
    try:
        if sys.version_info > (3, 0):
            import tkinter as tk
            from tkinter.filedialog import askopenfilename, Open
        else:
            import Tkinter as tk
            from tkFileDialog import askopenfilename, Open
    except ImportError:
        raise ImportError('Please install tkinter package.')

    # open window using tkinter
    root = tk.Tk() if root is None else root
    root.withdraw()
    file_path = askopenfilename(parent=root, initialdir=[init_dir], title=title)

    return file_path
