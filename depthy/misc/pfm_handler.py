import numpy as np
import re
import sys


def load_pfm(file_path: str = None) -> [np.ndarray, float]:
    """
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.

    :param file_path: file path to pgm file
    :return: grayscale image array, scale value
    """

    file_obj = open(file_path, 'rb')
    header = file_obj.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file_obj.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file_obj.readline().decode('utf-8').rstrip())
    if scale < 0:   # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'    # big-endian
    data = np.fromfile(file_obj, endian + 'f')
    file_obj.close()

    # reshape to numpy array
    shape = (height, width, 3) if color else (height, width)
    pfm_arr = np.reshape(data, shape)
    pfm_arr = np.flipud(pfm_arr)

    return pfm_arr, scale


def save_pfm(img_arr: np.ndarray = None,
             file_path: str = None,
             scale: float = 1) -> bool:
    """
    Save a Numpy array to a PFM file.

    :param img_arr: image array
    :param file_path: file path string
    :param scale: scale value
    :return: boolean
    """

    # ensure data type is float32
    img_arr = img_arr.astype(np.float32) if img_arr.dtype.name != 'float32' else img_arr

    # normalize image array
    img_arr = (img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr))

    # flip array upside down
    img_arr = np.flipud(img_arr)

    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:   # color image
        color = True
    elif len(img_arr.shape) == 2 or len(img_arr.shape) == 3 and img_arr.shape[2] == 1:    # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    # write to hard drive
    with open(file_path, 'w') as file_obj:
        file_obj.write('PF\n' if color else 'Pf\n')
        file_obj.write('%d %d\n' % (img_arr.shape[1], img_arr.shape[0]))
        endian = img_arr.dtype.byteorder
        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale
        file_obj.write('%f\n' % scale)
        img_arr.tofile(file_obj)

    return True
