import numpy as np
from color_space_converter import rgb2gry
from scipy.interpolate import interp1d, interp2d


def auto_disp_limits(img_l: np.ndarray = None, img_r: np.ndarray = None) -> [int, int]:
    """
    Estimate disparity range with maximum and minimum

    :param img_l: left image
    :param img_r: right image
    :return: disp_max, disp_min
    """

    max_ran = img_l.shape[0]//8*2
    if len(img_l.shape) == 3:
        img_l_shift = np.pad(img_l, ((0, 0), (max_ran//2, max_ran//2), (0, 0)))
    elif len(img_l.shape) == 2:
        img_l_shift = np.pad(img_l, ((0, 0), (max_ran//2, max_ran//2)))
    else:
        raise Exception('Image shape unrecognized')

    global_diff_list = []
    for d in range(max_ran):
        if len(img_r.shape) == 3:
            img_r_shift = np.pad(img_r, ((0, 0), (d, max_ran - d), (0, 0)))
        elif len(img_l.shape) == 2:
            img_r_shift = np.pad(img_r, ((0, 0), (d, max_ran - d)))
        else:
            raise Exception('Image shape unrecognized')
        global_sad = np.sum(np.abs(img_l_shift - img_r_shift))
        global_diff_list.append(global_sad)

    # pinpoint indices where SAD gradient changes its direction
    idxs = np.where(np.diff(np.sign(np.gradient(global_diff_list))) > 0)[0]
    idxs = [(idx-max_ran//2)*3 for idx in idxs]

    return idxs


def color_channel_adjustment(img_l: np.ndarray = None, img_r: np.ndarray = None) -> (np.ndarray, np.ndarray):
    """
    Validate channels of stereo image pairs match and reduce to monochromatic channel information

    :param img_l: left image
    :param img_r: right image
    :return: img_l, img_r of H x W x 1 size
    """

    if len(img_l.shape) == 3 and len(img_r.shape) == 3:
        img_l, img_r = rgb2gry(img_l)[..., 0], rgb2gry(img_r)[..., 0]
    elif len(img_l.shape) == 2 and len(img_r.shape) == 2:
        pass
    else:
        raise Exception('Image color channel mismatch')

    return img_l, img_r


def precise_sub_disp(cost_vec: np.ndarray = None, prec: float = 1):

    # get disparity from index where SAD is minimum
    if prec > 1:
        # find minimum (with sub-pixel precision using interpolation)
        if len(cost_vec.shape) == 1:
            # for 1-dimensional cost vector
            x_new = np.arange(0, len(cost_vec) - 1, 1. / prec)
            i_fun = interp1d(range(0, len(cost_vec)), cost_vec, kind='cubic')
            y_new = i_fun(x_new)
            k = x_new[np.argmin(y_new, axis=-1)]
        elif len(cost_vec.shape) == 2:
            # for 2-dimensional cost vector
            x_new = np.arange(0, cost_vec.shape[1] - 1, 1. / prec)
            i_fun = interp2d(range(0, cost_vec.shape[1]), range(0, cost_vec.shape[0]), cost_vec, kind='cubic')
            y_new = i_fun(x_new, np.arange(0, cost_vec.shape[0]))
            k = x_new[np.argmin(y_new, axis=-1)]
        else:
            raise Exception('Cost shape unrecognized')
    else:
        # get integer index
        k = np.argmin(np.array(cost_vec), axis=-1)

    return k
