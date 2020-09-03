import numpy as np

from depthy.lightfield.structure_tensor import local_structure_tensor
from depthy.misc import Normalizer, primal_dual_algo


def epi_depth(lf_img_arr: np.ndarray = None,
              lf_wid: int = 1,
              primal_opt: bool = True,
              perc_clip: float = 1) -> np.ndarray:
    """
    High-level function for depth map computation based on epipolar images from a light-field.

    :param lf_img_arr: light-field image array with odd number of light-field dimensions
    :param lf_wid: width of light-field rows and columns has to be an odd positive integer
    :param primal_opt: flag for usage of primal dual algorithm
    :param perc_clip: percentile for slope ratio extrema that will be clipped
    :return: disparity map
    """

    if lf_img_arr.shape[0] % 2 != 1 or lf_img_arr.shape[1] % 2 != 1 or lf_img_arr.shape[0] != lf_img_arr.shape[1]:
        raise Exception('Angular dimensions have to be odd and equally sized, but are %s ' % str(lf_img_arr.shape[:2]))

    if lf_wid % 2 != 1:
        raise Exception('lf_wid has to be an odd positive integer, but is %s ' % lf_wid)

    # array init
    disp_arr = np.zeros((2, lf_wid)+tuple(lf_img_arr.shape[2:]), dtype=np.float64)
    reli_arr = np.zeros((2, lf_wid)+tuple(lf_img_arr.shape[2:]), dtype=np.float64)

    # variable init
    lf_cen = lf_img_arr.shape[0] // 2
    lf_hwd = lf_wid // 2

    # iterate through horizontal and vertical directions
    for axis in [1, 0]:
        not_axis = not axis
        # iterate through light-field coordinates
        for angular_idx in range(lf_cen-lf_hwd, lf_cen+lf_hwd+1):
            # iterate through spatial coordinates
            for spatial_idx in range(lf_img_arr.shape[2 + not_axis]):

                # compute epipolar image (optionally swap axes to account for orientation)
                epi_img = extract_epi(lf_img_arr, spatial_idx, angular_idx, axis=not_axis)

                # extract local depth and reliability measure
                local_slopes, coherence, n = local_structure_tensor(epi_img, slope_method='eigen')

                # slice to single line - tbd: make use of excluded data
                ang_coord = angular_idx - lf_cen - lf_hwd
                if axis:
                    disp_arr[axis, ang_coord, spatial_idx, ...] = local_slopes[local_slopes.shape[0] // 2, ...]
                    reli_arr[axis, ang_coord, spatial_idx, ...] = coherence[coherence.shape[0] // 2, ...]
                else:
                    disp_arr[axis, ang_coord, :, spatial_idx, ...] = local_slopes[local_slopes.shape[0] // 2, ...]
                    reli_arr[axis, ang_coord, :, spatial_idx, ...] = coherence[coherence.shape[0] // 2, ...]

    # vstack results from angular width, then dstack results from vertical and horizontal directions
    disp_arr, reli_arr = np.dstack(np.vstack(disp_arr)), np.dstack(np.vstack(reli_arr))

    # clip disparities above and below percentiles
    disp_arr = Normalizer(disp_arr).perc_clip(perc_clip=perc_clip)

    # merge horizontal and vertical disparities using coherence
    disparity, _ = coherence_weighting(disp_arr, reli_arr)

    if primal_opt:
        # remove statistical image variances
        disparity, _ = primal_dual_algo(disparity, lambda_rof=1.5, theta=1, tau=.01, norm_l=7, max_iter=300)

    return disparity


def extract_epi(lf_img_arr: np.ndarray = None, spatial_idx: int = None, angular_idx: int = None, axis: bool = 0) \
        -> np.ndarray:
    """
    Compose epipolar image from light-field array given the reference coordinates.

    :param lf_img_arr: light-field image array
    :param spatial_idx: index for spatial image line (column or row)
    :param angular_idx: index for light-field dimension (column or row)
    :param axis: along vertical (False) or horizontal (True) direction
    :return: epipolar image as a copy of numpy array
    """

    # variable initializations
    angular_idx = int((lf_img_arr.shape[axis] - 1) / 2) if angular_idx is None else angular_idx
    spatial_idx = int((lf_img_arr.shape[axis+2] - 1) / 2) if spatial_idx is None else spatial_idx

    # compose epipolar image through slicing along provided axis
    epi_img = lf_img_arr[:, angular_idx, :, spatial_idx, ...] if axis else lf_img_arr[angular_idx, :, spatial_idx, ...]

    # return copy of numpy float object (copy as light field array may be visited several times)
    return epi_img.astype('float64').copy()


def coherence_weighting(disp_arr: np.ndarray = None, reli_arr: np.ndarray = None) -> [np.ndarray, np.ndarray]:
    """
    Merge disparity maps using weights from the coherence as a reliability measure.

    :param disp_arr: stacked array of disparity maps
    :param reli_arr: stacked array of coherence maps
    :return: weighted disparity, mean confidence
    """

    confidence = np.mean(reli_arr**2, axis=-1)
    disparity = np.mean(disp_arr*reli_arr**2, axis=-1) / confidence
    confidence = np.sqrt(confidence)

    return disparity, confidence
