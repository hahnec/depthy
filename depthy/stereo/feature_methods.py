import sys
import time as t
import numpy as np

from depthy.misc import Normalizer


def compute_census(img_l: np.ndarray = None, img_r: np.ndarray = None, offset: int = 7) -> (np.ndarray, np.ndarray):
    """
    Census feature extraction (for more details see https://en.wikipedia.org/wiki/Census_transform)

    :param img_l: left image
    :param img_r: right image
    :param offset: pixel offset on the four image borders
    :return: lcensus_values, rcensus_values
    """

    h, w, c = img_l.shape if len(img_l.shape) == 3 else img_l.shape + (1,)

    # convert to float
    img_l, img_r = Normalizer(img_l).norm_fun(), Normalizer(img_r).norm_fun()

    lcensus_values = np.zeros(shape=(h, w), dtype=np.uint64)
    rcensus_values = np.zeros(shape=(h, w), dtype=np.uint64)
    print('\tLeft and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # exclude pixels on the border (they will have no census values)
    for y in range(offset, h-offset):
        for x in range(offset, w-offset):

            # extract left block region and subtract current pixel intensity as offset from it
            image = img_l[y - offset:y + offset + 1, x - offset:x + offset + 1]
            roi_offset = image - img_l[y, x]
            # census calculation left image
            lcensus_values[y, x] = vectorized_census(roi_offset)

            # extract right block region and subtract current pixel intensity as offset from it
            image = img_r[y - offset:y + offset + 1, x - offset:x + offset + 1]
            roi_offset = image - img_r[y, x]
            # census calculation right image
            rcensus_values[y, x] = vectorized_census(roi_offset)

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return lcensus_values, rcensus_values


def vectorized_census(roi: np.ndarray = None) -> int:
    """
    Compute census in a numpy-vectorized fashion.

    :param roi: Region of Interest (RoI)
    :return: census value
    """

    if len(roi.shape) != 2:
        raise Exception('Data must be 2-dimensional')

    # binary census vector
    b = np.array(roi < 0).flatten()
    # remove central value
    central_idx = (roi.shape[0]*roi.shape[1])//2
    b = np.delete(b, central_idx)
    # convert binary vector to integer
    num = b.dot(1 << np.arange(b.size)[::-1])

    return num
