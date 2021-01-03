import sys
import time as t
import numpy as np

from depthy.misc import Normalizer


def compute_census(l_img, r_img, csize=7, truncate=False):
    """
    census calculation (see https://en.wikipedia.org/wiki/Census_transform)
    :param l_img:
    :param r_img:
    :param csize:
    :return: lcensus_values, rcensus_values
    """

    h, w, c = l_img.shape if len(l_img.shape) == 3 else l_img.shape + (1,)
    y_offset, x_offset = csize//2, csize//2

    # convert to float
    l_img, r_img = Normalizer(l_img).norm_fun(), Normalizer(r_img).norm_fun()

    lcensus_values = np.zeros(shape=(h, w), dtype=np.uint64)
    rcensus_values = np.zeros(shape=(h, w), dtype=np.uint64)
    print('\tLeft and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # exclude pixels on the border (they will have no census values)
    for y in range(y_offset, h-y_offset):
        for x in range(x_offset, w-x_offset):

            # extract left block region and subtract current pixel intensity as offset from it
            image = l_img[y-y_offset:y+y_offset+1, x-x_offset:x+x_offset+1]
            roi_offset = image - l_img[y, x]
            # census calculation left image
            lcensus_values[y, x] = vectorized_census(roi_offset)

            # extract right block region and subtract current pixel intensity as offset from it
            image = r_img[y-y_offset:y+y_offset+1, x-x_offset:x+x_offset+1]
            roi_offset = image - r_img[y, x]
            # census calculation right image
            rcensus_values[y, x] = vectorized_census(roi_offset)

    if truncate:
        lcensus_values, rcensus_values = np.uint8(lcensus_values), np.uint8(rcensus_values)

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return lcensus_values, rcensus_values


def vectorized_census(roi):

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
