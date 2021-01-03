import sys
import time as t
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from color_space_converter import rgb2gry

from depthy.misc import Normalizer
from depthy.stereo.cost_aggregation import aggregate_costs
from depthy.stereo.feature_methods import compute_census
from depthy.stereo.dissimilarity_measures import abs_diff, xor_similarity


def compute_costs(l_img: np.ndarray,
                  r_img: np.ndarray,
                  disp_max: int = 64,
                  disp_min: int = 0,
                  csize: (int, int) = (7, 7),
                  method: str = None
                  ) -> (np.ndarray, np.ndarray):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param l_img: left image
    :param r_img: right image
    :param disp_max: maximum disparity
    :param disp_min: minimum disparity
    :param csize: size of the kernel for the census transform
    :param method: dissimilarity measure (default is sum of absolute differences)
    :return: 2 x np.ndarrays with the matching costs of H x W x D size
    """
    assert l_img.shape[0] == r_img.shape[0] and l_img.shape[1] == r_img.shape[1], 'shapes of left & right are different'
    assert disp_max > 0, 'maximum disparity must be greater than 0'

    h, w, c = l_img.shape if len(l_img.shape) == 3 else l_img.shape + (1,)
    offset = csize[0]//2

    # select function for dissimilarity measure
    dsim_fun = xor_similarity if method == 'xor' else abs_diff

    # cost volume computation
    print('\tSimilarity from %s method...' % method, end='')
    sys.stdout.flush()
    dawn = t.time()

    l_cost_volume = np.zeros(shape=(h, w, disp_max-disp_min), dtype=np.uint32)
    r_cost_volume = np.zeros(shape=(h, w, disp_max-disp_min), dtype=np.uint32)
    l_img_shift = np.zeros(shape=(h, w), dtype=np.int64)
    r_img_shift = np.zeros(shape=(h, w), dtype=np.int64)
    for d in range(disp_min, disp_max):

        # shift right image columns to left by current disparity value
        r_img_shift[:, offset+d:w-offset] = r_img[:, offset:w-d-offset]
        # compute similarity measure and append costs of current disparity to volume
        l_cost_volume[..., d] = dsim_fun(l_img, r_img_shift)

        # shift left image columns to right by current disparity value
        l_img_shift[:, offset:w-d-offset] = l_img[:, offset+d:w-offset]
        # compute similarity measure and append costs of current disparity to volume
        r_cost_volume[..., d] = dsim_fun(r_img, l_img_shift)

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return l_cost_volume, r_cost_volume


def semi_global_matching(
                    img_l: np.ndarray = None,
                    img_r: np.ndarray = None,
                    disp_max: int = 64,
                    disp_min: int = 0,
                    bsize: int = 3,
                    csize: int = 7,
                    p1: float = 10,
                    p2: float = 120,
                    feat_method: str = 'census',
                    dsim_method: str = 'xor',
                    blur_opt: bool = False,
                    medi_opt: bool = False,
                    *args, **kwargs
                    ) -> (np.ndarray, np.ndarray):
    """
    the semi-global matching algorithm.
    :return: tuple of two numpy arrays for left and right disparity frames
    """

    # gray scale conversion
    gray_l, gray_r = rgb2gry(img_l)[..., 0], rgb2gry(img_r)[..., 0]
    gray_l, gray_r = Normalizer(gray_l).uint16_norm(), Normalizer(gray_r).uint16_norm()

    # remove high frequency noise
    if blur_opt and csize > 0:
        print('\nBlur computation...')
        gray_l, gray_r = gaussian_filter(gray_l, csize//2), gaussian_filter(gray_r, csize//2)

    print('\nFeature computation...')
    gray_l, gray_r = compute_census(gray_l, gray_r, csize) if feat_method == 'census' else (gray_l, gray_r)

    print('\nCost computation...')
    cost_l, cost_r = compute_costs(gray_l, gray_r, disp_max, disp_min, (csize, csize), dsim_method)

    print('\nLeft aggregation computation...')
    cost_l = aggregate_costs(cost_l, p1, p2)
    print('\nRight aggregation computation...')
    cost_r = aggregate_costs(cost_r, p1, p2)

    disp_l = Normalizer(np.argmin(cost_l, axis=2)).uint8_norm()
    disp_r = Normalizer(np.argmin(cost_r, axis=2)).uint8_norm()

    if medi_opt:
        print('\nMedian filter...')
        disp_l = median_filter(disp_l, (bsize, bsize))
        disp_r = median_filter(disp_r, (bsize, bsize))

    print('\nFinished')

    return disp_l, disp_r


if __name__ == '__main__':

    from os.path import splitext
    import matplotlib.pyplot as plt
    from depthy.misc import load_img_file
    from tests.test_stereo import gt_measure

    print('\nLoad images...')
    img_l = load_img_file('../../examples/data/cones/im2.png')
    img_r = load_img_file('../../examples/data/cones/im6.png')

    dawn = t.time()

    l_disparity_map, r_disparity_map = semi_global_matching(img_l, img_r, feat_method='census', dsim_method='xor')

    dusk = t.time()
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))

    # save images
    output_name = '../../examples/data/disparity_map.png'
    plt.imsave(fname=splitext(output_name)[0]+'_l.png', arr=Normalizer(l_disparity_map).uint8_norm())#, cmap='gray'
    plt.imsave(fname=splitext(output_name)[0]+'_r.png', arr=Normalizer(r_disparity_map).uint8_norm())#, cmap='gray'

    evaluation = True  # evaluate disparity map with 3 pixel error
    if evaluation:
        l_gt_name = '../../examples/data/cones/disp2.png'
        r_gt_name = '../../examples/data/cones/disp6.png'
        l_gt = load_img_file(l_gt_name, norm_opt=False)
        r_gt = load_img_file(r_gt_name, norm_opt=False)
        print('\nEvaluate left disparity map...')
        recall = gt_measure(l_disparity_map, l_gt)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))
        print('\nEvaluate right disparity map...')
        recall = gt_measure(r_disparity_map, r_gt)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))
