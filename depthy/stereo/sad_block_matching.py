import numpy as np

from depthy.stereo.dissimilarity_measures import sad_vectorized
from depthy.stereo.helper_funs import auto_disp_limits, color_channel_adjustment, precise_sub_disp


def sad_block_match_vector(img_l, img_r, disp_max=8, disp_min=0, ws=16, prec=1, *args, **kwargs) \
        -> (np.ndarray, np.ndarray):

    img_l, img_r = color_channel_adjustment(img_l, img_r)
    disp_map = np.zeros(img_l.shape[:2])

    for v in range(img_r.shape[1]):

        # reset cost vector
        cost_vec = np.ones([img_r.shape[0], disp_max-disp_min]) * disp_max

        # select new reference patch
        kernel = img_r[:, v:v+ws, ...]

        # seek corresponding patch (via block matching)
        for d in range(disp_min, disp_max):
            if 0 <= v+d <= img_r.shape[1] - ws:
                cost_vec[:, d - disp_min] = sad_vectorized(img_l[:, v + d:v + d + ws, ...], kernel)
            elif 0 > v+d:
                crop_kernel = kernel[:, abs(v+d):, ...]
                cost_vec[:, d - disp_min] = sad_vectorized(img_l[:, :ws, ...], crop_kernel)
            elif img_r.shape[1] - ws < v+d < img_r.shape[1]:
                crop_kernel = kernel[:, (v+d)-img_r.shape[1]:, ...]
                cost_vec[:, d - disp_min] = sad_vectorized(img_l[:, v + d:, ...], crop_kernel)

        # compute sub-disparity precision
        k = precise_sub_disp(cost_vec, prec)

        # place minimum cost index to disparity map
        disp_map[:, v] = k + disp_min

    return disp_map, disp_map


def sad_block_match_iter2d(img_l, img_r, disp_max=8, disp_min=0, ws=16, prec=1, *args, **kwargs) \
        -> (np.ndarray, np.ndarray):

    img_l, img_r = color_channel_adjustment(img_l, img_r)
    disp_map = np.zeros(img_l.shape[:2])
    ws = ws//2+1
    print_opt = kwargs['print_opt'] if 'print_opt' in kwargs else True

    print('\nCompute block matching cost...')

    for u in range(img_r.shape[0]):
        # percentage
        print('\r%s ' % str(round(u/img_r.shape[0]*100))+'%', end='') if print_opt else None
        for v in range(img_r.shape[1]):

            # reset cost vector
            cost_vec = np.ones(disp_max-disp_min) * disp_max

            # border-safe indices for reference kernel
            u_s = u-ws if u-ws >= 0 else 0
            v_s = v-ws if v-ws >= 0 else 0
            u_e = u+ws if u+ws <= img_r.shape[0] else img_r.shape[0]
            v_e = v+ws if v+ws <= img_r.shape[1] else img_r.shape[1]

            # select new reference patch
            kernel = img_r[u_s:u_e, v_s:v_e, ...]

            # seek corresponding patch (via SAD block matching)
            for d in range(disp_min, disp_max):
                # within image borders
                if 0 <= v_s+d < v_e+d <= img_l.shape[1]:
                    cost_vec[d-disp_min] = np.sum(np.abs(img_l[u_s:u_e, v_s+d:v_e+d, ...] - kernel))
                # left image border
                elif 0 > v_s+d:
                    crop_kernel = kernel[:, abs(d):, ...]
                    cost_vec[d-disp_min] = np.sum(np.abs(img_l[u_s:u_e, :v_e+d, ...] - crop_kernel))
                # right image border
                elif v_e+d > img_l.shape[1]:
                    crop_kernel = kernel[:, :img_l.shape[1]-(v_e+d), ...]
                    cost_vec[d-disp_min] = np.sum(np.abs(img_l[u_s:u_e, v_s+d:img_l.shape[1], ...] - crop_kernel))

            # compute sub-disparity precision
            k = precise_sub_disp(cost_vec, prec)

            # place minimum cost index to disparity map
            disp_map[u, v] = k + disp_min

    print('\n\nFinished')

    return disp_map, disp_map


if '__main__' == __name__:

    import matplotlib.pyplot as plt
    from depthy.misc.io_functions import load_img_file

    img_l = load_img_file('../../examples/data/cones/im2.png')
    img_r = load_img_file('../../examples/data/cones/im6.png')

    disp_lim = auto_disp_limits(img_l, img_r)
    print('\nDetected disparity range [min, max] amounts to %s.' % disp_lim)

    disp_map, _ = sad_block_match_vector(img_l, img_r, disp_max=disp_lim[1], disp_min=disp_lim[0], ws=25, prec=1)

    plt.imshow(disp_map/disp_map.max())
    plt.show()

    disp_map, _ = sad_block_match_iter2d(img_l, img_r, disp_max=disp_lim[1], disp_min=disp_lim[0]-1, ws=25, prec=10)

    plt.imshow(disp_map/disp_map.max())
    plt.show()
