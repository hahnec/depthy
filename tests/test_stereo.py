import unittest
from os.path import join, basename

import numpy as np

from depthy.stereo import auto_disp_limits, sad_block_match_vector, sad_block_matching, semi_global_matching
from depthy.misc import DataDownloader, load_img_file, save_pfm, save_ply, disp2pts, plot_point_cloud, Normalizer


class StereoTestCase(unittest.TestCase):

    def setUp(self):

        self.plot_opt = False

        # instantiate loader object
        self.loader = DataDownloader(print_opt=False)
        self.fp = join(self.loader.root_path, 'examples', 'data')
        self.test_set = 'cones'  # default stereo test set
        self.img_l, self.img_r = np.array([]), np.array([])
        self.load_data_set()
        self.disp_l, self.disp_r = np.zeros_like(self.img_l), np.zeros_like(self.img_r)

    @property
    def fname_set(self):
        return self.test_set + '-png-2.zip'

    def load_data_set(self, set_name: str = None):

        # retrieve data set (from hard drive)
        self.test_set = set_name if set_name is not None else self.test_set
        archive_fn = join(self.fp, basename(self.fname_set))
        self.loader.extract_archive(archive_fn)
        fnames = self.loader.find_archive_fnames(archive_fn, head_str='', tail_str='png')

        # load images
        fpaths = [join(self.fp, fname) for fname in fnames if basename(fname).split('.png')[0].startswith('im')]
        self.img_l, self.img_r = [load_img_file(fpath) for fpath in fpaths]

        # load ground truth references
        fpaths = [join(self.fp, fname) for fname in fnames if basename(fname).split('.png')[0].startswith('disp')]
        self.gt_l, self.gt_r = [load_img_file(fpath, norm_opt=False) for fpath in fpaths]

    def test_stereo_multi(self):

        norm_list = []
        for i in range(len(self.loader.set_names)):

            # pick next data set name and load it
            self.test_set = self.loader.set_names[i]
            self.load_data_set()

            # compute stereo depth based on provided test set
            self.test_stereo_single()

            norm_list.append([self.test_set, self.img_l2_norm])
            # optional plot
            self.plot() if self.plot_opt else None

        print(norm_list)

    def test_stereo_methods(self):

        norm_list = []
        function_list = [sad_block_match_vector, sad_block_matching, semi_global_matching]
        for func in function_list:

            # compute stereo depth based on provided method
            self.test_stereo_single(stereo_func=func)

            norm_list.append([self.test_set, self.img_l2_norm])
            # optional plot
            self.plot() if self.plot_opt else None

        print(norm_list)

    def test_stereo_single(self, norm_ref: float = float('Inf'), stereo_func: classmethod = sad_block_match_vector):

        # auto-estimate depth limits
        disp_lim = auto_disp_limits(self.img_l, self.img_r)

        # compute depth based on stereo images
        self.disp_l, self.disp_r = stereo_func(self.img_l, self.img_r, disp_max=disp_lim[1], print_opt=False)

        # export depth map as pfm file
        save_pfm(self.disp_l, file_path=join(self.fp, self.test_set + '_l_' + stereo_func.__name__ + '.pfm'), scale=1)
        save_pfm(self.disp_r, file_path=join(self.fp, self.test_set + '_r_' + stereo_func.__name__ + '.pfm'), scale=1)

        # export depth map as ply file
        pts_l = disp2pts(disp_img=self.disp_l, rgb_img=self.img_l)
        pts_r = disp2pts(disp_img=self.disp_r, rgb_img=self.img_r)
        save_ply(pts_l, file_path=join(self.fp, self.test_set + '_l_' + stereo_func.__name__ + '.ply'))
        save_ply(pts_r, file_path=join(self.fp, self.test_set + '_r_' + stereo_func.__name__ + '.ply'))

        for disp, gt in [(self.disp_l, self.gt_l), (self.disp_r, self.gt_r)]:
            # shape assertion
            self.assertTrue(len(disp.shape) == 2, msg="Depth map is not a 2-D array.")
            # norm assertion
            norm = self.img_l2_norm(disp, gt)
            self.assertTrue(norm < norm_ref, msg='Depth quality below limit.')

    @staticmethod
    def img_l2_norm(img, gt):
        return np.round(np.sqrt(np.sum(np.power(img - gt, 2))), 3)

    def plot(self):

        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title('disparity map')
        ax1.imshow(self.disp_l, cmap='gray')
        ax2.set_title('ground truth')
        ax2.imshow(self.gt_l, cmap='gray')
        ax3.set_title('difference')
        ax3.imshow(self.disp_l - self.gt_l, cmap='gray')
        plt.show()

        plot_point_cloud(disp_arr=self.disp_l, rgb_img=self.img_l, down_scale=4)
        plt.show()

    def test_opencv_benchmark(self, stereo_method=semi_global_matching):

        try:
            import cv2
        except ImportError:
            print('Please install package opencv-python-headless for the benchmark test')
            return False

        # normalize to 8 bit unsigned integer (as OpenCV's SGM method only accepts uint8)
        img_l, img_r = Normalizer(self.img_l).uint8_norm(), Normalizer(self.img_r).uint8_norm()

        disp_min, disp_max = 0, 64
        size_k = 3
        p1 = 10
        p2 = 120

        # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
        cv_sgm = cv2.StereoSGBM_create(
            minDisparity=disp_min,
            numDisparities=disp_max-disp_min,
            blockSize=size_k*2+1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=3,
            disp12MaxDiff=-1,
            P1=p1,
            P2=p2
        )

        opcv_l = cv_sgm.compute(img_l, img_r)
        opcv_r = cv_sgm.compute(img_r, img_l)

        disp_l, disp_r = stereo_method(
            img_l, img_r,
            disp_max=disp_max, disp_min=disp_min,
            p1=p1, p2=p2,
            feat_method='census', dsim_method='xor',
            size_k=size_k,
            blur_opt=False, medi_opt=True
        )

        our_norm = self.img_l2_norm(Normalizer(disp_l).uint16_norm(), Normalizer(self.gt_l).uint16_norm())
        cv2_norm = self.img_l2_norm(Normalizer(opcv_l).uint16_norm(), Normalizer(self.gt_l).uint16_norm())

        print('\nL2-norm for OpenCV result is %s and our %s yields %s.' % (cv2_norm, stereo_method.__name__, our_norm))

    def test_all(self):

        self.test_stereo_multi()
        self.test_stereo_methods()
        self.test_opencv_benchmark()


def gt_measure(disparity, gt):
    """
    computes the recall of the disparity map
    :param disparity: disparity image
    :param gt: ground-truth image
    :return: rate of correct predictions
    """

    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    return float(correct) / gt.size
