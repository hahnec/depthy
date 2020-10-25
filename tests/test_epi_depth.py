import unittest
from os.path import join, exists, basename
import numpy as np

from depthy.misc import DataDownloader, load_lf_arr, load_pfm, save_pfm, save_ply, disp2pts, plot_point_cloud
from depthy.lightfield import epi_depth


class EpiDepthTestCase(unittest.TestCase):

    def setUp(self):

        self.plot_opt = False

        # instantiate loader object
        self.loader = DataDownloader()
        self.fp = join(self.loader.root_path, 'examples', 'data')
        self.test_set = 'pens'  # default light-field test set
        self.load_data_set()

    def load_data_set(self, set_name: str = None):

        # retrieve data set (from web and/or hard drive)
        self.test_set = set_name if set_name is not None else self.test_set
        self.url_set = self.loader.uni_konstanz_urls(self.test_set)
        archive_fn = join(self.fp, basename(self.url_set))
        self.loader.download_data(self.url_set, fp=self.fp) if not exists(archive_fn) else None
        self.loader.extract_archive(archive_fn)

        # load images
        fnames = self.loader.find_archive_fnames(archive_fn, head_str=self.test_set, tail_str='png')
        fpaths = [join(self.fp, fname) for fname in fnames if fname.split('.png')[0][-3:].isdigit()]
        self.lf_img_arr = load_lf_arr(fpaths)

        # load ground truth reference
        gt_path = join(self.fp, self.test_set, 'gt_disp_lowres.pfm')
        self.gt_map, _ = load_pfm(gt_path) if exists(gt_path) else [np.zeros(self.lf_img_arr.shape[2:4]), None]

    def test_lf_depth_multi(self):

        norm_list = []
        for i in range(len(self.loader.set_names)):

            # pick next data set name and load it
            self.test_set = self.loader.set_names[i]
            self.load_data_set()

            # compute light-field depth based on epipolar images
            self.test_lf_depth_single()

            norm_list.append([self.test_set, self.img_l2_norm])
            # optional plot
            self.plot() if self.plot_opt else None

        print(norm_list)

    def test_lf_depth_single(self, norm_ref: float = float('Inf')):

        # compute light-field depth based on epipolar images
        self.disparity = epi_depth(lf_img_arr=self.lf_img_arr, lf_wid=1, perc_clip=1, primal_opt=True)

        # export depth map as pfm file
        save_pfm(self.disparity, file_path=join(self.fp, self.test_set + '.pfm'), scale=1)

        # export depth map as ply file
        lf_c = self.lf_img_arr.shape[0] // 2
        pts = disp2pts(disp_img=self.disparity, rgb_img=self.lf_img_arr[lf_c, lf_c, ...], focus_dist_mm=200)
        save_ply(pts, file_path=join(self.fp, self.test_set + '.ply'))

        # norm assertion
        self.assertTrue(self.img_l2_norm < norm_ref)

    @property
    def img_l2_norm(self):
        return np.sqrt(np.sum(np.power(self.disparity-self.gt_map, 2)))

    def plot(self):

        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title('disparity map')
        ax1.imshow(self.disparity, cmap='gray')
        ax2.set_title('ground truth')
        ax2.imshow(self.gt_map, cmap='gray')
        ax3.set_title('difference')
        ax3.imshow(self.disparity-self.gt_map, cmap='gray')
        plt.show()

        plot_point_cloud(disp_arr=self.disparity, rgb_img=self.lf_img_arr[4, 4, ...], down_scale=4)
        plt.show()

    def test_all(self):

        self.test_lf_depth_single(norm_ref=119)
