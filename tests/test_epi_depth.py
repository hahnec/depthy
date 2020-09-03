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
        self.set_name = 'pens'  # default light-field test set
        self.load_data_set()

    def load_data_set(self, set_name: str = None):

        # retrieve light-field data set (from web and/or hard drive)
        self.set_name = set_name if set_name is not None else self.set_name
        self.set_url = self.loader.uni_konstanz_urls(self.set_name)
        archive_fn = join(self.fp, basename(self.set_url))
        self.loader.download_data(self.set_url, fp=self.fp) if not exists(archive_fn) else None
        self.loader.extract_archive(archive_fn)

        # load light-field images
        fnames = self.loader.find_archive_fnames(archive_fn, head_str=self.set_name, tail_str='png')
        fpaths = [join(self.fp, fname) for fname in fnames if fname.split('.png')[0][-3:].isdigit()]
        self.lf_img_arr = load_lf_arr(fpaths)

        # load ground truth reference
        gt_path = join(self.fp, self.set_name, 'gt_disp_lowres.pfm')
        self.gt_map, _ = load_pfm(gt_path) if exists(gt_path) else [np.zeros(self.lf_img_arr.shape[2:4]), None]

    def test_lf_depth_multi(self):

        norm_list = []
        for i in range(len(self.loader.set_names)):

            # pick next data set name and load it
            self.set_name = self.loader.set_names[i]
            self.load_data_set()

            # compute light-field depth based on epipolar images
            self.test_lf_depth_single()

            norm_list.append([self.set_name, self.img_l2_norm])
            # optional plot
            self.plot() if self.plot_opt else None

        print(norm_list)

    def test_lf_depth_single(self, norm_ref: float = float('Inf')):

        # compute light-field depth based on epipolar images
        self.disparity = epi_depth(lf_img_arr=self.lf_img_arr, lf_wid=1, perc_clip=1, primal_opt=True)

        # export depth map as pfm file
        save_pfm(self.disparity, file_path=join(self.fp, self.set_name + '.pfm'), scale=1)

        # export depth map as ply file
        lf_c = self.lf_img_arr.shape[0] // 2
        pts = disp2pts(disp_img=self.disparity, rgb_img=self.lf_img_arr[lf_c, lf_c, ...])
        save_ply(pts, file_path=join(self.fp, self.set_name + '.ply'))

        # assert using norm
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
