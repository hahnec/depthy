import unittest
import numpy as np
import matplotlib.pyplot as plt

from depthy.lightfield.depth_label_vector import create_angle_masks, local_label_optimization, get_labels
from depthy.lightfield import local_structure_tensor
from depthy.misc import load_img_file

from tests.test_structure_tensor import StructTensorTestCase


class DepthLabelingTestCase(StructTensorTestCase, unittest.TestCase):

    def setUp(self):

        # data preparation
        self.epi_img = load_img_file('../examples/data/epi_img.png')
        self.epi_img /= self.epi_img.max()

        # settings
        self.plot_opt = True
        self.label_num = 25
        self.label_method = 'sqr'
        self.max_iter = 10

    def test_create_masks(self):

        labels = np.random.rand(9)*180-90
        labels_flip = [x for t in zip(labels, labels*-1) for x in t]
        masks = create_angle_masks(labels)

        self.assertEqual(masks.shape[0], len(labels)*2, msg='Number of masks not as expected')

        if self.plot_opt:
            fig, axs = plt.subplots(1, masks.shape[0])
            for i, (label, mask) in enumerate(zip(labels_flip, masks)):
                axs[i].imshow(mask)
                axs[i].set_title(str(round(label, 2)))
            plt.show()

    def test_get_labels(self):

        for label_num in range(0, 20, 2):
            for m in ['sqr', 'hist', 'disp', 'unrecognized']:

                local_disp, _, _ = local_structure_tensor(self.epi_img, slope_method='eigen')

                labels = get_labels(local_disp, label_num=label_num, label_method=m)

                try:
                    self.assertTrue(label_num == len(labels),
                                    'Requested label number not matching result for %s with method %s' % (label_num, m))
                except (TypeError, AssertionError) as e:
                    if label_num > 2:
                        raise e

    def test_local_label_optimization_real(self):

        local_disp, coherence, n = local_structure_tensor(self.epi_img, slope_method='eigen')

        labels = get_labels(local_disp, label_num=self.label_num, label_method=self.label_method)

        local_labels = local_label_optimization(local_disp, coherence=coherence, labels=labels, max_iter=self.max_iter)

        if self.plot_opt:
            fig, axs = plt.subplots(3, 1)
            axs[0].imshow(self.epi_img, cmap='gray', label='epipolar image')
            axs[1].imshow(local_disp/local_disp.max(), cmap='gray', label='stucture tensor')
            axs[2].imshow(local_labels/local_labels.max(), cmap='gray', label='depth labels')
            fig.legend()
            plt.show()

    def test_local_label_optimization_synth(self):

        test_epis = self.create_test_epis(m=49)
        np.random.seed(32)

        for _, test_epi in test_epis:

            local_disp, coherence, n = local_structure_tensor(test_epi, slope_method='eigen')

            # disturb disparity
            pmask_disp = np.array(np.random.rand(test_epi.shape[0], test_epi.shape[0])>.9, dtype=float)*.2
            nmask_disp = np.array(np.random.rand(test_epi.shape[0], test_epi.shape[0])>.9, dtype=float)*.2
            noise_disp = local_disp + pmask_disp[..., None] - nmask_disp[..., None]

            local_labels = local_label_optimization(noise_disp, coherence, max_iter=self.max_iter)

            ret = np.sum((local_disp - local_labels)**2)**.5
            self.assertTrue(ret, msg='Failed local constraint optimization for EPI')

            if self.plot_opt:
                fig, axs = plt.subplots(1, 3)
                axs[0].set_title('EPI', fontsize=8)
                axs[0].imshow(test_epi, cmap='gray')
                axs[1].set_title('Struct. Tensor', fontsize=8)
                axs[1].imshow(local_disp, cmap='gray')
                axs[2].set_title('Struct. Tensor w/ local constraint', fontsize=8)
                axs[2].imshow(local_labels, cmap='gray')
                plt.show()
