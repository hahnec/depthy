import unittest
import numpy as np
import matplotlib.pyplot as plt

from depthy.lightfield.depth_label_vector import create_angle_masks, local_label_optimization
from depthy.lightfield import local_structure_tensor
from depthy.misc import load_img_file

from tests.test_structure_tensor import StructTensorTestCase


class DepthLabelingTestCase(StructTensorTestCase, unittest.TestCase):

    def setUp(self):

        # data preparation
        self.epi_img = load_img_file('../examples/data/epi_img.png')
        #self.epi_img /= self.epi_img.max()

        self.plot_opt = False

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

    def test_local_label_optimization_real(self):

        local_disp, coherence, n = local_structure_tensor(self.epi_img, slope_method='eigen')
        print(local_disp.dtype, local_disp.shape)

        local_labels = local_label_optimization(local_disp, coherence, n, label_num=9, max_iter=2 * 10 ** 1)
        print(local_labels.dtype, local_labels.shape)

    def test_local_label_optimization_synth(self):

        test_epis = self.create_test_epis(m=49)
        np.random.seed(32)

        for _, test_epi in test_epis[2:]:

            local_disp, coherence, n = local_structure_tensor(test_epi, slope_method='eigen')
            print(local_disp.dtype, local_disp.shape)

            # disturb disparity
            pmask_disp = np.array(np.random.rand(test_epi.shape[0], test_epi.shape[0])>.9, dtype=float)*.2
            nmask_disp = np.array(np.random.rand(test_epi.shape[0], test_epi.shape[0])>.9, dtype=float)*.2
            noise_disp = local_disp + pmask_disp[..., None] - nmask_disp[..., None]

            local_labels = local_label_optimization(noise_disp, coherence, n, label_num=3, max_iter=1 * 10 ** 1)
            print(local_labels.dtype, local_labels.shape)

            ret = np.sum((local_disp - local_labels)**2)**.5
            self.assertTrue(ret, msg='Failed local constraint optimization for EPI')
