import unittest
import numpy as np

from depthy.misc import load_img_file, Normalizer
from depthy.lightfield import local_structure_tensor


class StructTensorTestCase(unittest.TestCase):

    def setUp(self):

        # data preparation
        self.epi_img = load_img_file('../examples/data/epi_img.png')

        self.ref_img = load_img_file('../examples/data/ref_img.png')
        self.ref_img = self.ref_img.sum(-1)
        self.ref_img = Normalizer(self.ref_img).perc_clip()
        self.ref_img = Normalizer(self.ref_img).type_norm()

    def create_test_epis(self, m=49):

        test_epis = list()
        # vertical edge
        test_epis.append((90, np.array([np.concatenate([np.zeros(m//2+1), np.ones(m//2)]) for _ in np.arange(m)])))
        # edge with slope 65 degrees
        test_epis.append((65, np.array([np.concatenate([np.zeros(m-m//4-i//2), np.ones(i//2+m//4)]) for i in np.arange(m)])))
        # edge with slope 45 degrees
        test_epis.append((45, np.array([np.concatenate([np.zeros(m-1-i), np.ones(i+1)]) for i in np.arange(m)])))
        # horizontal edge
        test_epis.append((90, np.array([np.concatenate([np.zeros(m//2+1), np.ones(m//2)]) for _ in np.arange(m)]).T))

        return test_epis

    def test_disparity(self):

        m = 49
        test_epis = self.create_test_epis(m)

        for res, test_epi in test_epis:
            local_disp, coherence, n = local_structure_tensor(test_epi, slope_method='eigen')
            found_disp = local_disp[m//2, m//2]
            slope_center = 1/found_disp #if found_disp != 0 else np.inf
            angle_degree = np.arctan(slope_center) / np.pi * 180
            ret = np.allclose(res, abs(angle_degree), atol=1e-0)
            self.assertTrue(ret, msg='failed for %s degrees which gave %s degrees' % (str(res), str(angle_degree)))

    def test_structure_tensor(self):

        for (val_exp, m) in [(21701.76, 'sobel'), (20577.883, 'gradient'), (21370.906, 'scharr')]:

            self.local_disp, coherence, n = local_structure_tensor(self.epi_img.copy(), si=0.8, so=2.0,
                                                                   slope_method='eigen', grad_method=m)
            self.local_disp = self.local_disp[:, 1:-1]
            self.local_disp = self.local_disp.sum(-1)
            self.local_disp = Normalizer(self.local_disp).perc_clip(norm=True)

            self.assertTrue(int(self.local_disp_norm.sum()) <= int(val_exp))

        return True

    @property
    def local_disp_norm(self):
        return np.power(self.ref_img - self.local_disp, 2)

    def plot(self):

        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.title('Reference image')
        ax1.imshow(self.ref_img, cmap='gray')
        ax2.title('Result')
        ax2.imshow(self.local_disp, cmap='gray')
        ax3.title('Normalized ')
        ax3.imshow(self.local_disp_norm, cmap='gray')
        plt.show()

    def test_all(self):

        self.test_structure_tensor()
