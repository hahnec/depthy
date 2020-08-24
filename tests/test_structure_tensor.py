import unittest
import numpy as np

from depthy.misc import load_img_file, Normalizer
from depthy.lightfield import local_structure_tensor


class StructTensorTestCase(unittest.TestCase):

    def setUp(self):

        # data preparation
        self.epi_img = load_img_file('./examples/data/epi_img.png')

        self.ref_img = load_img_file('./examples/data/ref_img.png')
        self.ref_img = self.ref_img.sum(-1)
        self.ref_img = Normalizer(self.ref_img).perc_clip()
        self.ref_img = Normalizer(self.ref_img).type_norm()

    def test_structure_tensor(self):

        for (val_exp, m) in [(21701.76, 'sobel'), (20577.883, 'gradient'), (21370.906, 'scharr')]:

            self.local_slopes, coherence, n = local_structure_tensor(self.epi_img.copy(), si=0.8, so=2.0,
                                                                     slope_method='eigen', grad_method=m)
            self.local_slopes = self.local_slopes[:, 1:-1]
            self.local_slopes = self.local_slopes.sum(-1)
            self.local_slopes = Normalizer(self.local_slopes).perc_clip(norm=True)

            self.assertTrue(int(self.local_slopes_norm.sum()) <= int(val_exp))

        return True

    @property
    def local_slopes_norm(self):
        return np.power(self.ref_img - self.local_slopes, 2)

    def plot(self):

        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.title('Reference image')
        ax1.imshow(self.ref_img, cmap='gray')
        ax2.title('Result')
        ax2.imshow(self.local_slopes, cmap='gray')
        ax3.title('Normalized ')
        ax3.imshow(self.local_slopes_norm, cmap='gray')
        plt.show()

    def test_all(self):

        self.test_structure_tensor()
