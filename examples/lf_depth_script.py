#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "info@christopherhahne.de"
__license__ = """
Copyright (c) 2020 Christopher Hahne <info@christopherhahne.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import matplotlib.pyplot as plt
import os

from depthy import FILE_EXTS
from depthy.lightfield.epi_depth import epi_depth
from depthy.misc import plot_point_cloud, Normalizer, load_lf_arr, primal_dual_algo
from depthy.misc.pfm_handler import save_pfm
from depthy.misc.ply_handler import save_ply, disp2pts

if __name__ == '__main__':

    # load data
    path = '/Users/Admin/Pictures/Plenoptic/INRIA_SIROCCO/Bee_2_colo/viewpoints_9px/'
    fnames = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(FILE_EXTS)]
    lf_img_arr = load_lf_arr(fnames)
    lf_img_arr = lf_img_arr.astype('float16') / lf_img_arr.max()
    lf_c = lf_img_arr.shape[0]//2

    # compute local depth
    cisparity = epi_depth(lf_img_arr.copy(), perc_clip=2, primal_opt=False)
    sisparity = epi_depth(lf_img_arr.copy(), perc_clip=2)

    values = [float(v)/1000 for v in list(range(5, 16, 1))]
    fig, axs = plt.subplots(1, len(values), figsize=(20, 5))
    for i in range(len(values)):
        res, _ = primal_dual_algo(cisparity, lambda_rof=0, theta=3, tau=values[i])
        axs[i].imshow(res, cmap='gray')
        axs[i].set_title(str(values[i]))
    plt.show()

    # plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    ax1.imshow(Normalizer(lf_img_arr[lf_c, lf_c, ...]).type_norm(), cmap='gray')
    ax2.imshow(Normalizer(cisparity).type_norm(), cmap='gray')
    ax3.imshow(Normalizer(sisparity).type_norm(), cmap='gray')
    plt.show()

    plot_point_cloud(disp_arr=sisparity, rgb_img=lf_img_arr[lf_c, lf_c, ...], down_scale=4)
    plt.show()

    # save results
    plt.imsave('depth.png', cisparity, cmap='gray')
    save_pfm(sisparity, file_path='./depth.pfm', scale=1)

    pts = disp2pts(sisparity, rgb_img=lf_img_arr[lf_c, lf_c, ...])
    save_ply(pts, file_path='depth_1.ply')
