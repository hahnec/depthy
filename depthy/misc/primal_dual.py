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

# inspired by version authored by Louise Naud

import numpy as np
from scipy.misc import face
import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn('matplotlib import failed')

norm1 = lambda x: np.abs(x).sum(-1)

norm2 = lambda x: np.sqrt((x ** 2).sum(-1))

convergence = lambda vals, rtol=1e-5: abs(vals[-2]-vals[-1]) < rtol*abs(vals[0]-vals[1])


def dual_energy_tvl1(y: np.ndarray = None, obs_img: np.ndarray = None) -> float:
    """
    Compute the dual energy of TV-L1 problem.

    :param y: numpy array, [MxNx2]
    :param obs_img: numpy array, observed image
    :return: float, dual energy
    """
    nrg = -0.5 * (obs_img - backward_divergence(y)) ** 2
    nrg = float(np.sum(nrg))
    return nrg


def dual_energy_rof(y: np.ndarray = None, obs_img: np.ndarray = None) -> float:
    """
    Compute the dual energy of Rudin, Osher and Fatemi (ROF) problem.

    :param y: numpy array, [MxNx2]
    :param obs_img: numpy array [MxN], observed image
    :return: float, dual energy
    """
    nrg = -0.5 * (obs_img - backward_divergence(y)) ** 2
    nrg = float(nrg.sum())
    return nrg


def primal_energy_rof(u: np.ndarray = None, obs_img: np.ndarray = None, clambda: float = 1) -> float:
    """
    Compute the Rudin, Osher and Fatemi (ROF) energy.

    :param u: numpy array, [MxN]
    :param obs_img: numpy array [MxN], observed image
    :param clambda: float, lambda parameter
    :return: float, primal ROF energy
    """
    energy_reg = norm1(forward_gradient(u)).sum()
    energy_data_term = 0.5*clambda * norm2(u - obs_img).sum()
    return energy_reg + energy_data_term


def primal_energy_tvl1(u: np.ndarray = None, obs_img: np.ndarray = None, clambda: float = 1) -> float:
    """
    Compute the Total Variance (TV) L1 norm.

    :param u: numpy array, [MxN]
    :param obs_img: numpy array [MxN], observed image
    :param clambda: float, lambda parameter
    :return: float, primal ROF energy
    """
    energy_reg = norm1(forward_gradient(u)).sum()
    energy_data_term = clambda * np.abs(u - obs_img).sum()
    return energy_reg + energy_data_term


def forward_gradient(img: np.ndarray = None) -> np.ndarray:
    """
    Compute the forward gradient of the image according to definition on http://www.ipol.im/pub/art/2014/103/, p208.

    :param img: numpy array [MxN], input image
    :return: numpy array [MxNx2], gradient of the input image, the first channel is the horizontal gradient, the second
    is the vertical gradient.
    """
    # array allocation
    gradient = np.zeros((img.shape[:2]+(2,)), img.dtype)
    # Horizontal direction
    gradient[:, :-1, 0] = img[:, 1:] - img[:, :-1]
    # Vertical direction
    gradient[:-1, :, 1] = img[1:, :] - img[:-1, :]

    return gradient


def backward_divergence(grad: np.ndarray = None) -> np.ndarray:
    """
    Compute the backward divergence according to definition on http://www.ipol.im/pub/art/2014/103/, p208.

    :param grad: numpy array [NxMx2], array with the same dimensions as the gradient of the image to denoise.
    :return: numpy array [NxM], backward divergence
    """

    # Horizontal direction
    d_h = np.zeros(grad.shape[:2], grad.dtype)
    d_h[:, 0] = grad[:, 0, 0]
    d_h[:, 1:-1] = grad[:, 1:-1, 0] - grad[:, :-2, 0]
    d_h[:, -1] = -grad[:, -2:-1, 0].flatten()

    # Vertical direction
    d_v = np.zeros(grad.shape[:2], grad.dtype)
    d_v[0, :] = grad[0, :, 1]
    d_v[1:-1, :] = grad[1:-1, :, 1] - grad[:-2, :, 1]
    d_v[-1, :] = -grad[-2:-1, :, 1].flatten()

    # Divergence
    div = d_h + d_v
    return div


def prox_tv(y: np.ndarray = None, r: float = 1.0) -> np.ndarray:
    """
    Proximal operator for total variation (tv) which is equivalent to the integral of the gradient magnitude.

    :param y: numpy array [MxNx2],
    :param r: float, radius of infinity norm ball.
    :return: numpy array, same dimensions as y
    """

    n_y = np.maximum(1, norm2(y) / r)
    y /= n_y[..., np.newaxis]

    return y


def prox_l1(u: np.ndarray = None, f: np.ndarray = None, clambda: float = 1) -> np.ndarray:
    """
    :param u: numpy array, [MxN], primal variable,
    :param f: numpy array, [MxN], observed image,
    :param clambda: float, parameter for data term.
    :return: numpy array, [MxN]
    """
    return u + np.clip(f - u, -clambda, clambda)


def primal_dual_algo(data: np.ndarray,
                     lambda_rof: float = 1.0,
                     tau: float = 0.01,
                     theta: float = 1.0,
                     norm_l: float = 7.0,
                     max_iter: int = 100) -> [np.ndarray, tuple]:
    """
    Compute primal dual given observed data using the Rudin, Osher and Fatemi (ROF) method.

    :param data: numpy array, [MxN], observed data
    :param norm_l: float
    :param tau: float
    :param theta: float
    :param lambda_rof: float, penalty according to
    :param max_iter: int, maximum number of iterations.
    :return: numpy array, [MxN], denoised data
    """

    # parameter init
    sigma = 1.0 / (norm_l * tau)
    u = data.copy()
    y = forward_gradient(u)     # solve ROF

    primal, dual, gap = [[0] for _ in range(3)]

    for i in range(max_iter):

        # compute energies to track convergence
        primal.append(primal_energy_rof(u, data, sigma))
        dual.append(dual_energy_rof(y, data))
        gap.append(primal[-1] - dual[-1])

        # dual update
        y = y + sigma * forward_gradient(u)
        y = prox_tv(y, 1.0)    # projection

        # primal update
        u_new = (u + tau * backward_divergence(y) + lambda_rof * tau * data) / (1.0 + lambda_rof * tau)

        # smoothing
        u = u_new + theta * (u_new - u)

        if convergence(gap):
            break

    return u, (primal, dual, gap)


if __name__ == '__main__':

    # generate image data and add noise
    img_ref = np.array(face(True))
    img_ref = img_ref.astype('float') / np.max(img_ref)
    sig_add = .1
    img_obs = img_ref + sig_add * np.random.randn(*img_ref.shape)

    # run the optimization
    img_res, (primal, dual, gap) = primal_dual_algo(img_obs, lambda_rof=9.0, max_iter=100)

    # plot the energies
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    ax1.plot(range(len(primal)), primal, label="Primal Energy")
    ax1.set_title("Primal energy")
    ax1.legend()
    ax2.plot(range(len(dual)), dual, label="Dual Energy")
    ax2.set_title("Dual energy")
    ax2.legend()
    ax3.plot(range(len(gap)), gap, label="Gap")
    ax3.set_title("Gap")
    ax3.legend()
    plt.show()

    # Plot reference, observed and denoised image
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(20, 10))
    ax1.imshow(img_ref, cmap='gray')
    ax1.set_title("Ground-truth image")
    ax2.imshow(img_obs, cmap='gray')
    ax2.set_title("Observed image")
    ax3.imshow(img_res, cmap='gray')
    ax3.set_title("Denoised image")
    plt.show()
