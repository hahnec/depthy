import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_point_cloud(disp_arr: np.ndarray,
                     rgb_img: np.ndarray = None,
                     scale: int = 1,
                     view_angles: (int, int) = (10, 135),
                     ax: Axes3D = None) -> plt.Axes:
    """
    Plots a point cloud using the well-known matplotlib.

    :param disp_arr: numpy array [MxN], representing the disparity map
    :param rgb_img: numpy array [MxNx3], containing RGB pixel colors
    :param scale: int, downscale factor
    :param view_angles: tuple(int, int) containing elevator and azimuth angle, respectively
    :param ax: Axes object, optional for accumulative plotting
    :return: Axes object, containing point cloud
    """

    fig, ax = plt.figure(), plt.axes(projection='3d') if ax is None else (None, ax)#.set(projection='3d')
    zz = disp_arr[::scale, ::scale, ...]
    xx, yy = np.meshgrid(np.arange(zz.shape[1]), np.arange(zz.shape[0]))
    rgb = rgb_img[::scale, ::scale, ...] if rgb_img is not None else np.ones(disp_arr.shape)[::scale, ::scale, ...]
    ax.scatter(xx, yy, zz, c=rgb.reshape(-1, rgb.shape[-1]) / rgb.max(), s=0.5)
    ax.view_init(view_angles[0], view_angles[1])

    return ax
