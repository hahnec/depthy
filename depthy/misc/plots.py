import numpy as np
import warnings
from depthy.misc import Normalizer

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D
except ImportError:
    warnings.warn('matplotlib import failed')
    Axes3D = None


def plot_point_cloud(disp_arr: np.ndarray,
                     rgb_img: np.ndarray = None,
                     down_scale: int = 1,
                     view_angles: (int, int) = (10, 135),
                     s: float = 0.5,
                     ax: Axes3D = None,
                     show_axes: bool = False) -> Axes3D:
    """
    Plots a point cloud using the famous matplotlib.

    :param disp_arr: numpy array [MxN], representing the disparity map
    :param rgb_img: numpy array [MxNx3], containing RGB pixel colors
    :param down_scale: int, downscale factor
    :param view_angles: tuple(int, int) containing elevator and azimuth angle, respectively
    :param s: float, size of a point
    :param ax: Axes3D object, optional for accumulative plotting
    :param show_axes: bool, option for axes plot
    :return: Axes3D object, containing point cloud
    """

    # validate downscale value
    if down_scale < 1 or down_scale > min(disp_arr.shape[:2])//2:
        raise IndexError('Downscale factor is %s and out-of-range.' % down_scale)

    # rgb image presence/absence handling
    if rgb_img is None or disp_arr.shape[:2] != rgb_img.shape[:2] or len(rgb_img.shape) != 3:
        rgb = np.zeros(disp_arr.shape+(3,))[::down_scale, ::down_scale, ...]
        if rgb_img is not None:
            warnings.warn('Depth map and RGB image dimension mismatch.')
    else:
        # flip x-axis and downscale rgb image
        rgb = rgb_img[:, ::-1, ...][::down_scale, ::down_scale, ...]
        # normalize rgb values to 0-1 range
        rgb = Normalizer(rgb).type_norm(new_min=max(0, rgb.min()), new_max=1)

    # flip x-axis and downscale depth map
    zz = disp_arr[:, ::-1][::down_scale, ::down_scale, ...]
    xx, yy = np.meshgrid(np.arange(zz.shape[1]), np.arange(zz.shape[0]))

    # sort according to depth value to avoid occlusion order problem
    order = np.argsort(zz.ravel())

    # plot depth data
    fig, ax = (plt.figure(), plt.axes(projection='3d')) if ax is None else (None, ax)
    ax.set_axis_on() if show_axes else ax.set_axis_off()
    ax.scatter(xx.ravel()[order], yy.ravel()[order], zz.ravel()[order], c=rgb.reshape(-1, rgb.shape[2])[order], s=s)
    ax.view_init(view_angles[0], view_angles[1])
    ax.set_ylim(0, zz.shape[0])
    ax.set_xlim(0, zz.shape[1])

    return ax
