import numpy as np
import warnings

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
                     ax: Axes3D = None) -> Axes3D:
    """
    Plots a point cloud using the well-known matplotlib.

    :param disp_arr: numpy array [MxN], representing the disparity map
    :param rgb_img: numpy array [MxNx3], containing RGB pixel colors
    :param down_scale: int, downscale factor
    :param view_angles: tuple(int, int) containing elevator and azimuth angle, respectively
    :param ax: Axes3D object, optional for accumulative plotting
    :return: Axes3D object, containing point cloud
    """

    # validate downscale value
    if down_scale < 1 or down_scale > min(disp_arr.shape[:2])//2:
        raise IndexError('Downscale factor is %s and out-of-range.' % down_scale)

    # rgb image presence/absence handling
    if rgb_img is None or disp_arr.shape[:2] != rgb_img.shape[:2]:
        rgb = np.ones(disp_arr.shape)[::down_scale, ::down_scale, ...]
        if rgb_img is not None:
            warnings.warn('Depth map and RGB image dimension mismatch.')
    else:
        # flip x-axis and downscale rgb image
        rgb = rgb_img[:, ::-1, ...][::down_scale, ::down_scale, ...]

    # flip x-axis and downscale depth map
    zz = disp_arr[:, ::-1][::down_scale, ::down_scale, ...]
    xx, yy = np.meshgrid(np.arange(zz.shape[1]), np.arange(zz.shape[0]))

    # plot depth data
    fig, ax = plt.figure(), plt.axes(projection='3d') if ax is None else (None, ax)
    ax.scatter(xx, yy, zz, c=rgb.reshape(-1, rgb.shape[-1]) / rgb.max(), s=0.5)
    ax.view_init(view_angles[0], view_angles[1])

    return ax
