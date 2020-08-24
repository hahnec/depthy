import numpy as np


def save_ply(pts: np.ndarray = None, file_path: str = './depth.ply') -> bool:
    """
    Creates an ASCII text file containing a point cloud which is in line with the Polygon File Format (PLY).
    See https://en.wikipedia.org/wiki/PLY_(file_format) for further information.

    :param pts: numpy array [Zx6] carrying x,y,z as well as R,G,B information
    :param file_path: file path string for file creation
    :return: True once saving succeeded
    """

    # remove invalid points
    valid_pts = pts[np.sum(np.isinf(pts) + np.isnan(pts), axis=1) == 0]

    pts_str_list = ["%010f %010f %010f %d %d %d\n" % tuple(pt) for pt in valid_pts]

    with open(file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % len(pts_str_list))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        f.writelines(pts_str_list)

    return True


def disp2pts(disp_img: np.ndarray = None,
             rgb_img: np.ndarray = None,
             focal_length_mm: float = 1,
             focus_dist_mm: float = 1,
             baseline_mm: float = 1,
             sensor_mm: float = 1) -> np.ndarray:
    """
    Convert disparity image to an array of points representing the point cloud.

    :param disp_img: disparity image [MxN]
    :param rgb_img: RGB image [MxNx3] which corresponds to disparity image, but is left optional
    :param focal_length_mm: focal length in mm
    :param focus_dist_mm: distance at which image is focused in mm
    :param baseline_mm: spacing between optical centers of cameras in mm
    :param sensor_mm: sensor size in mm
    :return: array of points [Zx6] where Z=M*N
    """

    h, w = np.shape(disp_img)
    max_res = max(w, h)
    b = baseline_mm * focal_length_mm * max_res

    # compute x,y,z coordinates
    pts = np.zeros((disp_img.size, 6))
    xx, yy = np.meshgrid(range(h), range(w))

    zz = (b * focus_dist_mm) / (disp_img.T * focus_dist_mm * sensor_mm + b)
    xx = (xx / (h - 1) - .5) * sensor_mm * zz / focal_length_mm
    yy = (yy / (w - 1) - .5) * sensor_mm * zz / focal_length_mm

    # add coordinates
    pts[:, 0] = xx.flatten()
    pts[:, 1] = -yy.flatten()
    pts[:, 2] = -zz.flatten()

    if rgb_img is not None:
        pts[:, 3] = rgb_img[..., 0].flatten()
        pts[:, 4] = rgb_img[..., 1].flatten()
        pts[:, 5] = rgb_img[..., 2].flatten()

    return pts
