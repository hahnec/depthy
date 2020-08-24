import numpy as np
from scipy.ndimage import gaussian_filter, convolve


def local_structure_tensor(img: np.ndarray,
                           si: float = 0.8,
                           so: float = 1.6,
                           slope_method: str = 'eigen',
                           grad_method: str = None,
                           f: float = 1) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    This function computes the local slopes of a given input image (e.g. epipolar image) using a structure tensor.

    :param img: image input (e.g. epipolar image)
    :param si: sigma for inner Gaussian kernel
    :param so: sigma for outer Gaussian kernel
    :param slope_method: 'eigen' for eigendecomposition
    :param grad_method: partial derivative method with 'scharr' as default and 'sobel' or 'gradient' as alternatives
    :param f: focal length scaling local slope values
    :return: local_slopes, coherence, n
    """

    img = img if len(img.shape) == 3 else img[..., np.newaxis]
    chs = img.shape[-1] if len(img.shape) == 3 else (1,)
    grad_method = 'scharr' if grad_method is None else grad_method

    jyy, jxx, jxy = np.zeros((3,) + img.shape)
    for ch in range(chs):
        # gaussian filter for smoothness/de-noising
        img[..., ch] = gaussian_filter(img[..., ch], si)

        # compute image gradients
        grad_y, grad_x = partial_img_gradients(img[..., ch], method=grad_method)

        # compute structure tensor (using gradient maps)
        jyy[..., ch] = gaussian_filter(grad_y**2, so)
        jxx[..., ch] = gaussian_filter(grad_x**2, so)
        jxy[..., ch] = gaussian_filter(grad_x * grad_y, so)

    # local gradients of structure tensor
    if slope_method == 'eigen':
        num = -.5 * (jxx - jyy - np.sqrt((jxx-jyy)**2 + 4*jxy**2))
        denom = jxy
    else:
        raise Exception('Local slope method %s not recognized' % slope_method)
    local_slopes = f*np.divide(num, denom, out=np.zeros_like(denom), where=denom != 0)

    # slope direction as vector n
    n = np.array([(jyy-jxx), (2*jxy)])

    # coherence as reliability measure
    coherence = np.sqrt(np.divide((jyy-jxx)**2+4*jxy**2, (jxx+jyy)**2, out=np.zeros_like(jxx), where=jxx+jyy != 0))

    return local_slopes, coherence, n


def partial_img_gradients(img: np.ndarray, method: str = 'gradient') -> [np.ndarray, np.ndarray]:
    """
    Compute partial derivatives of a 2-dimensional image.

    :param img: input image
    :param method: method for first-order partial derivative featuring 'scharr', 'sobel' and 'gradient'.
    :return: vertical partial gradient, horizontal partial gradient
    """

    if method == 'scharr':
        kernel = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
        grad_y = convolve(img, kernel)
        grad_x = convolve(img, kernel.T)
    elif method == 'sobel':
        kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        grad_y = convolve(img, kernel)
        grad_x = convolve(img, kernel.T)
    elif method == 'gradient':
        grad_y = np.gradient(img, axis=0)
        grad_x = np.gradient(img, axis=1)
    else:
        raise Exception('Gradient method %s not supported' % method)

    return grad_y, grad_x
