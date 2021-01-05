import numpy as np

# sum of absolute differences
sad = lambda window, kernel: np.sum(np.abs(np.subtract(window, kernel)))
sad_vectorized = lambda window, kernel, axis=1: np.sum(np.abs(np.subtract(window, kernel)), axis=axis)
abs_diff = lambda window, kernel: np.abs(np.subtract(window, kernel))


def xor_similarity(census_values, census):
    """
    Dissimilarity measure based on XOR operation from previously computed census values.
    :param census_values:
    :param census:
    :return: H x W x D array as numpy uint32 type.
    """

    xor_img = np.int64(np.bitwise_xor(np.int64(census_values), census))
    cost = np.zeros(shape=census.shape, dtype=np.uint32)
    while not np.all(xor_img == 0):
        tmp = xor_img - 1
        mask = xor_img != 0
        xor_img[mask] = np.bitwise_and(xor_img[mask], tmp[mask])
        cost[mask] = cost[mask] + 1

    return cost


def birchfield_tomasi():
    pass
