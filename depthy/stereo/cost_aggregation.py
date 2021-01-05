import sys
import time as t

import numpy as np

from depthy.misc import Normalizer


def get_indices(offset: int = 0, dim: int = 0, direction: tuple = (0, 0), height: int = 0) -> (np.ndarray, np.ndarray):
    """
    Compute array of indices diagonal directions (SE, SW, NW, NE) of the slice.

    :param offset: difference with the main diagonal of the cost volume
    :param dim: number of elements along the path
    :param direction: current aggregation direction
    :param height: vertical dimension of the cost volume
    :return: tuple of two numpy arrays containing indices for the vertical and horizontal direction
    """

    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == (1, 1):
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(i)
            else:
                y_indices.append(i)
                x_indices.append(offset + i)

        if direction == (-1, 1):
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice: np.ndarray = None, offset: int = 0, p1: float = 10, p2: float = 120) -> np.ndarray:
    """
    Find the minimum costs in a D x M slice where M represents the number of pixels in the given direction.

    :param slice: M x D array from cost volume
    :param offset: ignore pixels at border
    :param p1: minor penalty cost
    :param p2: major penalty cost
    :return: M x D array of the minimum costs for a given slice in a given direction
    """
    y_dim, x_dim = slice.shape[0], slice.shape[1]

    disparities = np.repeat(np.arange(x_dim)[np.newaxis, ...], repeats=x_dim, axis=0)
    penalties = np.zeros(shape=(x_dim, x_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = p1
    penalties[np.abs(disparities - disparities.T) > 1] = p2

    minimum_cost_path = np.copy(slice)

    for i in range(offset, y_dim):
        previous_cost = minimum_cost_path[i-1, :]
        costs = np.repeat(previous_cost[..., np.newaxis], repeats=x_dim, axis=1)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] += costs - np.amin(previous_cost)

    return minimum_cost_path


def aggregate_costs(cost_volume: np.ndarray, p1: float = 10, p2: float = 120, path_num: int = 8) -> np.ndarray:
    """
    Compute aggregated matching costs for N possible directions (by default: N=8).

    :param cost_volume: array containing the matching costs
    :param p1: minor penalty cost
    :param p2: major penalty cost
    :param path_num: number of paths N (either 4 or 8)
    :return: H x W x D x N array of matching cost for N directions
    """

    assert path_num in [4, 8], 'Unsupported number of paths: %s' % path_num

    h, w = cost_volume.shape[0], cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start, end = -(h-1), w-1

    paths = ['vertical', 'horizontal', 'ascending', 'descending'] if path_num == 8 else ['vertical', 'horizontal']
    aggregation_tensor = np.zeros(shape=(h, w, disparities, 2*len(paths)), dtype=np.float)

    for i, path in enumerate(paths):
        print('\tProcess paths {}...'.format(path), end='')
        sys.stdout.flush()
        dawn = t.time()

        curr_aggregations = np.zeros(shape=(h, w, disparities, 2), dtype=cost_volume.dtype)

        if path == 'vertical':
            for x in range(w):
                south = cost_volume[:h, x, :]
                north = np.flip(south, axis=0)
                curr_aggregations[:, x, :, 0] = get_path_cost(south, 1, p1, p2)
                curr_aggregations[:, x, :, 1] = np.flip(get_path_cost(north, 1, p1, p2), axis=0)

        if path == 'horizontal':
            for y in range(h):
                east = cost_volume[y, :w, :]
                west = np.flip(east, axis=0)
                curr_aggregations[y, :, :, 0] = get_path_cost(east, 1, p1, p2)
                curr_aggregations[y, :, :, 1] = np.flip(get_path_cost(west, 1, p1, p2), axis=0)

        if path == 'descending':
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, (1, 1), None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                curr_aggregations[y_se_idx, x_se_idx, :, 0] = get_path_cost(south_east, 1, p1, p2)
                curr_aggregations[y_nw_idx, x_nw_idx, :, 1] = get_path_cost(north_west, 1, p1, p2)

        if path == 'ascending':
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, (-1, 1), h-1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                curr_aggregations[y_sw_idx, x_sw_idx, :, 0] = get_path_cost(south_west, 1, p1, p2)
                curr_aggregations[y_ne_idx, x_ne_idx, :, 1] = get_path_cost(north_east, 1, p1, p2)

        # append aggregations of current directions
        aggregation_tensor[..., i*2:i*2+2] = curr_aggregations

        dusk = t.time()
        print('\t(done in {:.2f}s)'.format(dusk - dawn))

    # integrate over all neighbours
    volume = np.sum(aggregation_tensor, axis=3)

    # normalize and convert float to uint16
    volume = Normalizer(volume).uint16_norm()

    return volume
