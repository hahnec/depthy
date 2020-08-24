#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "info@christopherhahne.de"
__license__ = """
    Copyright (c) 2019 Christopher Hahne <info@christopherhahne.de>

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

import numpy as np


class Normalizer(object):

    def __init__(self, data: np.ndarray = None, min: float = None, max: float = None):

        self._data, self._min, self._max = None, None, None
        self._var_init(data, min, max)

    def _var_init(self, data: np.ndarray = None, min: float = None, max: float = None):

        self._data = self._data if data is None else np.asarray(data, dtype='float32')
        self._dtype = str(self._data.dtype) if isinstance(self._data, np.ndarray) else 'float32'

        self._min = self._min if min is None else min
        self._max = self._max if max is None else max
        self._min = self._data.min() if not any([self._min, min]) and isinstance(self._data, np.ndarray) else self._min
        self._max = self._data.max() if not any([self._max, max]) and isinstance(self._data, np.ndarray) else self._max

    def uint16_norm(self) -> np.ndarray:
        """ Normalize image array to 16-bit unsigned integer. """

        return np.asarray(np.round(self.norm_fun()*(2**16-1)), dtype=np.uint16)

    def uint8_norm(self) -> np.ndarray:
        """ Normalize image array to 8-bit unsigned integer. """

        return np.asarray(np.round(self.norm_fun()*(2**8-1)), dtype=np.uint8)

    def type_norm(self, data: np.ndarray = None,
                        min: float = None, max: float = None,
                        new_min: float = None, new_max: float = None) -> np.ndarray:
        """
        Normalize numpy image array for provided data type.

        :param data:
        :param min:
        :param max:
        :param new_min:
        :param new_max:
        :return:
        """

        self._var_init(data, min, max)

        if self._dtype.startswith(('int', 'uint')):
            new_max = np.iinfo(np.dtype(self._dtype)).max if new_max is None else new_max
            new_min = np.iinfo(np.dtype(self._dtype)).min if new_min is None else new_min
            img_norm = np.round(self.norm_fun() * (new_max - new_min) + new_min)
        else:
            new_max = 1.0 if new_max is None else new_max
            new_min = 0.0 if new_min is None else new_min
            img_norm = self.norm_fun() * (new_max - new_min) + new_min

        return np.asarray(img_norm, dtype=self._dtype)

    def norm_fun(self) -> np.ndarray:
        """ Normalize image to values between 1 and 0. """

        norm = (self._data - self._min) / (self._max - self._min) if self._max != (self._min and 0) else self._data

        # prevent wrap-around
        norm[norm < 0] = 0
        norm[norm > 1] = 1

        return norm

    def perc_clip(self, data: np.ndarray = None, perc_clip: float = 5, norm: bool = False) -> np.ndarray:
        """
        Clip input data at provided percentiles.

        :param data: input data values as numpy array
        :param perc_clip: percentile at which values are clipped
        :param norm: normalization option
        :return: clipped data array
        """

        new_max = np.percentile(self._data, 100-perc_clip)
        new_min = np.percentile(self._data, perc_clip)
        self._var_init(data, new_min, new_max)

        self._data[self._data > new_max] = new_max
        self._data[self._data < new_min] = new_min

        # 0 to 1 normalization if option set
        self._data = self.type_norm() if norm else self._data

        return self._data
