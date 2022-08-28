# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Demo tests for raw template.
"""

import unittest

import numpy as np

from ibicus.utils import sort_array_like_another_one


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_sort_array_like_another_one(self):
        x = np.random.random(1000)
        y = np.random.random(1000)
        x_sorted_like_y = sort_array_like_another_one(x, y)

        assert all(np.argsort(x_sorted_like_y) == np.argsort(y))
