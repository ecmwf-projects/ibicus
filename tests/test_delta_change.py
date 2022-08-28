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

from ibicus.debias import DeltaChange


def check_different_maximally_up_to_1(x, y):
    return np.abs(x - y) <= 1


def check_equal_up_to_distance(x, y, distance):
    return all(np.abs(x - y) < distance)


def get_min_distance_in_array(x):
    x = np.sort(x)
    return np.min(np.abs(x[0 : (x.size - 1)] - x[1:]))


class TestCDFt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_from_variable(self):

        # Check default arguments
        hurs = DeltaChange.from_variable("hurs")
        assert hurs.delta_type == "multiplicative"

        pr = DeltaChange.from_variable("pr")
        assert pr.delta_type == "multiplicative"

        psl = DeltaChange.from_variable("psl")
        assert psl.delta_type == "additive"

        rlds = DeltaChange.from_variable("rlds")
        assert rlds.delta_type == "additive"

        rsds = DeltaChange.from_variable("rsds")
        assert rsds.delta_type == "multiplicative"

        sfcWind = DeltaChange.from_variable("sfcWind")
        assert sfcWind.delta_type == "multiplicative"

        tas = DeltaChange.from_variable("tas")
        assert tas.delta_type == "additive"

        tasmin = DeltaChange.from_variable("tasmin")
        assert tasmin.delta_type == "additive"

        tasmax = DeltaChange.from_variable("tasmax")
        assert tasmax.delta_type == "additive"
