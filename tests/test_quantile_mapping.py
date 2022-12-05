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
import scipy.stats

from ibicus.debias import QuantileMapping
from ibicus.utils import PrecipitationHurdleModelGamma


def check_different_maximally_up_to_1(x, y):
    return np.abs(x - y) <= 1


def check_equal_up_to_distance(x, y, distance):
    return all(np.abs(x - y) < distance)


def get_min_distance_in_array(x):
    x = np.sort(x)
    return np.min(np.abs(x[0 : (x.size - 1)] - x[1:]))


def gen_precip_data(p0, n, *gamma_args):
    nr_of_dry_days = scipy.stats.binom.rvs(n, p0)
    return np.concatenate(
        [
            np.repeat(0, nr_of_dry_days),
            scipy.stats.gamma.rvs(size=n - nr_of_dry_days, *gamma_args),
        ]
    )


class TestQuantileMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_from_variable(self):
        # Check default arguments
        hurs = QuantileMapping.from_variable("hurs")
        assert hurs.distribution == scipy.stats.beta
        assert hurs.detrending == "multiplicative"

        pr = QuantileMapping.from_variable("pr")
        assert pr.distribution == PrecipitationHurdleModelGamma
        assert pr.detrending == "multiplicative"

        psl = QuantileMapping.from_variable("psl")
        assert psl.distribution == scipy.stats.beta
        assert psl.detrending == "additive"

        rlds = QuantileMapping.from_variable("rlds")
        assert rlds.distribution == scipy.stats.beta
        assert rlds.detrending == "additive"

        with self.assertRaises(ValueError):
            QuantileMapping.from_variable("rsds")

        sfcWind = QuantileMapping.from_variable("sfcWind")
        assert sfcWind.distribution == scipy.stats.gamma
        assert sfcWind.detrending == "multiplicative"

        tas = QuantileMapping.from_variable("tas")
        assert tas.distribution == scipy.stats.norm
        assert tas.detrending == "additive"

        tasmin = QuantileMapping.from_variable("tasmin")
        assert tasmin.distribution == scipy.stats.beta
        assert tasmin.detrending == "additive"

        tasmax = QuantileMapping.from_variable("tasmax")
        assert tasmax.distribution == scipy.stats.beta
        assert tasmax.detrending == "additive"
