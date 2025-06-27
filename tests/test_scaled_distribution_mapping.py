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

from ibicus.debias import ScaledDistributionMapping


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


class TestScaledDistributionMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_from_variable(self):
        # Check default arguments
        with self.assertRaises(ValueError):
            ScaledDistributionMapping.from_variable("hurs")

        pr = ScaledDistributionMapping.from_variable("pr")
        assert pr.distribution == scipy.stats.gamma
        assert pr.mapping_type == "relative"

        with self.assertRaises(ValueError):
            ScaledDistributionMapping.from_variable("psl")

        with self.assertRaises(ValueError):
            ScaledDistributionMapping.from_variable("rlds")

        with self.assertRaises(ValueError):
            ScaledDistributionMapping.from_variable("rsds")

        with self.assertRaises(ValueError):
            ScaledDistributionMapping.from_variable("sfcWind")

        tas = ScaledDistributionMapping.from_variable("tas")
        assert tas.distribution == scipy.stats.norm
        assert tas.mapping_type == "absolute"

        tasmin = ScaledDistributionMapping.from_variable("tasmin")
        assert tasmin.distribution == scipy.stats.norm
        assert tasmin.mapping_type == "absolute"

        tasmax = ScaledDistributionMapping.from_variable("tasmax")
        assert tasmax.distribution == scipy.stats.norm
        assert tasmax.mapping_type == "absolute"

    def test_absolute_sdm_apply_location(self):
        # Systematic tests of absolute SDM
        debiaser = ScaledDistributionMapping.from_variable("tas")

        n = 10000
        np.random.seed(1234)
        for mean_obs in [0, 5]:
            for scale_obs in [1.0, 2.0]:
                for bias in [0, 1, 2, 10]:
                    for scale_bias in [1.0, 1.5, 2.0]:
                        for trend in [0.0, 10.0, 20.0]:
                            for trend_scale in [1.0, 2.0]:

                                obs = np.random.normal(size=n) * scale_obs + mean_obs
                                cm_hist = (
                                    np.random.normal(size=n) * scale_obs * scale_bias
                                    + mean_obs
                                    + bias
                                )
                                cm_fut = (
                                    np.random.normal(size=n)
                                    * scale_obs
                                    * scale_bias
                                    * trend_scale
                                    + mean_obs
                                    + bias
                                    + trend
                                )

                                debiased_cm_fut = debiaser.apply_location(
                                    obs, cm_hist, cm_fut
                                )
                                assert (
                                    np.abs(np.mean(debiased_cm_fut) - trend - mean_obs)
                                    < 0.5
                                )
                                assert (
                                    np.abs(
                                        np.std(debiased_cm_fut) / trend_scale
                                        - scale_obs
                                    )
                                    < 0.1
                                )
