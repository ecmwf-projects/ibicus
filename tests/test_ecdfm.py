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

from ibicus.debias import ECDFM
from ibicus.utils import (
    PrecipitationHurdleModelGamma,
    gen_PrecipitationGammaLeftCensoredModel,
)


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


class TestECDFM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_from_variable(self):
        tas = ECDFM.from_variable("tas")
        assert tas.distribution == scipy.stats.beta

        pr = ECDFM.from_variable("pr")
        assert pr.distribution == PrecipitationHurdleModelGamma

        # Check default arguments
        hurs = ECDFM.from_variable("hurs")
        assert hurs.distribution == scipy.stats.beta

        pr = ECDFM.from_variable("pr")
        assert pr.distribution == PrecipitationHurdleModelGamma

        psl = ECDFM.from_variable("psl")
        assert psl.distribution == scipy.stats.beta

        rlds = ECDFM.from_variable("rlds")
        assert rlds.distribution == scipy.stats.beta

        with self.assertRaises(ValueError):
            ECDFM.from_variable("rsds")

        sfcWind = ECDFM.from_variable("sfcWind")
        assert sfcWind.distribution == scipy.stats.gamma

        tas = ECDFM.from_variable("tas")
        assert tas.distribution == scipy.stats.beta

        tasmin = ECDFM.from_variable("tasmin")
        assert tasmin.distribution == scipy.stats.beta

        tasmax = ECDFM.from_variable("tasmax")
        assert tasmax.distribution == scipy.stats.beta

    def test__init__(self):
        tas_1 = ECDFM.from_variable("tas")
        tas_2 = ECDFM(distribution=scipy.stats.beta)
        assert tas_1 == tas_2

        tas_1 = ECDFM.from_variable("tas", distribution=scipy.stats.norm)
        tas_2 = ECDFM(distribution=scipy.stats.norm)
        assert tas_1 == tas_2

        pr_1 = ECDFM.from_variable("pr")
        pr_2 = ECDFM(distribution=PrecipitationHurdleModelGamma)
        assert pr_1 == pr_2

    def test_for_precipitation(self):
        pr_1 = ECDFM.from_variable("pr")
        pr_2 = ECDFM.for_precipitation()

        assert pr_1 == pr_2

        pr_1 = ECDFM(distribution=gen_PrecipitationGammaLeftCensoredModel(0.2))
        pr_2 = ECDFM.for_precipitation(
            model_type="censored",
            censoring_threshold=0.2,
            hurdle_model_randomization=False,
        )

        assert pr_1 == pr_2

        pr_1 = ECDFM(distribution=gen_PrecipitationGammaLeftCensoredModel(0.1))
        pr_2 = ECDFM.for_precipitation(model_type="censored", censoring_threshold=0.2)

        assert pr_1 != pr_2

        pr_1 = ECDFM.for_precipitation(
            distribution=gen_PrecipitationGammaLeftCensoredModel(0.2)
        )
        pr_2 = ECDFM.for_precipitation(
            model_type="censored",
            censoring_threshold=0.2,
            hurdle_model_randomization=False,
        )

        assert pr_1 == pr_2

        pr_1 = ECDFM.for_precipitation(distribution=PrecipitationHurdleModelGamma)
        pr_2 = ECDFM.for_precipitation(
            model_type="hurdle",
            censoring_threshold=0.2,
            hurdle_model_randomization=True,
        )

        assert pr_1 == pr_2

        pr_1 = ECDFM.for_precipitation(distribution=PrecipitationHurdleModelGamma)
        pr_2 = ECDFM.for_precipitation(
            model_type="hurdle",
            censoring_threshold=0.2,
            hurdle_model_randomization=False,
        )

        assert pr_1 != pr_2

    def test_apply_location_tas(self):
        tas = ECDFM.from_variable("tas")

        # Test: perfect match between obs and cm_hist
        obs = np.random.beta(5, 2, size=1000)
        cm_hist = obs
        cm_future = np.random.beta(7, 3, size=1000)

        assert np.allclose(tas.apply_location(obs, cm_hist, cm_future), cm_future)

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = np.random.beta(5, 2, size=1000)
        cm_hist = obs
        cm_future = np.random.uniform(low=0, high=1, size=1000)

        assert np.allclose(tas.apply_location(obs, cm_hist, cm_future), cm_future)

        # Test: corrects mean difference
        obs = np.random.beta(5, 2, size=1000)
        cm_hist = np.random.beta(5, 2, size=1000) + 5
        cm_future = np.random.beta(7, 3, size=1000) + 5

        assert (
            np.abs(np.mean(tas.apply_location(obs, cm_hist, cm_future)) - np.mean(obs))
            < 0.1
        )

        # Test: perfect match between obs and cm_hist up to translation and cm_fut from new distribution
        obs = np.random.beta(5, 2, size=1000)
        cm_hist = obs + 5
        cm_future = np.random.beta(7, 4, size=1000) + 7
        assert np.allclose(
            tas.apply_location(obs, cm_hist, cm_future), cm_future - 5, atol=1e-5
        )

    def test_apply_location_pr(self):
        # Compare all values
        pr = ECDFM.for_precipitation(hurdle_model_randomization=False)

        # Test: perfect match between obs and cm_hist
        obs = gen_precip_data(0.4, 1000, 5, 2)
        cm_hist = obs
        cm_future = gen_precip_data(0.4, 1000, 5, 2)

        assert np.allclose(pr.apply_location(obs, cm_hist, cm_future), cm_future)

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = gen_precip_data(0.4, 1000, 5, 2)
        cm_hist = obs
        cm_future = gen_precip_data(0.6, 2000, 13, 9)

        assert np.allclose(pr.apply_location(obs, cm_hist, cm_future), cm_future)

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = gen_precip_data(0.4, 1000, 5, 2)
        cm_hist = obs
        cm_future = gen_precip_data(0.2, 2000, 13, 9)

        assert np.allclose(pr.apply_location(obs, cm_hist, cm_future), cm_future)

        # Compare non-zero values
        pr = ECDFM.for_precipitation(hurdle_model_randomization=True)

        # Test: perfect match between obs and cm_hist
        obs = gen_precip_data(0.4, 1000, 5, 2)
        cm_hist = obs
        cm_future = gen_precip_data(0.4, 1000, 5, 2)
        cm_future_non_zero = cm_future > 0

        assert np.allclose(
            pr.apply_location(obs, cm_hist, cm_future)[cm_future_non_zero],
            cm_future[cm_future_non_zero],
        )

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = gen_precip_data(0.4, 1000, 5, 2)
        cm_hist = obs
        cm_future = gen_precip_data(0.6, 2000, 13, 9)
        cm_future_non_zero = cm_future > 0

        assert np.allclose(
            pr.apply_location(obs, cm_hist, cm_future)[cm_future_non_zero],
            cm_future[cm_future_non_zero],
        )

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = gen_precip_data(0.4, 1000, 5, 2)
        cm_hist = obs
        cm_future = gen_precip_data(0.2, 2000, 13, 9)
        cm_future_non_zero = cm_future > 0

        assert np.allclose(
            pr.apply_location(obs, cm_hist, cm_future)[cm_future_non_zero],
            cm_future[cm_future_non_zero],
        )

        # Test: corrects mean difference
        pr = ECDFM.for_precipitation(hurdle_model_randomization=False)

        obs = gen_precip_data(0.4, 1000, 5, 2)
        cm_hist = gen_precip_data(0.4, 1000, 5, 2) + 5
        cm_future = gen_precip_data(0.4, 1000, 5, 2) + 5

        assert (
            np.abs(np.mean(pr.apply_location(obs, cm_hist, cm_future)) - np.mean(obs))
            < 1
        )
