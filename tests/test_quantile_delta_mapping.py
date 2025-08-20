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

from ibicus.debias import QuantileDeltaMapping
from ibicus.utils import gen_PrecipitationGammaLeftCensoredModel


def check_different_maximally_up_to_1(x, y):
    return np.abs(x - y) <= 1


def check_equal_up_to_distance(x, y, distance):
    return all(np.abs(x - y) < distance)


def get_min_distance_in_array(x):
    x = np.sort(x)
    return np.min(np.abs(x[0 : (x.size - 1)] - x[1:]))


def nested_tuples_similar_up_to_diff(tuple1, tuple2, diff):
    if len(tuple1) != len(tuple2):
        return False

    for elem1, elem2 in zip(tuple1, tuple2):
        if isinstance(elem1, tuple):
            if isinstance(elem2, tuple):
                return nested_tuples_similar_up_to_diff(elem1, elem2, diff)
            else:
                return False
        if not np.abs(elem1 - elem2) < diff:
            return False
    return True


class TestQuantileDeltaMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_from_variable(self):
        tas = QuantileDeltaMapping.from_variable("tas")
        assert tas.censor_values_to_zero is False
        assert tas.distribution is None
        assert tas.mapping_type == "nonparametric"
        assert tas.trend_preservation == "absolute"

        pr1 = QuantileDeltaMapping.from_variable("pr")
        assert pr1.censor_values_to_zero is True
        assert pr1.censoring_threshold == 0.05 / 86400
        assert pr1.trend_preservation == "relative"

        pr2 = QuantileDeltaMapping.from_variable("pr", censoring_threshold=0.1)
        assert pr2.censor_values_to_zero

        pr3 = QuantileDeltaMapping.for_precipitation(censoring_threshold=0.1)
        assert pr2 == pr3

        pr4 = QuantileDeltaMapping.for_precipitation(distribution=scipy.stats.norm)
        assert pr4.distribution is None
        assert pr4.mapping_type == "nonparametric"
        assert pr4.censoring_threshold == 0.05 / 86400
        assert pr4.censor_values_to_zero

        pr5 = QuantileDeltaMapping.for_precipitation(
            mapping_type="parametric", distribution=scipy.stats.norm
        )
        assert pr5.distribution == scipy.stats.norm
        assert pr5.censoring_threshold == 0.05 / 86400
        assert pr5.censor_values_to_zero

        pr6 = QuantileDeltaMapping.for_precipitation(
            mapping_type="parametric", censoring_threshold=0.1
        )
        assert isinstance(pr6.distribution, gen_PrecipitationGammaLeftCensoredModel)
        assert pr6.distribution.censoring_threshold == pr3.censoring_threshold

        # Check default arguments
        hurs = QuantileDeltaMapping.from_variable("hurs")
        assert hurs.distribution is None
        assert hurs.mapping_type == "nonparametric"
        assert hurs.trend_preservation == "relative"

        pr = QuantileDeltaMapping.from_variable("pr")
        assert pr.distribution is None
        assert pr.trend_preservation == "relative"
        assert pr.mapping_type == "nonparametric"
        assert pr.censor_values_to_zero is True

        psl = QuantileDeltaMapping.from_variable("psl")
        assert psl.distribution is None
        assert psl.mapping_type == "nonparametric"
        assert psl.trend_preservation == "absolute"

        rlds = QuantileDeltaMapping.from_variable("rlds")
        assert rlds.distribution is None
        assert rlds.mapping_type == "nonparametric"
        assert rlds.trend_preservation == "absolute"

        with self.assertRaises(ValueError):
            QuantileDeltaMapping.from_variable("rsds")

        sfcWind = QuantileDeltaMapping.from_variable("sfcWind")
        assert sfcWind.distribution is None
        assert sfcWind.mapping_type == "nonparametric"
        assert sfcWind.trend_preservation == "relative"

        tas = QuantileDeltaMapping.from_variable("tas")
        assert tas.distribution is None
        assert tas.mapping_type == "nonparametric"
        assert tas.trend_preservation == "absolute"

        tasmin = QuantileDeltaMapping.from_variable("tasmin")
        assert tasmin.distribution is None
        assert tasmin.mapping_type == "nonparametric"
        assert tasmin.trend_preservation == "absolute"

        tasmax = QuantileDeltaMapping.from_variable("tasmax")
        assert tasmax.distribution is None
        assert tasmax.mapping_type == "nonparametric"
        assert tasmax.trend_preservation == "absolute"

    def test__init__(self):
        tas_1 = QuantileDeltaMapping.from_variable("tas")
        tas_2 = QuantileDeltaMapping(distribution=None, trend_preservation="absolute")
        assert tas_1 == tas_2

        pr_1 = QuantileDeltaMapping.from_variable("pr")
        assert pr_1.censor_values_to_zero

    def test__get_obs_and_cm_hist_fits(self):
        tas = QuantileDeltaMapping.from_variable(
            "tas", distribution=scipy.stats.norm, mapping_type="parametric"
        )

        obs = scipy.stats.norm.rvs(loc=2, scale=4, size=1000)
        cm_hist = scipy.stats.norm.rvs(loc=8, scale=2, size=1000)
        assert nested_tuples_similar_up_to_diff(
            tas._get_obs_and_cm_hist_fits(obs, cm_hist), ((2, 4), (8, 2)), 0.1
        )

        pr = QuantileDeltaMapping.for_precipitation(mapping_type="parametric")

        model = gen_PrecipitationGammaLeftCensoredModel(pr.censoring_threshold)
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = model.ppf(np.random.random(size=1000), *(5, 0, 7))

        assert nested_tuples_similar_up_to_diff(
            pr._get_obs_and_cm_hist_fits(obs, cm_hist),
            (model.fit(obs), model.fit(cm_hist)),
            0.1,
        )

    def test_tas_apply_on_seasonal_and_future_window(self):

        # Systematic tests of tas
        debiaser = QuantileDeltaMapping.from_variable(
            "tas",
            running_window_mode=False,
            running_window_mode_over_years_of_cm_future=False,
        )

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
                                # assert np.abs(np.std(debiased_cm_fut)/trend_scale - scale_obs) < 0.5

    def test_pr_apply_on_seasonal_and_future_window(self):

        # Systematic tests of tas
        debiaser = QuantileDeltaMapping.from_variable(
            "pr",
            running_window_mode=False,
            running_window_mode_over_years_of_cm_future=False,
        )

        n = 10000
        np.random.seed(1234)
        for scale_bias in [1.0, 1.5, 2.0]:
            for trend_scale in [1.0, 2.0]:

                obs = np.exp(np.random.normal(size=n))
                cm_hist = np.exp(np.random.normal(size=n)) * scale_bias
                cm_fut = np.exp(np.random.normal(size=n)) * scale_bias * trend_scale

                debiased_cm_fut = debiaser.apply_location(obs, cm_hist, cm_fut)
                assert (
                    np.abs(np.mean(debiased_cm_fut) / trend_scale - np.mean(obs)) < 0.1
                )

    def test_apply_on_seasonal_and_future_window_tas(self):
        tas = QuantileDeltaMapping.from_variable("tas")

        # Test same vectors:
        obs = scipy.stats.norm.rvs(size=1000)
        cm_hist = obs
        cm_future = obs

        # distance = get_min_distance_in_array(obs)

        assert np.allclose(
            tas.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future),
            cm_future,
        )

        # Test: perfect match between obs and cm_hist
        obs = np.random.random(size=1000)
        cm_hist = obs
        cm_future = np.random.random(size=1000)

        assert np.allclose(
            tas.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future),
            cm_future,
        )

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = np.random.random(size=1000)
        cm_hist = obs
        cm_future = np.random.normal(size=1000)

        assert np.allclose(
            tas.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future),
            cm_future,
        )

        # Test: corrects mean difference
        obs = np.random.normal(size=1000)
        cm_hist = np.random.normal(size=1000) + 5
        cm_future = np.random.normal(size=1000) + 5

        assert (
            np.abs(
                np.mean(
                    tas.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future)
                )
                - np.mean(obs)
            )
            < 0.1
        )

        # Test: perfect match between obs and cm_hist up to translation and cm_fut from new distribution
        obs = np.random.random(size=1000)
        cm_hist = obs + 5
        cm_future = np.random.normal(size=1000)

        assert np.allclose(
            tas.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future),
            cm_future - 5,
        )

    def test_apply_on_seasonal_and_future_window_pr(self):
        pr = QuantileDeltaMapping.from_variable("pr")
        model = gen_PrecipitationGammaLeftCensoredModel(pr.censoring_threshold)

        # Test same vectors:
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = obs
        cm_future = obs

        # distance = get_min_distance_in_array(obs)

        assert np.allclose(
            pr.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future),
            cm_future,
        )

        # Test: perfect match between obs and cm_hist
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = obs
        cm_future = model.ppf(np.random.random(size=1000), *(2, 0, 3))

        assert np.allclose(
            pr.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future),
            cm_future,
        )

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = obs
        cm_future = model.ppf(np.random.random(size=1000), *(5, 0, 2))

        assert np.allclose(
            pr.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future),
            cm_future,
        )

        # Test: corrects mean relative difference
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = model.ppf(np.random.random(size=1000), *(2, 0, 3)) * 5
        cm_future = model.ppf(np.random.random(size=1000), *(2, 0, 3)) * 5

        assert (
            np.abs(
                np.mean(pr.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future))
                / np.mean(obs)
                - 1
            )
            < 0.1
        )

        # Test: perfect match between obs and cm_hist up to mutiplication and cm_fut from new distribution
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = obs * 5
        cm_future = model.ppf(np.random.random(size=1000), *(2, 0, 5)) * 5

        assert all(
            np.abs(
                pr.apply_on_seasonal_and_future_window(obs, cm_hist, cm_future)
                - cm_future / 5
            )
            < 1e-5
        )
