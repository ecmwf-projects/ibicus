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
        assert tas.distribution == scipy.stats.norm
        assert tas.trend_preservation == "absolute"

        pr1 = QuantileDeltaMapping.from_variable("pr")
        assert pr1.censor_values_to_zero is True
        assert pr1.censoring_threshold == pr1.distribution.censoring_threshold
        assert pr1.trend_preservation == "relative"

        pr2 = QuantileDeltaMapping.from_variable("pr", censoring_threshold=0.1)
        assert pr2.distribution.censoring_threshold == pr2.censoring_threshold
        assert pr2.censor_values_to_zero

        pr3 = QuantileDeltaMapping.for_precipitation(0.1)
        assert pr2 == pr3

        pr4 = QuantileDeltaMapping.for_precipitation(distribution=scipy.stats.norm)
        assert pr4.distribution == scipy.stats.norm
        assert pr4.censoring_threshold == 0.05 / 86400
        assert pr4.censor_values_to_zero

        # Check default arguments
        hurs = QuantileDeltaMapping.from_variable("hurs")
        assert hurs.distribution == scipy.stats.beta
        assert hurs.trend_preservation == "relative"

        pr = QuantileDeltaMapping.from_variable("pr")
        assert pr.distribution == gen_PrecipitationGammaLeftCensoredModel(
            censoring_threshold=0.05 / 86400, censor_in_ppf=False
        )
        assert pr.trend_preservation == "relative"
        assert pr.censor_values_to_zero is True

        psl = QuantileDeltaMapping.from_variable("psl")
        assert psl.distribution == scipy.stats.beta
        assert psl.trend_preservation == "absolute"

        rlds = QuantileDeltaMapping.from_variable("rlds")
        assert rlds.distribution == scipy.stats.beta
        assert rlds.trend_preservation == "absolute"

        with self.assertRaises(ValueError):
            QuantileDeltaMapping.from_variable("rsds")

        sfcWind = QuantileDeltaMapping.from_variable("sfcWind")
        assert sfcWind.distribution == scipy.stats.gamma
        assert sfcWind.trend_preservation == "relative"

        tas = QuantileDeltaMapping.from_variable("tas")
        assert tas.distribution == scipy.stats.norm
        assert tas.trend_preservation == "absolute"

        tasmin = QuantileDeltaMapping.from_variable("tasmin")
        assert tasmin.distribution == scipy.stats.beta
        assert tasmin.trend_preservation == "absolute"

        tasmax = QuantileDeltaMapping.from_variable("tasmax")
        assert tasmax.distribution == scipy.stats.beta
        assert tasmax.trend_preservation == "absolute"

    def test__init__(self):
        tas_1 = QuantileDeltaMapping.from_variable("tas")
        tas_2 = QuantileDeltaMapping(
            distribution=scipy.stats.norm, trend_preservation="absolute"
        )
        assert tas_1 == tas_2

        pr_1 = QuantileDeltaMapping.from_variable("pr")
        assert pr_1.censor_values_to_zero

    def test__get_obs_and_cm_hist_fits(self):
        tas = QuantileDeltaMapping.from_variable("tas")

        obs = scipy.stats.norm.rvs(loc=2, scale=4, size=1000)
        cm_hist = scipy.stats.norm.rvs(loc=8, scale=2, size=1000)
        assert nested_tuples_similar_up_to_diff(
            tas._get_obs_and_cm_hist_fits(obs, cm_hist), ((2, 4), (8, 2)), 0.1
        )

        pr = QuantileDeltaMapping.from_variable("pr")

        model = gen_PrecipitationGammaLeftCensoredModel(pr.censoring_threshold)
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = model.ppf(np.random.random(size=1000), *(5, 0, 7))

        assert nested_tuples_similar_up_to_diff(
            pr._get_obs_and_cm_hist_fits(obs, cm_hist),
            (model.fit(obs), model.fit(cm_hist)),
            0.1,
        )

    def test__apply_debiasing_steps_tas(self):
        tas = QuantileDeltaMapping.from_variable("tas")

        # Test same vectors:
        obs = scipy.stats.norm.rvs(size=1000)
        cm_hist = obs
        cm_future = obs

        # distance = get_min_distance_in_array(obs)

        fit_obs, fit_cm_hist = tas._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert np.allclose(
            tas._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist), cm_future
        )

        # Test: perfect match between obs and cm_hist
        obs = np.random.random(size=1000)
        cm_hist = obs
        cm_future = np.random.random(size=1000)

        fit_obs, fit_cm_hist = tas._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert np.allclose(
            tas._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist), cm_future
        )

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = np.random.random(size=1000)
        cm_hist = obs
        cm_future = np.random.normal(size=1000)

        fit_obs, fit_cm_hist = tas._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert np.allclose(
            tas._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist), cm_future
        )

        # Test: corrects mean difference
        obs = np.random.normal(size=1000)
        cm_hist = np.random.normal(size=1000) + 5
        cm_future = np.random.normal(size=1000) + 5

        fit_obs, fit_cm_hist = tas._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert (
            np.abs(
                np.mean(tas._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist))
                - np.mean(obs)
            )
            < 0.1
        )

        # Test: perfect match between obs and cm_hist up to translation and cm_fut from new distribution
        obs = np.random.random(size=1000)
        cm_hist = obs + 5
        cm_future = np.random.normal(size=1000)

        fit_obs, fit_cm_hist = tas._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert np.allclose(
            tas._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist), cm_future - 5
        )

    def test__apply_debiasing_steps_pr(self):
        pr = QuantileDeltaMapping.from_variable("pr")
        model = gen_PrecipitationGammaLeftCensoredModel(pr.censoring_threshold)

        # Test same vectors:
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = obs
        cm_future = obs

        # distance = get_min_distance_in_array(obs)

        fit_obs, fit_cm_hist = pr._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert np.allclose(
            pr._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist), cm_future
        )

        # Test: perfect match between obs and cm_hist
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = obs
        cm_future = model.ppf(np.random.random(size=1000), *(2, 0, 3))

        fit_obs, fit_cm_hist = pr._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert np.allclose(
            pr._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist), cm_future
        )

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = obs
        cm_future = model.ppf(np.random.random(size=1000), *(5, 0, 2))

        fit_obs, fit_cm_hist = pr._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert np.allclose(
            pr._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist), cm_future
        )

        # Test: corrects mean relative difference
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = model.ppf(np.random.random(size=1000), *(2, 0, 3)) * 5
        cm_future = model.ppf(np.random.random(size=1000), *(2, 0, 3)) * 5

        fit_obs, fit_cm_hist = pr._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert (
            np.abs(
                np.mean(pr._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist))
                / np.mean(obs)
                - 1
            )
            < 0.1
        )

        # Test: perfect match between obs and cm_hist up to mutiplication and cm_fut from new distribution
        obs = model.ppf(np.random.random(size=1000), *(2, 0, 3))
        cm_hist = obs * 5
        cm_future = model.ppf(np.random.random(size=1000), *(2, 0, 5)) * 5

        fit_obs, fit_cm_hist = pr._get_obs_and_cm_hist_fits(obs, cm_hist)
        assert all(
            np.abs(
                pr._apply_debiasing_steps(cm_future, fit_obs, fit_cm_hist)
                - cm_future / 5
            )
            < 1e-5
        )
