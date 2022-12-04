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

from ibicus.debias import LinearScaling


def check_different_maximally_up_to_1(x, y):
    return np.abs(x - y) <= 1


def check_equal_up_to_distance(x, y, distance):
    return all(np.abs(x - y) < distance)


def get_min_distance_in_array(x):
    x = np.sort(x)
    return np.min(np.abs(x[0 : (x.size - 1)] - x[1:]))


class TestLinearScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_from_variable(self):
        tas = LinearScaling.from_variable("tas")
        assert tas.delta_type == "additive"

        pr = LinearScaling.from_variable("pr")
        assert pr.delta_type == "multiplicative"

        with self.assertRaises(ValueError):
            LinearScaling(delta_type="none")

        # Check default arguments
        hurs = LinearScaling.from_variable("hurs")
        assert hurs.delta_type == "multiplicative"

        pr = LinearScaling.from_variable("pr")
        assert pr.delta_type == "multiplicative"

        psl = LinearScaling.from_variable("psl")
        assert psl.delta_type == "additive"

        rlds = LinearScaling.from_variable("rlds")
        assert rlds.delta_type == "additive"

        rsds = LinearScaling.from_variable("rsds")
        assert rsds.delta_type == "multiplicative"

        sfcWind = LinearScaling.from_variable("sfcWind")
        assert sfcWind.delta_type == "multiplicative"

        tas = LinearScaling.from_variable("tas")
        assert tas.delta_type == "additive"

        tasmin = LinearScaling.from_variable("tasmin")
        assert tasmin.delta_type == "additive"

        tasmax = LinearScaling.from_variable("tasmax")
        assert tasmax.delta_type == "additive"

    def test__init__(self):
        tas_1 = LinearScaling.from_variable("tas")
        tas_2 = LinearScaling(delta_type="additive")
        assert tas_1 == tas_2

        tas_1 = LinearScaling.from_variable("tas", delta_type="multiplicative")
        tas_2 = LinearScaling(delta_type="multiplicative")
        assert tas_1 == tas_2

        pr_1 = LinearScaling.from_variable("pr")
        pr_2 = LinearScaling(delta_type="multiplicative")
        assert pr_1 == pr_2

        pr_1 = LinearScaling.from_variable("pr", delta_type="additive")
        pr_2 = LinearScaling(delta_type="additive")
        assert pr_1 == pr_2

    def test_apply_location_additive(self):
        tas = LinearScaling.from_variable("tas")

        # Test: perfect match between obs and cm_hist
        obs = np.random.normal(size=1000)
        cm_hist = obs
        cm_future = np.random.normal(size=1000)

        assert np.allclose(tas.apply_location(obs, cm_hist, cm_future), cm_future)

        # Test: corrects mean of cm hist
        obs = np.random.normal(size=1000)
        cm_hist = np.random.normal(size=1000) + 5
        cm_future = cm_hist

        assert np.isclose(
            np.mean(tas.apply_location(obs, cm_hist, cm_future)), np.mean(obs)
        )

        # Test: corrects mean difference
        obs = np.random.normal(size=1000)
        cm_hist = np.random.normal(size=1000) + 5
        cm_future = np.random.normal(size=1000) + 5

        assert (
            np.abs(np.mean(tas.apply_location(obs, cm_hist, cm_future)) - np.mean(obs))
            < 0.3
        )

    def test_apply_location_multiplicative(self):
        pr = LinearScaling.from_variable("pr")

        # Test: perfect match between obs and cm_hist
        obs = np.random.uniform(size=1000)
        cm_hist = obs
        cm_future = np.random.uniform(size=1000)

        assert np.allclose(pr.apply_location(obs, cm_hist, cm_future), cm_future)

        # Test: corrects mean of cm hist
        obs = scipy.stats.gamma.rvs(a=2, scale=3, size=1000)
        cm_hist = scipy.stats.gamma.rvs(a=2, scale=3, size=1000) + 5
        cm_future = cm_hist

        assert np.isclose(
            np.mean(pr.apply_location(obs, cm_hist, cm_future)), np.mean(obs)
        )

        # Test: corrects mean and variance difference
        obs = scipy.stats.gamma.rvs(a=2, scale=3, size=1000)
        cm_hist = scipy.stats.gamma.rvs(a=2, scale=3, size=1000) * 5
        cm_future = scipy.stats.gamma.rvs(a=2, scale=3, size=1000) * 5

        assert (
            np.abs(np.mean(pr.apply_location(obs, cm_hist, cm_future)) - np.mean(obs))
            < 0.3
        )
