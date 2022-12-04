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
from datetime import date

import numpy as np

from ibicus.debias import CDFt


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
        tas = CDFt.from_variable("tas")
        assert tas.SSR is False

        pr = CDFt.from_variable("pr")
        assert pr.SSR is True

        # Check default arguments
        hurs = CDFt.from_variable("hurs")
        assert hurs.delta_shift == "multiplicative"

        pr = CDFt.from_variable("pr")
        assert pr.delta_shift == "additive"
        assert pr.SSR is True

        psl = CDFt.from_variable("psl")
        assert psl.delta_shift == "additive"

        rlds = CDFt.from_variable("rlds")
        assert rlds.delta_shift == "additive"

        rsds = CDFt.from_variable("rsds")
        assert rsds.delta_shift == "multiplicative"

        sfcWind = CDFt.from_variable("sfcWind")
        assert sfcWind.delta_shift == "multiplicative"

        tas = CDFt.from_variable("tas")
        assert tas.delta_shift == "additive"

        tasmin = CDFt.from_variable("tasmin")
        assert tasmin.delta_shift == "additive"

        tasmax = CDFt.from_variable("tasmax")
        assert tasmax.delta_shift == "additive"

        tasrange = CDFt.from_variable("tasrange")
        assert tasrange.delta_shift == "additive"

        tasskew = CDFt.from_variable("tasskew")
        assert tasskew.delta_shift == "multiplicative"

    def test__init__(self):
        tas_1 = CDFt.from_variable("tas")
        tas_2 = CDFt()
        assert tas_1 == tas_2

        pr_1 = CDFt.from_variable("pr")
        pr_2 = CDFt(SSR=True)
        assert pr_1 == pr_2

    def test__apply_CDFt_mapping(self):
        tas = CDFt.from_variable("tas")

        # Test same vectors:
        obs = np.random.random(size=1000)
        cm_hist = obs
        cm_future = obs

        distance = get_min_distance_in_array(obs)

        assert check_equal_up_to_distance(
            tas._apply_CDFt_mapping(obs, cm_hist, cm_future), obs, distance
        )

        # Test: perfect match between obs and cm_hist
        obs = np.random.random(size=1000)
        cm_hist = obs
        cm_future = np.random.random(size=1000)

        assert np.allclose(tas._apply_CDFt_mapping(obs, cm_hist, cm_future), cm_future)

        # Test: perfect match between obs and cm_hist and cm_fut from new distribution
        obs = np.random.random(size=1000)
        cm_hist = obs
        cm_future = np.random.normal(size=1000)

        assert np.allclose(tas._apply_CDFt_mapping(obs, cm_hist, cm_future), cm_future)

        # Test: corrects mean difference
        obs = np.random.normal(size=1000)
        cm_hist = np.random.normal(size=1000) + 5
        cm_future = np.random.normal(size=1000) + 5

        assert (
            np.abs(
                np.mean(tas._apply_CDFt_mapping(obs, cm_hist, cm_future)) - np.mean(obs)
            )
            < 0.1
        )

        # Test: perfect match between obs and cm_hist up to translation and cm_fut from new distribution
        obs = np.random.random(size=1000)
        cm_hist = obs + 5
        cm_future = np.random.normal(size=1000)
        assert np.allclose(
            tas._apply_CDFt_mapping(obs, cm_hist, cm_future), cm_future - 5
        )

    def test__get_threshold(self):
        x, y, z = np.arange(-100, 200), np.arange(-100, 200), np.arange(-100, 200)
        assert CDFt._get_threshold(x, y, z) == 1

    def test__randomize_zero_values_between_zero_and_threshold(self):
        x = np.random.random(size=1000)
        threshold = 0.5
        x_new = CDFt._randomize_zero_values_between_zero_and_threshold(x, threshold)

        assert all(x[x >= threshold] == x_new[x >= threshold])
        assert all(x_new[x < threshold] < threshold)

    def test__set_values_below_threshold_to_zero(self):
        x = np.random.random(size=1000)
        threshold = 0.5
        x_new = CDFt._set_values_below_threshold_to_zero(x, threshold)
        assert all(x_new[x < threshold] == 0)

    def test__apply_SSR_steps_before_adjustment(self):
        x, y, z = (
            np.random.uniform(low=-1, high=1, size=1000),
            np.random.uniform(low=-1, high=1, size=1000),
            np.random.uniform(low=-1, high=1, size=1000),
        )
        x[x < 0], y[y < 0], z[z < 0] = 0, 0, 0

        x_new, y_new, z_new, threshold = CDFt._apply_SSR_steps_before_adjustment(
            x, y, z
        )

        assert all(x_new > 0)
        assert all(y_new > 0)
        assert all(z_new > 0)

        assert all(x_new[x > threshold] == x[x > threshold])
        assert all(y_new[y > threshold] == y[y > threshold])
        assert all(z_new[z > threshold] == z[z > threshold])

    def test__apply_SSR_steps_after_adjustment(self):
        x = np.random.random(size=1000)
        x_new = CDFt._apply_SSR_steps_after_adjustment(x, 0.5)

        assert all(x_new[x < 0.5] == 0)
        assert all(x_new[x >= 0.5] == x[x >= 0.5])

    def test__apply_on_window(self):
        tas = CDFt.from_variable("tas")

        # Test perfect match up to translation, depending on season
        obs = np.concatenate(
            [
                np.sin(2 * np.pi * i / 12) * np.random.random(size=1000)
                for i in range(12)
            ]
        )
        shift_cm_hist = np.concatenate(
            [
                np.repeat(np.random.uniform(low=-5, high=5, size=1), 1000)
                for i in range(12)
            ]
        )
        cm_hist = obs + shift_cm_hist
        cm_future = np.concatenate(
            [
                np.sin(2 * np.pi * i / 12) * np.random.random(size=1000)
                for i in range(12)
            ]
        )

        time_obs = np.concatenate(
            [np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)]
        )
        time_cm_hist = np.concatenate(
            [np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)]
        )
        time_cm_future = np.concatenate(
            [np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)]
        )

        debiased_cm_future = tas._apply_on_window(
            obs, cm_hist, cm_future, time_obs, time_cm_hist, time_cm_future
        )

        assert np.allclose(debiased_cm_future, cm_future - shift_cm_hist)

    def test_apply_location(self):
        tas = CDFt.from_variable("tas")

        # Test perfect match up to translation, depending on season
        obs = np.concatenate(
            [
                np.sin(2 * np.pi * i / 12) * np.random.random(size=1000)
                for i in range(12)
            ]
        )
        shift_cm_hist = np.concatenate(
            [
                np.repeat(np.random.uniform(low=-5, high=5, size=1), 1000)
                for i in range(12)
            ]
        )
        cm_hist = obs + shift_cm_hist
        cm_future = np.concatenate(
            [
                np.sin(2 * np.pi * i / 12) * np.random.random(size=1000)
                for i in range(12)
            ]
        )

        time_obs = np.concatenate(
            [np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)]
        )
        time_cm_hist = np.concatenate(
            [np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)]
        )
        time_cm_future = np.concatenate(
            [np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)]
        )

        debiased_cm_future = tas.apply_location(
            obs, cm_hist, cm_future, time_obs, time_cm_hist, time_cm_future
        )

        assert np.allclose(debiased_cm_future, cm_future - shift_cm_hist)

    def test_apply_location_without_time_specification(self):
        tas = CDFt.from_variable("tas")

        # Test perfect match
        obs = np.random.random(20000)
        cm_hist = obs
        cm_future = np.random.random(2000)

        debiased_cm_future = tas.apply_location(obs, cm_hist, cm_future)
        assert np.allclose(debiased_cm_future, cm_future)
