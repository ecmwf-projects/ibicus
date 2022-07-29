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

from PACKAGE_NAME.debias import CDFt


def check_different_maximally_up_to_1(x, y):
    return np.abs(x - y) <= 1


def check_equal_up_to_distance(x, y, distance):
    return all(np.abs(x - y) < distance)


def get_min_distance_in_array(x):
    x = np.sort(x)
    return np.min(np.abs(x[0 : (x.size - 1)] - x[1:]))


class TestCDFt(unittest.TestCase):
    def test_from_variable(self):
        tas = CDFt.from_variable("tas")
        assert tas.SSR is False

        pr = CDFt.from_variable("pr")
        assert pr.SSR is True

    def test__init__(self):
        tas_1 = CDFt.from_variable("tas")
        tas_2 = CDFt()
        assert tas_1 == tas_2

        pr_1 = CDFt.from_variable("pr")
        pr_2 = CDFt(SSR=True)
        assert pr_1 == pr_2

    def test__get_if_in_chosen_years(self):
        years = np.arange(2010, 2050)
        chosen_years = np.arange(2040 - 10 // 2, 2040 + 10 // 2)

        assert CDFt._get_if_in_chosen_years(years, chosen_years).sum() == 10

    def test__get_years_forming_window_centers(self):
        tas = CDFt.from_variable("tas")

        # Only 1 year
        unique_years = np.array([2020])
        assert all(tas._get_years_forming_window_centers(unique_years) == np.array([2020]))

        # Only 5 years
        unique_years = np.arange(2015, 2021)
        assert tas._get_years_forming_window_centers(unique_years).size == 1

        # Exactly tas.running_window_step_length_in_years years
        unique_years = np.arange(2015, 2015 + tas.running_window_step_length_in_years)
        assert tas._get_years_forming_window_centers(unique_years).size == 1

        # Exactly tas.running_window_step_length_in_years + 1 years
        unique_years = np.arange(2015, 2015 + tas.running_window_step_length_in_years + 1)
        assert tas._get_years_forming_window_centers(unique_years).size == 2

        # Enough window centers
        unique_years = np.arange(2020, 2100)
        nr_of_window_centers = tas._get_years_forming_window_centers(unique_years).size
        assert check_different_maximally_up_to_1(
            nr_of_window_centers, unique_years.size // tas.running_window_step_length_in_years
        )

        # First and last one not drastically different
        unique_years = np.arange(2020, 2100)
        window_centers = tas._get_years_forming_window_centers(unique_years)
        assert check_different_maximally_up_to_1(
            window_centers[0] - unique_years.min(), unique_years.max() - window_centers[-1]
        )

        # Equally spaced
        unique_years = np.arange(2020, 2100)
        window_centers = tas._get_years_forming_window_centers(unique_years)
        assert all(
            window_centers[1 : (window_centers.size - 1)] - window_centers[0 : (window_centers.size - 2)]
            == tas.running_window_step_length_in_years
        )

    def test__get_years_in_window(self):
        tas = CDFt.from_variable("tas")
        years_in_window = tas._get_years_in_window(2020)

        assert years_in_window.size == tas.running_window_length_in_years
        assert all(years_in_window == np.arange(2012, 2028 + 1))

    def test__get_years_in_window_that_are_bias_corrected(self):
        tas = CDFt.from_variable("tas")
        years_in_window_bias_corrected = tas._get_years_in_window_that_are_bias_corrected(2020)

        assert years_in_window_bias_corrected.size == tas.running_window_step_length_in_years
        assert all(years_in_window_bias_corrected == np.arange(2016, 2024 + 1))

    def _test_all_bias_corrected_and_length_each_window(self, debiaser, start_year, end_year):
        years = np.arange(start_year, end_year)
        window_centers = debiaser._get_years_forming_window_centers(years)

        debiased_years = []
        for window_center in window_centers:
            years_in_window = debiaser._get_years_in_window(window_center)
            years_to_debias = debiaser._get_years_in_window_that_are_bias_corrected(window_center)

            assert years_in_window.size == debiaser.running_window_length_in_years
            assert years_to_debias.size == debiaser.running_window_step_length_in_years

            debiased_years.append(years_to_debias)

        debiased_years = np.concatenate(debiased_years)

        # Check that no years are double debiased
        assert len(np.unique(debiased_years)) == len(debiased_years)

        # Check that all years are given
        assert all(np.in1d(years, debiased_years))

    def test_all_bias_corrected_and_length_each_window(self):
        tas = CDFt.from_variable("tas")
        for i in range(2000, 2050):
            for j in range(i + 1, 2050):
                self._test_all_bias_corrected_and_length_each_window(tas, i, j)

    def test__apply_CDFt_mapping(self):
        tas = CDFt.from_variable("tas")

        # Test same vectors:
        obs = np.random.random(size=1000)
        cm_hist = obs
        cm_future = obs

        distance = get_min_distance_in_array(obs)

        assert check_equal_up_to_distance(tas._apply_CDFt_mapping(obs, cm_hist, cm_future), obs, distance)

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

        assert np.abs(np.mean(tas._apply_CDFt_mapping(obs, cm_hist, cm_future)) - np.mean(obs)) < 0.1

        # Test: perfect match between obs and cm_hist up to translation and cm_fut from new distribution
        obs = np.random.random(size=1000)
        cm_hist = obs + 5
        cm_future = np.random.normal(size=1000)
        assert np.allclose(tas._apply_CDFt_mapping(obs, cm_hist, cm_future), cm_future - 5)

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

        x_new, y_new, z_new, threshold = CDFt._apply_SSR_steps_before_adjustment(x, y, z)

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
        obs = np.concatenate([np.sin(2 * np.pi * i / 12) * np.random.random(size=1000) for i in range(12)])
        shift_cm_hist = np.concatenate([np.repeat(np.random.uniform(low=-5, high=5, size=1), 1000) for i in range(12)])
        cm_hist = obs + shift_cm_hist
        cm_future = np.concatenate([np.sin(2 * np.pi * i / 12) * np.random.random(size=1000) for i in range(12)])

        time_obs = np.concatenate([np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)])
        time_cm_hist = np.concatenate([np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)])
        time_cm_future = np.concatenate([np.repeat(date(2000, i, 1), 1000) for i in range(1, 13)])

        debiased_cm_future = tas._apply_on_window(obs, cm_hist, cm_future, time_obs, time_cm_hist, time_cm_future)

        assert np.allclose(debiased_cm_future, cm_future - shift_cm_hist)
