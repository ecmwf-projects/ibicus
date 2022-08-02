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

from PACKAGE_NAME.utils import RunningWindowModeOverYears


def check_different_maximally_up_to_1(x, y):
    return np.abs(x - y) <= 1


class TestCDFt(unittest.TestCase):
    def test__init__(self):
        window1 = RunningWindowModeOverYears(9, 17)
        window2 = RunningWindowModeOverYears(window_step_length_in_years=17, window_length_in_years=9)
        window3 = RunningWindowModeOverYears(window_length_in_years=4, window_step_length_in_years=2)

        # Test equality
        assert window1 == window2
        assert window1 != window3

        # Test validators
        with self.assertRaises(ValueError):
            RunningWindowModeOverYears(0, 2)

        with self.assertRaises(ValueError):
            RunningWindowModeOverYears(2, 0)

        with self.assertRaises(TypeError):
            RunningWindowModeOverYears("2", 0)

        # Test converter
        window4 = RunningWindowModeOverYears(8.6, 17.0)
        assert window1 == window4

    def test_get_if_in_chosen_years(self):
        years = np.arange(2010, 2050)
        chosen_years = np.arange(2040 - 10 // 2, 2040 + 10 // 2)

        assert RunningWindowModeOverYears.get_if_in_chosen_years(years, chosen_years).sum() == 10

    def test__get_years_forming_window_centers(self):
        window = RunningWindowModeOverYears(9, 17)

        # Only 1 year
        unique_years = np.array([2020])
        assert np.array_equal(window._get_years_forming_window_centers(unique_years), np.array([2020]))

        # Only 5 years
        unique_years = np.arange(2015, 2021)
        assert window._get_years_forming_window_centers(unique_years).size == 1

        # Exactly tas.window_step_length_in_years years
        unique_years = np.arange(2015, 2015 + window.window_step_length_in_years)
        assert window._get_years_forming_window_centers(unique_years).size == 1

        # Exactly tas.window_step_length_in_years + 1 years
        unique_years = np.arange(2015, 2015 + window.window_step_length_in_years + 1)
        assert window._get_years_forming_window_centers(unique_years).size == 2

        # Enough window centers
        unique_years = np.arange(2020, 2100)
        nr_of_window_centers = window._get_years_forming_window_centers(unique_years).size
        assert check_different_maximally_up_to_1(
            nr_of_window_centers, unique_years.size // window.window_step_length_in_years
        )

        # First and last one not drastically different
        unique_years = np.arange(2020, 2100)
        window_centers = window._get_years_forming_window_centers(unique_years)
        assert check_different_maximally_up_to_1(
            window_centers[0] - unique_years.min(), unique_years.max() - window_centers[-1]
        )

        # Equally spaced
        unique_years = np.arange(2020, 2100)
        window_centers = window._get_years_forming_window_centers(unique_years)
        assert all(
            window_centers[1 : (window_centers.size - 1)] - window_centers[0 : (window_centers.size - 2)]
            == window.window_step_length_in_years,
        )

    def test__get_years_in_window(self):
        window = RunningWindowModeOverYears(17, 9)
        years_in_window = window._get_years_in_window(2020)

        assert years_in_window.size == window.window_length_in_years
        assert np.array_equal(years_in_window, np.arange(2012, 2028 + 1))

    def test__get_years_in_window_that_are_adjusted(self):
        window = RunningWindowModeOverYears(17, 9)
        years_in_window_bias_corrected = window._get_years_in_window_that_are_adjusted(2020)

        assert years_in_window_bias_corrected.size == window.window_step_length_in_years
        assert np.array_equal(years_in_window_bias_corrected, np.arange(2016, 2024 + 1))

    def _test_all_bias_corrected_and_length_each_window(self, window, start_year, end_year):
        years = np.arange(start_year, end_year)

        debiased_years = []
        for years_to_adjust, years_in_window in window.use(years):

            assert years_in_window.size == window.window_length_in_years
            assert check_different_maximally_up_to_1(years_to_adjust.size, window.window_step_length_in_years)

            debiased_years.append(years_to_adjust)

        test = debiased_years.copy()
        debiased_years = np.concatenate(debiased_years)

        # Check that no years are double debiased
        assert len(np.unique(debiased_years)) == len(debiased_years)

        # Check that all years are given
        assert all(np.in1d(years, debiased_years))

    def test_use(self):
        for step_length in range(1, 10):
            for length in range(1, 10):
                window = RunningWindowModeOverYears(length, step_length)
                for i in range(2000, 2050):
                    for j in range(i + 1, 2050):
                        self._test_all_bias_corrected_and_length_each_window(window, i, j)
