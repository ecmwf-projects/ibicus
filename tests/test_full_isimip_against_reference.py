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

import iris
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from cf_units import num2date

from ibicus.debias import ISIMIP

# ----- Helpers ----- #

# Given an iris-cube this returns the dates stored in the last time-dimension
def get_dates(x):
    time_dimension = x.coords()[2]
    dates = time_dimension.units.num2date(time_dimension.points)
    return dates


get_dates = np.vectorize(get_dates)

# This reads in the testing-data from ISIMIP stored in isimip3basd-master/data
def read_in_and_preprocess_testing_data(variable, data_path="tests/isimip_reference_data/"):

    # Load in data
    obs = iris.load_cube(data_path + variable + "_obs-hist_coarse_1979-2014.nc")
    cm_hist = iris.load_cube(data_path + variable + "_sim-hist_coarse_1979-2014.nc")
    cm_future = iris.load_cube(data_path + variable + "_sim-fut_coarse_2065-2100.nc")

    # Extract dates
    dates = {"time_obs": get_dates(obs), "time_cm_hist": get_dates(cm_hist), "time_cm_future": get_dates(cm_future)}

    # Convert to np.array (from masked-array)
    obs = np.array(obs.data)
    cm_hist = np.array(cm_hist.data)
    cm_future = np.array(cm_future.data)

    # Move time to first axis (our convention)
    obs = np.moveaxis(obs, -1, 0)
    cm_hist = np.moveaxis(cm_hist, -1, 0)
    cm_future = np.moveaxis(cm_future, -1, 0)

    return obs, cm_hist, cm_future, dates


def read_in_reference_data(variable, data_path="tests/isimip_reference_data/"):

    # Load in data
    debiased_data_reference = iris.load_cube(data_path + variable + "_sim-fut-basd_coarse_2065-2100.nc")

    # Move time to first axis (our convention)
    debiased_data_reference = np.array(debiased_data_reference.data)
    debiased_data_reference = np.moveaxis(debiased_data_reference, -1, 0)

    return debiased_data_reference


class TestFullISIMIPAgainstReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def _run_isimip(self, variable):
        obs, cm_hist, cm_future, dates = read_in_and_preprocess_testing_data(variable)
        debiaser = ISIMIP.from_variable(variable)
        debiased_values = debiaser.apply(obs, cm_hist, cm_future, **dates)
        return debiased_values

    def _test_pct_agreement(self, debiased_values, debiased_values_reference):
        pct = np.sum(np.isclose(debiased_values, debiased_values_reference)) / debiased_values.size
        assert np.isclose(pct, 1)

    def _test_linear_regression(
        self, debiased_values, debiased_values_reference, max_deviation_slope=1e-4, max_deviation_intercept=1e-3
    ):
        regression = scipy.stats.linregress(debiased_values_reference.flatten(), debiased_values.flatten())
        assert np.abs(regression.slope - 1) < max_deviation_slope
        assert np.abs(regression.intercept) < max_deviation_intercept

    def test_tas(self):
        variable = "tas"
        debiased_values = self._run_isimip(variable)
        debiased_values_reference = read_in_reference_data(variable)
        self._test_pct_agreement(debiased_values, debiased_values_reference)
        self._test_linear_regression(debiased_values, debiased_values_reference)

    def test_pr(self):
        variable = "pr"
        debiased_values = self._run_isimip(variable)
        debiased_values_reference = read_in_reference_data(variable)
        self._test_linear_regression(debiased_values, debiased_values_reference)

    def test_ps(self):
        variable = "ps"
        debiased_values = self._run_isimip(variable)
        debiased_values_reference = read_in_reference_data(variable)
        self._test_pct_agreement(debiased_values, debiased_values_reference)
        self._test_linear_regression(debiased_values, debiased_values_reference)

    def test_rlds(self):
        variable = "rlds"
        debiased_values = self._run_isimip(variable)
        debiased_values_reference = read_in_reference_data(variable)
        self._test_pct_agreement(debiased_values, debiased_values_reference)
        self._test_linear_regression(debiased_values, debiased_values_reference)

    def test_sfcwind(self):
        variable = "sfcWind"
        debiased_values = self._run_isimip(variable)
        debiased_values_reference = read_in_reference_data(variable)
        self._test_linear_regression(debiased_values, debiased_values_reference)

    def test_tasrange(self):
        variable = "tasrange"
        debiased_values = self._run_isimip(variable)
        debiased_values_reference = read_in_reference_data(variable)
        self._test_linear_regression(debiased_values, debiased_values_reference)

    def test_tasskew(self):
        variable = "tasskew"
        debiased_values = self._run_isimip(variable)
        debiased_values_reference = read_in_reference_data(variable)
        self._test_linear_regression(debiased_values, debiased_values_reference)

    def test_hurs(self):
        variable = "hurs"
        debiased_values = self._run_isimip(variable)
        debiased_values_reference = read_in_reference_data(variable)
        self._test_linear_regression(debiased_values, debiased_values_reference)
