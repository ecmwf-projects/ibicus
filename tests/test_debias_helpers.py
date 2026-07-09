# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Unit tests for the per-window transformation helpers and the constructor
validation of the simpler debiasers (LinearScaling, DeltaChange,
ScaledDistributionMapping, QuantileDeltaMapping) and the running-window base
class :py:class:`SeasonalRunningWindowDebiaser`.
"""

import unittest

import numpy as np

from ibicus.debias import (
    DeltaChange,
    LinearScaling,
    QuantileDeltaMapping,
    ScaledDistributionMapping,
)


class TestLinearScalingHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)
        cls.obs = np.random.normal(loc=0, scale=1, size=1000)
        cls.cm_hist = np.random.normal(loc=5, scale=1, size=1000)
        cls.cm_future = np.random.normal(loc=7, scale=1, size=1000)

    def test_apply_on_seasonal_window_additive(self):
        debiaser = LinearScaling.from_variable("tas")
        assert debiaser.delta_type == "additive"

        out = debiaser.apply_on_seasonal_window(self.obs, self.cm_hist, self.cm_future)
        expected = self.cm_future - (np.mean(self.cm_hist) - np.mean(self.obs))
        assert np.allclose(out, expected)

    def test_apply_on_seasonal_window_multiplicative(self):
        debiaser = LinearScaling.from_variable("pr")
        assert debiaser.delta_type == "multiplicative"

        obs = np.abs(self.obs) + 1
        cm_hist = np.abs(self.cm_hist) + 1
        cm_future = np.abs(self.cm_future) + 1

        out = debiaser.apply_on_seasonal_window(obs, cm_hist, cm_future)
        expected = cm_future * (np.mean(obs) / np.mean(cm_hist))
        assert np.allclose(out, expected)

    def test_invalid_delta_type_rejected(self):
        with self.assertRaises(ValueError):
            LinearScaling.from_variable("tas", delta_type="not_a_type")


class TestDeltaChangeHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)
        cls.obs = np.random.normal(loc=0, scale=1, size=1000)
        cls.cm_hist = np.random.normal(loc=5, scale=1, size=1000)
        cls.cm_future = np.random.normal(loc=7, scale=1, size=1000)

    def test_apply_within_year_window_additive(self):
        debiaser = DeltaChange.from_variable("tas")
        assert debiaser.delta_type == "additive"

        out = debiaser._apply_on_within_year_window(
            self.obs, self.cm_hist, self.cm_future
        )
        expected = self.obs + (np.mean(self.cm_future) - np.mean(self.cm_hist))
        assert np.allclose(out, expected)

    def test_apply_within_year_window_multiplicative(self):
        debiaser = DeltaChange.from_variable("pr")
        assert debiaser.delta_type == "multiplicative"

        obs = np.abs(self.obs) + 1
        cm_hist = np.abs(self.cm_hist) + 1
        cm_future = np.abs(self.cm_future) + 1

        out = debiaser._apply_on_within_year_window(obs, cm_hist, cm_future)
        expected = obs * (np.mean(cm_future) / np.mean(cm_hist))
        assert np.allclose(out, expected)

    def test_delta_change_output_has_obs_length(self):
        # DeltaChange transforms the observations, so output length matches obs
        debiaser = DeltaChange.from_variable("tas")
        out = debiaser._apply_on_within_year_window(
            self.obs[:500], self.cm_hist, self.cm_future
        )
        assert out.shape == self.obs[:500].shape


class TestScaledDistributionMappingHelpers(unittest.TestCase):
    def test_for_precipitation_sets_threshold(self):
        debiaser = ScaledDistributionMapping.for_precipitation(pr_lower_threshold=0.5)
        assert debiaser.pr_lower_threshold == 0.5
        assert debiaser.variable == "Daily mean precipitation flux"

    def test_invalid_mapping_type_rejected(self):
        with self.assertRaises(ValueError):
            ScaledDistributionMapping.from_variable("tas", mapping_type="not_a_type")

    def test_mapping_types_available(self):
        # tas uses absolute scaled distribution mapping
        tas = ScaledDistributionMapping.from_variable("tas")
        assert tas.mapping_type == "absolute"
        # pr uses relative scaled distribution mapping
        pr = ScaledDistributionMapping.from_variable("pr")
        assert pr.mapping_type == "relative"


class TestQuantileDeltaMappingHelpers(unittest.TestCase):
    def test_for_precipitation_nonparametric_and_parametric(self):
        nonparam = QuantileDeltaMapping.for_precipitation(mapping_type="nonparametric")
        assert nonparam.variable == "Daily mean precipitation flux"

        param = QuantileDeltaMapping.for_precipitation(mapping_type="parametric")
        assert param.variable == "Daily mean precipitation flux"

    def test_for_precipitation_invalid_mapping_type_raises(self):
        with self.assertRaises(ValueError):
            QuantileDeltaMapping.for_precipitation(mapping_type="not_a_type")

    def test_invalid_trend_preservation_rejected(self):
        with self.assertRaises(ValueError):
            QuantileDeltaMapping.from_variable("tas", trend_preservation="not_a_method")

    def test_invalid_mapping_type_rejected(self):
        with self.assertRaises(ValueError):
            QuantileDeltaMapping.from_variable("tas", mapping_type="not_a_type")


class TestRunningWindowDebiaserValidation(unittest.TestCase):
    def test_step_length_greater_than_length_raises(self):
        # In running-window mode the step length must not exceed the window length
        with self.assertRaises(ValueError):
            LinearScaling.from_variable(
                "tas",
                running_window_mode=True,
                running_window_length=5,
                running_window_step_length=11,
            )

    def test_valid_running_window_configuration(self):
        debiaser = LinearScaling.from_variable(
            "tas",
            running_window_mode=True,
            running_window_length=31,
            running_window_step_length=3,
        )
        assert hasattr(debiaser, "running_window")

    def test_non_positive_window_length_rejected(self):
        with self.assertRaises(ValueError):
            LinearScaling.from_variable("tas", running_window_length=0)


if __name__ == "__main__":
    unittest.main()
