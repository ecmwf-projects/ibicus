# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Unit tests for the individual helper methods and ISIMIP steps of the
:py:class:`ibicus.debias.ISIMIP` debiaser.

These complement the integration-style tests in ``test_isimip.py`` by testing
the building blocks (bound/threshold logic, the trend-transfer of step 5, the
step-6 frequency-adjustment helpers, scaling of step 1/8, ...) in isolation.
"""

import unittest

import numpy as np
import scipy.stats

from ibicus.debias import ISIMIP


def _make_isimip(variable="tas", **kwargs):
    return ISIMIP.from_variable(variable, **kwargs)


class TestISIMIPBoundAndThresholdProperties(unittest.TestCase):
    def test_properties_for_lower_bounded_variable(self):
        # pr has a lower bound and lower threshold but no upper ones
        debiaser = _make_isimip("pr")
        assert debiaser.has_lower_bound
        assert debiaser.has_lower_threshold
        assert not debiaser.has_upper_bound
        assert not debiaser.has_upper_threshold
        assert debiaser.has_bound
        assert debiaser.has_threshold

    def test_properties_for_unbounded_variable(self):
        # tas has neither bounds nor thresholds
        debiaser = _make_isimip("tas")
        assert not debiaser.has_lower_bound
        assert not debiaser.has_lower_threshold
        assert not debiaser.has_upper_bound
        assert not debiaser.has_upper_threshold
        assert not debiaser.has_bound
        assert not debiaser.has_threshold

    def test_properties_for_double_bounded_variable(self):
        # hurs / rsds-like: set both bounds and thresholds manually
        debiaser = _make_isimip("tas")
        debiaser.lower_bound = 0.0
        debiaser.lower_threshold = 1.0
        debiaser.upper_bound = 100.0
        debiaser.upper_threshold = 99.0

        assert debiaser.has_lower_bound
        assert debiaser.has_lower_threshold
        assert debiaser.has_upper_bound
        assert debiaser.has_upper_threshold
        assert debiaser.has_bound
        assert debiaser.has_threshold


class TestISIMIPThresholdMasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.debiaser = _make_isimip("tas")
        cls.debiaser.lower_threshold = 0.0
        cls.debiaser.upper_threshold = 10.0

    def test_mask_for_values_beyond_lower_threshold(self):
        x = np.array([-1.0, 0.0, 1.0, 5.0])
        mask = self.debiaser._get_mask_for_values_beyond_lower_threshold(x)
        assert np.array_equal(mask, np.array([True, True, False, False]))

    def test_mask_for_values_beyond_upper_threshold(self):
        x = np.array([5.0, 9.0, 10.0, 11.0])
        mask = self.debiaser._get_mask_for_values_beyond_upper_threshold(x)
        assert np.array_equal(mask, np.array([False, False, True, True]))

    def test_mask_for_values_between_thresholds(self):
        x = np.array([-1.0, 0.0, 5.0, 10.0, 11.0])
        mask = self.debiaser._get_mask_for_values_between_thresholds(x)
        assert np.array_equal(mask, np.array([False, False, True, False, False]))

    def test_get_values_between_thresholds(self):
        x = np.array([-1.0, 0.0, 3.0, 7.0, 10.0, 11.0])
        out = self.debiaser._get_values_between_thresholds(x)
        assert np.array_equal(out, np.array([3.0, 7.0]))

    def test_proportion_of_days_beyond_thresholds(self):
        x = np.array([-1.0, 0.0, 5.0, 10.0])
        # 2 of 4 are at/below the lower threshold (0.0)
        assert np.isclose(
            self.debiaser.get_proportion_of_days_beyond_lower_threshold(x), 0.5
        )
        # 1 of 4 is at/above the upper threshold (10.0)
        assert np.isclose(
            self.debiaser.get_proportion_of_days_beyond_upper_threshold(x), 0.25
        )


class TestISIMIPStep1And8Scaling(unittest.TestCase):
    def test_scale_and_rescale_are_inverse(self):
        vals = np.array([2.0, 4.0, 6.0, 8.0])
        days_of_year_vals = np.array([1, 2, 3, 2])
        annual_cycle = np.array([2.0, 4.0, 5.0])
        unique_days = np.array([1, 2, 3])

        scaled = ISIMIP._step1_scale_by_annual_cycle_of_upper_bounds(
            vals, days_of_year_vals, annual_cycle, unique_days
        )
        rescaled = ISIMIP._step8_rescale_by_annual_cycle_of_upper_bounds(
            scaled, days_of_year_vals, annual_cycle, unique_days
        )
        assert np.allclose(rescaled, vals)

    def test_scale_handles_zero_in_annual_cycle(self):
        vals = np.array([3.0, 5.0])
        days_of_year_vals = np.array([1, 2])
        annual_cycle = np.array([0.0, 5.0])  # a zero entry should not divide
        unique_days = np.array([1, 2])

        scaled = ISIMIP._step1_scale_by_annual_cycle_of_upper_bounds(
            vals, days_of_year_vals, annual_cycle, unique_days
        )
        # Zero in the cycle => scaling factor of 1.0, value unchanged
        assert scaled[0] == 3.0
        assert np.isclose(scaled[1], 1.0)

    def test_calculate_debiased_annual_cycle_equal_days(self):
        annual_cycle_obs = np.array([2.0, 4.0])
        annual_cycle_cm_hist = np.array([1.0, 2.0])
        annual_cycle_cm_future = np.array([2.0, 2.0])
        days = np.array([1, 2])

        debiased = ISIMIP._step1_calculate_debiased_annual_cycle_of_upper_bounds(
            annual_cycle_obs,
            days,
            annual_cycle_cm_hist,
            days,
            annual_cycle_cm_future,
            days,
        )
        # factor = clip(cm_future/cm_hist, 0.1, 10) = [2, 1]; obs * factor = [4, 4]
        assert np.allclose(debiased, np.array([4.0, 4.0]))


class TestISIMIPStep2Imputation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_mask_for_values_to_impute(self):
        debiaser = _make_isimip("prsnratio")
        x = np.array([1.0, np.nan, np.inf, 2.0])
        mask = debiaser._step2_get_mask_for_values_to_impute(x)
        assert np.array_equal(mask, np.array([False, True, True, False]))

    def test_impute_values_single_valid_value(self):
        debiaser = _make_isimip("prsnratio")
        x = np.array([np.nan, 0.7, np.nan, np.inf])
        out = debiaser._step2_impute_values(x.copy())
        # The single valid value is inserted everywhere
        assert np.allclose(out, np.repeat(0.7, 4))

    def test_impute_values_all_invalid_raises(self):
        debiaser = _make_isimip("prsnratio")
        x = np.array([np.nan, np.nan, np.inf])
        with self.assertRaises(ValueError):
            debiaser._step2_impute_values(x)

    def test_impute_values_keeps_valid_values(self):
        debiaser = _make_isimip("prsnratio")
        x = np.concatenate([np.random.uniform(0, 1, 100), [np.nan, np.nan]])
        out = debiaser._step2_impute_values(x.copy())
        # No missing values left, and imputed values are within the original range
        assert not np.any(np.isnan(out))
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestISIMIPStep3Detrending(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_remove_trend_removes_linear_trend(self):
        debiaser = _make_isimip("tas")
        years = np.repeat(np.arange(2000, 2050), 4)
        # Strong linear trend in the yearly means
        trend_per_year = 0.5
        x = (years - 2000) * trend_per_year + np.random.normal(
            scale=0.01, size=years.size
        )

        detrended, trend = debiaser._step3_remove_trend(x, years)

        # The detrended series has (almost) no remaining yearly trend
        unique_years = np.unique(years)
        yearly_means = np.array([np.mean(detrended[years == y]) for y in unique_years])
        slope = scipy.stats.linregress(unique_years, yearly_means).slope
        assert np.abs(slope) < 1e-2

        # Reconstruction holds
        assert np.allclose(x, detrended + trend)

    def test_remove_trend_no_significant_trend(self):
        debiaser = _make_isimip("tas")
        debiaser.detrending_with_significance_test = True
        years = np.repeat(np.arange(2000, 2050), 4)
        # No trend, just noise
        x = np.random.normal(scale=1.0, size=years.size)

        detrended, trend = debiaser._step3_remove_trend(x, years)
        # Insignificant trend is not removed
        assert np.allclose(trend, 0.0)
        assert np.allclose(detrended, x)


class TestISIMIPStep4Randomization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_randomize_values_between_lower_threshold_and_bound(self):
        debiaser = _make_isimip("pr")
        # Values at/below the lower threshold get randomized into (bound, threshold)
        vals = np.concatenate([np.zeros(50), np.random.uniform(1, 5, 50)])
        out = debiaser._step4_randomize_values_between_lower_threshold_and_bound(
            vals.copy()
        )
        below = debiaser._get_mask_for_values_beyond_lower_threshold(vals)
        # Randomized values lie strictly between lower bound and lower threshold
        assert np.all(out[below] >= debiaser.lower_bound)
        assert np.all(out[below] <= debiaser.lower_threshold)
        assert np.mean(out[below]) > debiaser.lower_bound
        assert np.mean(out[below]) < debiaser.lower_threshold

        # Values above threshold are untouched
        assert np.allclose(out[~below], vals[~below])

    def test_randomize_values_between_upper_threshold_and_bound(self):
        debiaser = _make_isimip("tas")
        debiaser.upper_bound = 10.0
        debiaser.upper_threshold = 8.0
        vals = np.concatenate([np.repeat(10.0, 30), np.random.uniform(0, 7, 30)])
        out = debiaser._step4_randomize_values_between_upper_threshold_and_bound(
            vals.copy()
        )
        beyond = debiaser._get_mask_for_values_beyond_upper_threshold(vals)
        assert np.all(out[beyond] >= debiaser.upper_threshold)
        assert np.all(out[beyond] <= debiaser.upper_bound)
        assert np.mean(out[beyond]) < debiaser.upper_bound
        assert np.mean(out[beyond]) > debiaser.upper_threshold


class TestISIMIPStep5TrendTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_additive_trend(self):
        debiaser = _make_isimip("tas")
        debiaser.trend_preservation_method = "additive"
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        cm_hist = np.array([1.0, 2.0, 3.0, 4.0])
        cm_future = cm_hist + 2.0
        out = debiaser._step5_transfer_trend(obs, cm_hist, cm_future)
        assert np.allclose(out, obs + 2.0)

    def test_multiplicative_trend(self):
        debiaser = _make_isimip("pr")
        debiaser.trend_preservation_method = "multiplicative"
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        cm_hist = np.array([1.0, 2.0, 3.0, 4.0])
        cm_future = cm_hist * 2.0
        out = debiaser._step5_transfer_trend(obs, cm_hist, cm_future)
        assert np.allclose(out, obs * 2.0)

    def test_mixed_trend_runs(self):
        debiaser = _make_isimip("tas")
        debiaser.trend_preservation_method = "mixed"
        obs = np.random.uniform(1, 10, 100)
        cm_hist = np.random.uniform(1, 10, 100)
        cm_future = cm_hist + np.random.uniform(1, 2, 100)
        out = debiaser._step5_transfer_trend(obs, cm_hist, cm_future)
        assert out.shape == obs.shape
        assert np.all(np.isfinite(out))
        assert np.all(out >= obs + 1)

    def test_bounded_trend_respects_bounds(self):
        debiaser = _make_isimip("tas")
        debiaser.trend_preservation_method = "bounded"
        debiaser.lower_bound = 0.0
        debiaser.upper_bound = 1.0
        obs = np.random.uniform(0.1, 0.9, 200)
        cm_hist = np.random.uniform(0.1, 0.9, 200)
        cm_future = np.random.uniform(0.1, 0.9, 200)
        out = debiaser._step5_transfer_trend(obs, cm_hist, cm_future)
        # Result is enforced to lie within [lower_bound, upper_bound]
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_invalid_trend_preservation_rejected(self):
        # The validator forbids invalid trend preservation methods
        with self.assertRaises(ValueError):
            _make_isimip("tas", trend_preservation_method="not_a_method")


class TestISIMIPStep6Helpers(unittest.TestCase):
    def test_calculate_percent_values_beyond_threshold(self):
        mask = np.array([True, True, False, False, False])
        assert np.isclose(
            ISIMIP._step6_calculate_percent_values_beyond_threshold(mask), 0.4
        )

    def test_get_P_obs_future_equal_bias(self):
        # P_cm_hist close to P_obs_hist -> returns P_cm_future
        assert ISIMIP._step6_get_P_obs_future(0.5, 0.5, 0.7) == 0.7

    def test_get_P_obs_future_positive_bias_branch(self):
        # P_cm_future <= P_cm_hist and P_cm_hist > P_obs_hist
        out = ISIMIP._step6_get_P_obs_future(0.2, 0.8, 0.4)
        assert np.isclose(out, 0.2 * 0.4 / 0.8)

    def test_get_P_obs_future_negative_bias_branch(self):
        # P_cm_future >= P_cm_hist and P_cm_hist < P_obs_hist
        out = ISIMIP._step6_get_P_obs_future(0.8, 0.2, 0.4)
        assert np.isclose(out, 1 - (1 - 0.8) * (1 - 0.4) / (1 - 0.2))

    def test_get_P_obs_future_else_branch(self):
        # Additive correction case
        out = ISIMIP._step6_get_P_obs_future(0.5, 0.3, 0.2)
        assert np.isclose(out, 0.5 + 0.2 - 0.3)

    def test_scale_nr_of_entries_to_set_to_bounds(self):
        lower, upper = ISIMIP._step6_scale_nr_of_entries_to_set_to_bounds(60, 60, 100)
        assert lower == 50
        assert upper == 50

    def test_get_mask_for_entries_to_set_to_lower_bound(self):
        cm_future_sorted = np.arange(10.0)
        mask = ISIMIP._step6_get_mask_for_entries_to_set_to_lower_bound(
            3, cm_future_sorted
        )
        assert np.array_equal(np.where(mask)[0], np.array([0, 1, 2]))

    def test_get_mask_for_entries_to_set_to_upper_bound(self):
        cm_future_sorted = np.arange(10.0)
        mask = ISIMIP._step6_get_mask_for_entries_to_set_to_upper_bound(
            2, cm_future_sorted
        )
        assert np.array_equal(np.where(mask)[0], np.array([8, 9]))

    def test_fit_good_enough(self):
        np.random.seed(1)
        data = scipy.stats.norm.rvs(size=2000)
        good_fit = scipy.stats.norm.fit(data)
        assert ISIMIP._step6_fit_good_enough(data, scipy.stats.norm, good_fit)

        # A badly mismatched fit fails the goodness-of-fit check
        bad_fit = (50.0, 1.0)
        assert not ISIMIP._step6_fit_good_enough(data, scipy.stats.norm, bad_fit)

    def test_get_nr_of_entries_to_set_to_bound_no_frequency_correction(self):
        debiaser = _make_isimip("tas")
        debiaser.bias_correct_frequencies_of_values_beyond_thresholds = False
        obs_mask = np.array([True, True, False, False])  # 50% beyond
        cm_hist_mask = np.array([True, False, False, False])
        cm_future_mask = np.array([True, True, True, False, False, False])
        nr = debiaser._step6_get_nr_of_entries_to_set_to_bound(
            obs_mask, cm_hist_mask, cm_future_mask
        )
        # Without frequency correction, uses obs proportion (0.5) * size (6) = 3
        assert nr == 3


class TestISIMIPStep7(unittest.TestCase):
    def test_step7_adds_trend_when_detrending(self):
        debiaser = _make_isimip("tas")
        debiaser.detrending = True
        cm_future = np.array([1.0, 2.0, 3.0])
        trend = np.array([0.5, 0.5, 0.5])
        out = debiaser.step7(cm_future, trend)
        assert np.allclose(out, cm_future + trend)

    def test_step7_no_op_when_not_detrending(self):
        debiaser = _make_isimip("tas")
        debiaser.detrending = False
        cm_future = np.array([1.0, 2.0, 3.0])
        trend = np.array([0.5, 0.5, 0.5])
        out = debiaser.step7(cm_future, trend)
        assert np.allclose(out, cm_future)


class TestISIMIPConstructionValidation(unittest.TestCase):
    def test_distribution_none_and_not_nonparametric_raises(self):
        # __attrs_post_init__ requires a distribution unless nonparametric_qm
        with self.assertRaises(ValueError):
            ISIMIP(
                trend_preservation_method="additive",
                distribution=None,
                nonparametric_qm=False,
                detrending=False,
            )

    def test_distribution_none_with_nonparametric_is_allowed(self):
        debiaser = ISIMIP(
            trend_preservation_method="additive",
            distribution=None,
            nonparametric_qm=True,
            detrending=False,
        )
        assert debiaser.distribution is None
        assert debiaser.nonparametric_qm

    def test_invalid_ecdf_method_rejected(self):
        with self.assertRaises(ValueError):
            _make_isimip("tas", ecdf_method="not_a_method")


if __name__ == "__main__":
    unittest.main()
