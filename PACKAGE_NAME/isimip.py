from dataclasses import dataclass
from typing import Optional, Union

import attrs
import numpy as np
import scipy.special
import scipy.stats

from .debiaser import Debiaser
from .isimip_options import isimip_2_5, standard_variables_isimip
from .math_helpers import ecdf, iecdf
from .utils import (
    get_chunked_mean,
    interp_sorted_cdf_vals_on_given_length,
    threshold_cdf_vals,
)


# Reference TODO
@dataclass
class ISIMIP(Debiaser):
    # Variables
    distribution: scipy.stats.rv_continuous
    trend_preservation_method: str = attrs.field(
        validator=attrs.validators.in_(
            ["additive", "multiplicative", "mixed", "bounded"]
        )
    )
    detrending: bool = attrs.field(validator=attrs.validators.instance_of(bool))
    reasonable_physical_range: Optional[list] = attrs.field(default=None)

    @reasonable_physical_range.validator
    def validate_reasonable_physical_range(x):
        if len(x) != 2:
            raise ValueError(
                "reasonable_physical_range should have only a lower and upper physical range"
            )
        if not all(isinstance(elem, int) for elem in x):
            raise ValueError("reasonable_physical_range needs to be a list of floats")
        if not x[0] < x[1]:
            raise ValueError(
                "lower bounds needs to be smaller than upper bound in reasonable_physical_range"
            )

    variable: str = attrs.field(default="unknown", eq=False)

    # Bounds
    lower_bound: float = attrs.field(
        default=np.inf, validator=attrs.validators.instance_of(float)
    )
    lower_threshold: float = attrs.field(
        default=np.inf, validator=attrs.validators.instance_of(float)
    )
    upper_bound: float = attrs.field(
        default=-np.inf, validator=attrs.validators.instance_of(float)
    )
    upper_threshold: float = attrs.field(
        default=-np.inf, validator=attrs.validators.instance_of(float)
    )

    # ISIMIP behavior
    trend_removal_with_significance_test: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )  # step 3
    trend_transfer_only_for_values_within_threshold: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )  # step 5
    event_likelihood_adjustment: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )  # step 6

    # math functions
    ecdf_method: str = attrs.field(
        default="step_function",
        validator=attrs.validators.in_(
            [
                "inverted_cdf",
                "averaged_inverted_cdf",
                "closest_observation",
                "interpolated_inverted_cdf",
                "hazen",
                "weibull",
                "linear",
                "median_unbiased",
                "normal_unbiased",
            ]
        ),
    )
    iecdf_method: str = attrs.field(
        default="inverted_cdf",
        validator=attrs.validators.in_(
            ["kernel_density", "linear_interpolation", "step_function"]
        ),
    )

    # iteration
    running_window_mode: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )
    running_window_length: int = attrs.field(
        default=31,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )
    running_window_step_length: int = attrs.field(
        default=1, validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)]
    )

    @classmethod
    def from_variable(cls, variable):
        if variable not in standard_variables_isimip.keys():
            raise ValueError(
                "variable needs to be one of %s" % standard_variables_isimip.keys()
            )
        isimip_instance = cls(
            variable=variable,
            **isimip_2_5.get("variables").get(variable),
            **isimip_2_5.get("isimip_run")
        )
        return isimip_instance

    @property
    def has_lower_threshold(self):
        if self.lower_threshold is not None and self.lower_threshold > -np.inf:
            return True
        else:
            return False

    @property
    def has_lower_bound(self):
        if self.lower_bound is not None and self.lower_bound > -np.inf:
            return True
        else:
            return False

    @property
    def has_upper_threshold(self):
        if self.upper_threshold is not None and self.upper_threshold < np.inf:
            return True
        else:
            return False

    @property
    def has_upper_bound(self):
        if self.upper_bound is not None and self.upper_bound < np.inf:
            return True
        else:
            return False

    # ----- Non public helpers: General ----- #

    def _get_mask_for_observations_beyond_lower_threshold(self, x):
        return x < self.lower_threshold

    def _get_mask_for_observations_beyond_upper_threshold(self, x):
        return x > self.upper_threshold

    def _get_mask_for_observations_between_thresholds(self, x):
        return (x > self.lower_threshold) & (x < self.upper_threshold)

    def _get_observations_between_thresholds(self, x):
        return x[self._get_mask_for_observations_between_thresholds(x)]

    def _check_reasonable_physical_range(self, obs_hist, cm_hist, cm_future):
        if self.reasonable_physical_range is not None:
            if np.any(
                (obs_hist < self.reasonable_physical_range[0])
                | (obs_hist > self.reasonable_physical_range[1])
            ):
                raise ValueError(
                    "Values of obs_hist lie outside the reasonable physical range of %s"
                    % self.reasonable_physical_range
                )

            if np.any(
                (cm_hist < self.reasonable_physical_range[0])
                | (cm_hist > self.reasonable_physical_range[1])
            ):
                raise ValueError(
                    "Values of cm_hist lie outside the reasonable physical range of %s"
                    % self.reasonable_physical_range
                )

            if np.any(
                (cm_future < self.reasonable_physical_range[0])
                | (cm_future > self.reasonable_physical_range[1])
            ):
                raise ValueError(
                    "Values of cm_future lie outside the reasonable physical range of %s"
                    % self.reasonable_physical_range
                )

    # ----- Non public helpers: ISIMIP-steps ----- #

    def _step3_remove_trend(self, x):
        annual_means = get_chunked_mean(x, self.running_window_length)
        years = np.arange(annual_means.size)
        regression = scipy.stats.linregress(years, annual_means)
        if regression.pvalue < 0.05 and self.trend_removal_with_significance_test:
            annual_trend = regression.slope * (years - np.mean(years))
        else:
            annual_trend = np.zeros(years.size, dtype=x.dtype)
        trend = np.repeat(annual_trend, self.running_window_length)[0 : x.size]
        x = x - trend
        return x, trend

    def _step6_calculate_percent_values_beyond_threshold(
        mask_for_observations_beyond_threshold,
    ):
        return (
            mask_for_observations_beyond_threshold.sum()
            / mask_for_observations_beyond_threshold.size
        )

    # TODO: change equation to v2.3 one or add option
    @staticmethod
    def _step6_get_P_obs_future(P_obs_hist, P_cm_hist, P_cm_future):
        if np.isclose(P_cm_hist, P_cm_future):
            return P_obs_hist
        elif P_cm_hist > P_cm_future:
            return P_obs_hist * P_cm_future / P_cm_hist
        else:
            return 1 - (1 - P_obs_hist) * (1 - P_cm_future) / (1 - P_cm_hist)

    def _step6_get_nr_of_entries_to_set_to_bound(
        self,
        mask_for_observations_beyond_threshold_obs_hist_sorted,
        mask_for_observations_beyond_threshold_cm_hist_sorted,
        mask_for_observations_beyond_threshold_cm_future_sorted,
    ):

        P_obs_hist = ISIMIP._step6_calculate_percent_values_beyond_threshold(
            mask_for_observations_beyond_threshold_obs_hist_sorted
        )
        P_cm_hist = ISIMIP._step6_calculate_percent_values_beyond_threshold(
            mask_for_observations_beyond_threshold_cm_hist_sorted
        )
        P_cm_future = ISIMIP._step6_calculate_percent_values_beyond_threshold(
            mask_for_observations_beyond_threshold_cm_future_sorted
        )

        P_obs_future = ISIMIP._step6_get_P_obs_future(
            P_obs_hist, P_cm_hist, P_cm_future
        )

        return round(
            mask_for_observations_beyond_threshold_cm_future_sorted.size * P_obs_future
        )

    @staticmethod
    def _step6_transform_nr_of_entries_to_set_to_lower_bound_to_mask_for_cm_future(
        nr, cm_future_sorted
    ):
        mask = np.zeros_like(cm_future_sorted, dtype=bool)
        mask[0:nr] = True
        return mask

    def _step6_get_mask_for_entries_to_set_to_lower_bound(
        self, obs_hist_sorted, cm_hist_sorted, cm_future_sorted
    ):

        nr_of_entries_to_set_to_lower_bound = (
            self._step6_get_nr_of_entries_to_set_to_bound(
                self._get_mask_for_observations_beyond_lower_threshold(obs_hist_sorted),
                self._get_mask_for_observations_beyond_lower_threshold(cm_hist_sorted),
                self._get_mask_for_observations_beyond_lower_threshold(
                    cm_future_sorted
                ),
            )
        )

        mask_for_entries_to_set_to_lower_bound = ISIMIP._step6_transform_nr_of_entries_to_set_to_lower_bound_to_mask_for_cm_future(
            nr_of_entries_to_set_to_lower_bound, cm_future_sorted
        )
        return mask_for_entries_to_set_to_lower_bound

    @staticmethod
    def _step6_transform_nr_of_entries_to_set_to_upper_bound_to_mask_for_cm_future(
        nr, cm_future_sorted
    ):
        mask = np.zeros_like(cm_future_sorted, dtype=bool)
        mask[(cm_future_sorted.size - nr) :] = True
        return mask

    def _step6_get_mask_for_entries_to_set_to_upper_bound(
        self, obs_hist_sorted, cm_hist_sorted, cm_future_sorted
    ):

        nr_of_entries_to_set_to_upper_bound = (
            self._step6_get_nr_of_entries_to_set_to_bound(
                self._get_mask_for_observations_beyond_upper_threshold(obs_hist_sorted),
                self._get_mask_for_observations_beyond_upper_threshold(cm_hist_sorted),
                self._get_mask_for_observations_beyond_upper_threshold(
                    cm_future_sorted
                ),
            )
        )

        mask_for_entries_to_set_to_upper_bound = ISIMIP._step6_transform_nr_of_entries_to_set_to_upper_bound_to_mask_for_cm_future(
            nr_of_entries_to_set_to_upper_bound, cm_future_sorted
        )
        return mask_for_entries_to_set_to_upper_bound

    def _step6_adjust_values_between_thresholds(
        self, obs_hist_sorted, obs_future_sorted, cm_hist_sorted, cm_future_sorted
    ):

        # Calculate cdf-fits
        fit_cm_future = self.distribution.fit(cm_future_sorted)

        fit_obs_hist = self.distribution.fit(obs_hist_sorted)
        fit_obs_future = self.distribution.fit(obs_future_sorted)
        fit_cm_hist = self.distribution.fit(cm_hist_sorted)

        # Get the cdf-vals and interpolate if there are unequal sample sizes (following Switanek 2017):
        cdf_vals_cm_future = threshold_cdf_vals(
            self.distribution.cdf(cm_future_sorted, *fit_cm_future)
        )

        cdf_vals_obs_hist = interp_sorted_cdf_vals_on_given_length(
            threshold_cdf_vals(self.distribution.cdf(obs_hist_sorted, *fit_obs_hist)),
            cdf_vals_cm_future.size,
        )
        cdf_vals_cm_hist = interp_sorted_cdf_vals_on_given_length(
            threshold_cdf_vals(self.distribution.cdf(cm_hist_sorted, *fit_cm_hist)),
            cdf_vals_cm_future.size,
        )

        # Event likelihood adjustment only happens in certain cases
        if not self.event_likelihood_adjustment:
            mapped_vals = self.distribution.ppf(cdf_vals_cm_future, *fit_obs_future)
        else:
            # Calculate L-values and delta log-odds for mapping, following formula 11-14
            L_obs_hist = scipy.special.logit(cdf_vals_obs_hist)
            L_cm_hist = scipy.special.logit(cdf_vals_cm_hist)
            L_cm_future = scipy.special.logit(cdf_vals_cm_future)

            delta_log_odds = np.maximum(
                -np.log(10), np.minimum(np.log(10), L_cm_future - L_cm_hist)
            )

            # Map values following formula 10
            mapped_vals = self.distribution.ppf(
                scipy.special.expit(L_obs_hist + delta_log_odds), *fit_obs_future
            )
        return mapped_vals

    # ----- ISIMIP calculation steps ----- #
    def step1(self, obs_hist, cm_hist, cm_future):
        scale = None
        if self.variable == "rsds":
            # TODO
            pass

        return obs_hist, cm_hist, cm_future, scale

    # TODO: make work with mask instead of nan
    def step2(self, obs_hist, cm_hist, cm_future):
        if self.variable == "prsnratio":
            obs_hist = np.where(
                np.isnan(obs_hist),
                iecdf(x=obs_hist, p=np.random.uniform(size=obs_hist.size)),
                obs_hist,
            )
            cm_hist = np.where(
                np.isnan(cm_hist),
                iecdf(x=cm_hist, p=np.random.uniform(size=cm_hist.size)),
                cm_hist,
            )
            cm_future = np.where(
                np.isnan(cm_future),
                iecdf(x=cm_future, p=np.random.uniform(size=cm_future.size)),
                cm_future,
            )

        return obs_hist, cm_hist, cm_future

    def step3(self, obs_hist, cm_hist, cm_future):
        trend_cm_future = np.zeros_like(cm_future)
        if self.detrending:
            obs_hist, _ = self._step3_remove_trend(obs_hist)
            cm_hist, _ = self._step3_remove_trend(cm_hist)
            cm_future, trend_cm_future = self._step3_remove_trend(cm_future)
        return obs_hist, cm_hist, cm_future, trend_cm_future

    # TODO: v2.5 has randomised values sorted like original ones.
    def step4(self, obs_hist, cm_hist, cm_future):
        if self.has_lower_bound and self.has_lower_threshold:
            # TODO: Is this how to construct a power law that is increasing towards the left bound? Also which power?
            cm_future = np.where(
                cm_future <= self.lower_threshold,
                self.lower_bound
                + (
                    1 - np.random.power(a=1)
                )  # Possible: powerlaw_exponent_step4 as attribute
                / (self.lower_threshold - self.lower_bound),
                cm_future,
            )
        if self.has_upper_bound and self.has_upper_threshold:
            cm_future = np.where(
                cm_future >= self.upper_threshold,
                self.upper_threshold
                + np.random.power(a=1)  # Possible: powerlaw_exponent_step4 as attribute
                / (self.upper_bound - self.upper_threshold),
                cm_future,
            )

        return obs_hist, cm_hist, cm_future

    # Generate pseudo-future observations. Here x = x_obs_hist = obs_hist
    # TODO: add option to do trend transfer only for values between thresholds.
    def step5(self, obs_hist, cm_hist, cm_future):

        # Compute p = F_obs_hist(x) with x in obs_hist
        p = ecdf(obs_hist, obs_hist, method=self.ecdf_method)

        # Compute q-vals: q = IECDF(p)
        q_obs_hist = obs_hist  # TODO: = iecdf_obs_hist(p), appears in eq. 7
        q_cm_future = iecdf(cm_future, p, method=self.iecdf_method)
        q_cm_hist = iecdf(cm_hist, p, method=self.iecdf_method)

        if self.trend_preservation_method == "additive":
            delta_add = q_cm_future - q_cm_hist
            return obs_hist + delta_add
        elif self.trend_preservation_method == "multiplicative":
            delta_star_mult = np.where(q_cm_hist == 0, 1, q_cm_future / q_cm_hist)
            delta_mult = np.maximum(0.01, np.minimum(100, delta_star_mult))
            return obs_hist * delta_mult
        elif self.trend_preservation_method == "mixed":
            # Formula 7
            condition1 = q_cm_hist >= q_obs_hist
            condition2 = (q_cm_hist < q_obs_hist) & (q_obs_hist < 9 * q_cm_hist)

            gamma = np.zeros(len(obs_hist))
            gamma[condition1] = 1
            gamma[condition2] = 0.5 * (
                1
                + np.cos(q_obs_hist[condition2] / q_cm_hist[condition2] - 1) * np.pi / 8
            )

            # Formula 6
            delta_add = q_cm_future - q_cm_hist
            delta_star_mult = np.where(q_cm_hist == 0, 1, q_cm_future / q_cm_hist)
            delta_mult = np.maximum(0.01, np.minimum(100, delta_star_mult))
            return gamma * obs_hist * delta_mult + (1 - gamma) * (obs_hist + delta_add)
        elif self.trend_preservation_method == "bounded":
            a = self.lower_bound
            b = self.upper_bound

            # Formula 8
            condition1 = q_cm_hist > q_cm_future
            condition2 = np.isclose(q_cm_hist, q_cm_future)

            return_vals = b - (b - obs_hist) * (b - q_cm_future) / (b - q_cm_hist)
            return_vals[condition1] = a + (obs_hist[condition1] - a) * (
                q_cm_future[condition1] - a
            ) / (q_cm_hist[condition1] - a)
            return_vals[condition2] = obs_hist[condition2]
            return return_vals
        else:
            raise ValueError(
                """ Wrong value for self.trend_preservation_method.
                    Needs to be one of ['additive', 'multiplicative', 'mixed', 'bounded'] """
            )

    # Core of the isimip-method: parametric quantile mapping
    def step6(self, obs_hist, obs_future, cm_hist, cm_future):

        # Sort arrays to apply parametric quantile mapping (values of equal rank are mapped together).
        # Save sort-order of cm_future for backsorting
        cm_future_argsort = np.argsort(cm_future)
        cm_future_sorted = cm_future[cm_future_argsort]

        obs_hist_sorted = np.sort(obs_hist)
        obs_future_sorted = np.sort(obs_future)
        cm_hist_sorted = np.sort(cm_hist)

        # Calculate values that are set to lower bound for bounded/thresholded variables
        mask_for_entries_to_set_to_lower_bound = np.zeros_like(
            cm_future_sorted, dtype=bool
        )
        if self.has_lower_threshold:
            mask_for_entries_to_set_to_lower_bound = (
                self._step6_get_mask_for_entries_to_set_to_lower_bound(
                    obs_hist_sorted, cm_hist_sorted, cm_future_sorted
                )
            )

        # Calculate values that are set to lower bound for bounded/thresholded variables
        mask_for_entries_to_set_to_upper_bound = np.zeros_like(
            cm_future_sorted, dtype=bool
        )
        if self.has_upper_threshold:
            mask_for_entries_to_set_to_upper_bound = (
                self._step6_get_mask_for_entries_to_set_to_upper_bound(
                    obs_hist_sorted, cm_hist_sorted, cm_future_sorted
                )
            )

        # Set values to upper or lower bound for bounded/thresholded variables
        cm_future_sorted[mask_for_entries_to_set_to_lower_bound] = self.lower_bound
        cm_future_sorted[mask_for_entries_to_set_to_upper_bound] = self.upper_bound
        mask_for_entries_not_set_to_either_bound = np.logical_and(
            np.logical_not(mask_for_entries_to_set_to_lower_bound),
            np.logical_not(mask_for_entries_to_set_to_upper_bound),
        )

        # Calculate values between bounds
        cm_future_sorted[
            mask_for_entries_not_set_to_either_bound
        ] = self._step6_adjust_values_between_thresholds(
            self._get_observations_between_thresholds(obs_hist_sorted),
            self._get_observations_between_thresholds(obs_future_sorted),
            self._get_observations_between_thresholds(cm_hist_sorted),
            cm_future_sorted[mask_for_entries_not_set_to_either_bound],
        )

        # Return values inserted back at correct locations
        reverse_sorting_idx = np.argsort(cm_future_argsort)
        return cm_future_sorted[reverse_sorting_idx]

    def step7(self, cm_future, trend_cm_future):
        if self.detrending:
            cm_future = cm_future + trend_cm_future
        return cm_future

    def step8(self, cm_future, scale):
        if self.variable == "rsds":
            pass
        return cm_future

    # ----- Apply location function -----
    def _apply_on_window(self, obs_hist, cm_hist, cm_future):
        # Steps
        obs_hist, cm_hist, cm_future, scale = self.step1(obs_hist, cm_hist, cm_future)
        obs_hist, cm_hist, cm_future = self.step2(obs_hist, cm_hist, cm_future)
        obs_hist, cm_hist, cm_future, trend_cm_future = self.step3(
            obs_hist, cm_hist, cm_future
        )
        obs_hist, cm_hist, cm_future = self.step4(obs_hist, cm_hist, cm_future)
        obs_future = self.step5(obs_hist, cm_hist, cm_future)
        cm_future = self.step6(obs_hist, obs_future, cm_hist, cm_future)
        cm_future = self.step7(cm_future, trend_cm_future)
        cm_future = self.step8(cm_future, scale)

        return cm_future

    def apply_location(self, obs_hist, cm_hist, cm_future, *args, **kwargs):
        self._check_reasonable_physical_range(obs_hist, cm_hist, cm_future)

        # TODO: how do we integrate if day of year or month is passed as an argument
        if self.running_window_mode:
            day_of_year_obs_hist = kwargs.get("day_of_year_obs")
            day_of_year_cm_hist = kwargs.get("day_of_year_cm_hist")
            day_of_year_cm_future = kwargs.get("day_of_yearcm_future")

        if (
            day_of_year_obs_hist is not None
            and day_of_year_cm_hist is not None
            and day_of_year_cm_future is not None
        ):
            pass
            # do stuff
        else:
            # window_center = np.arange(365)[:: self.step_window_step_length]
            return cm_future

        months_obs_hist = kwargs.get("months_obs")
        months_cm_hist = kwargs.get("months_cm_hist")
        months_cm_future = kwargs.get("months_cm_future")

        if (
            months_obs_hist is not None
            and months_cm_hist is not None
            and months_cm_future is not None
        ):
            debiased_cm_future = np.zeros_like(cm_future)
            for month in months_obs_hist:
                idxs_month_in_obs_hist = np.where(months_obs_hist == month)
                idxs_month_in_cm_hist = np.where(months_cm_hist == month)
                idxs_month_in_cm_future = np.where(months_cm_future == month)

                debiased_cm_future[idxs_month_in_cm_future] = self._apply_on_window(
                    obs_hist=obs_hist[idxs_month_in_obs_hist],
                    cm_hist=cm_hist[idxs_month_in_cm_hist],
                    cm_future=cm_future[idxs_month_in_cm_future],
                )
            return debiased_cm_future

        return cm_future
