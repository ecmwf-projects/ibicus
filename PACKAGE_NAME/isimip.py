import numpy as np
import scipy.stats
import scipy.special

from debiaser import Debiaser
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from math_helpers import IECDF

standard_variables_isimip = {
    "hurs": {
        "lower_bound": 0,
        "lower_threshold": 0.01,
        "upper_bound": 100,
        "upper_threshold": 99.99,
        "distribution": scipy.stats.beta,
        "trend_preservation": "bounded",
        "detrending": False,
    },
    "pr": {
        "lower_bound": 0,
        "lower_threshold": 0.1 / 86400,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.gamma,
        "trend_preservation": "mixed",
        "detrending": False,
    },
    "prsnratio": {
        "lower_bound": 0,
        "lower_threshold": 0.0001,
        "upper_bound": 1,
        "upper_threshold": 0.9999,
        "distribution": scipy.stats.beta,
        "trend_preservation": "bounded",
        "detrending": False,
    },
    "psl": {
        "lower_bound": -np.inf,
        "lower_threshold": -np.inf,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.norm,
        "trend_preservation": "additive",
        "detrending": True,
    },
    "rsds": {
        "lower_bound": 0,
        "lower_threshold": 0.0001,
        "upper_bound": 1,
        "upper_threshold": 0.9999,
        "distribution": scipy.stats.beta,
        "trend_preservation": "bounded",
        "detrending": False,
    },
    "sfcWind": {
        "lower_bound": 0,
        "lower_threshold": 0.01,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.exponweib,  # TODO: needs to be real weibull (log of exponweib)
        "trend_preservation": "mixed",
        "detrending": False,
    },
    "tas": {
        "lower_bound": -np.inf,
        "lower_threshold": -np.inf,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.norm,
        "trend_preservation": "additive",
        "detrending": True,
        "reasonable_physical_range": [0, 400],  # TODO: needs to appear everywhere
    },
    "tasrange": {
        "lower_bound": 0,
        "lower_threshold": 0.01,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.rice,
        "trend_preservation": "mixed",
        "detrending": False,
    },
    "tasskew": {
        "lower_bound": 0,
        "lower_threshold": 0.0001,
        "upper_bound": 1,
        "upper_threshold": 0.9999,
        "distribution": scipy.stats.beta,
        "trend_preservation": "bounded",
        "detrending": False,
    },
}


# Reference TODO
class ISIMIP(Debiaser):
    def __init__(self, variable):
        if not variable in standard_variables_isimip.keys():
            raise ValueError(
                "variable needs to be one of %s" % standard_variables_isimip.keys()
            )

        # Main properties:
        self.variable = variable
        self.lower_bound = standard_variables_isimip.get(variable).get("lower_bound")
        self.lower_threshold = standard_variables_isimip.get(variable).get(
            "lower_threshold"
        )
        self.upper_bound = standard_variables_isimip.get(variable).get("upper_bound")
        self.upper_threshold = standard_variables_isimip.get(variable).get(
            "upper_threshold"
        )
        self.distribution = standard_variables_isimip.get(variable).get("distribution")
        self.trend_preservation = standard_variables_isimip.get(variable).get(
            "trend_preservation"
        )
        self.detrending = standard_variables_isimip.get(variable).get("detrending")

        # TODO: needs to be fully integrated
        self.reasonable_physical_range = standard_variables_isimip.get(variable).get(
            "reasonable_physical_range"
        )

        if (self.upper_bound < np.inf and self.upper_threshold < np.inf) or (
            self.lower_bound > -np.inf and self.lower_threshold > -np.inf
        ):
            self.powerlaw_exponent_step4 = 2

    @staticmethod
    def get_rolling_mean(x, n=365):
        ret = np.cumsum(x, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    @staticmethod
    def get_chunked_mean(x, n):
        ids = np.arange(len(x)) // n
        return np.bincount(ids, x) / np.bincount(ids)

    @staticmethod
    def apply_cdf_thresholding(cdf, cdf_threshold=0.0001):
        return np.maximum(np.minimum(cdf, 1 - cdf_threshold), cdf_threshold)

    @staticmethod
    def interp_cdf_values_on_len_cm_future(cdf_vals, len_cm_future):
        return np.interp(
            np.linspace(1, len(cdf_vals), len_cm_future),
            np.linspace(1, len(cdf_vals), len(cdf_vals)),
            cdf_vals,
        )

    def step1(self, obs, cm_hist, cm_future):
        scale = None
        if variable == "rsds":
            # TODO
            pass

        return obs, cm_hist, cm_future, scale

    # TODO: make work with mask instead of nan
    def step2(self, obs, cm_hist, cm_future):
        if self.variable == "prsnratio":
            # Compute iecdfs to get values related to percentiles
            iecdf_obs = IECDF(obs)
            iecdf_cm_hist = IECDF(cm_hist)
            iecdf_cm_future = IECDF(cm_future)

            obs_hist = np.where(
                np.isnan(obs),
                iecdf_obs(np.random.uniform(size=len(obs))),
                obs,
            )
            cm_hist = np.where(
                np.isnan(cm_hist),
                iecdf_cm_hist(np.random.uniform(size=len(cm_hist))),
                cm_hist,
            )
            cm_future = np.where(
                np.isnan(cm_future),
                iecdf_cm_future(np.random.uniform(size=len(cm_future))),
                cm_future,
            )

        return obs_hist, cm_hist, cm_future

    @staticmethod
    def step3_remove_trend(x):
        # TODO: if we regress on annual means then why not with a rolling window. Only question: how to fill?
        # annual_means = np.zeros_like(x); annual_means[0:len(x)-365] = rolling_mean(x); annual_means[len(x)-365:] = ann annual_means[]
        annual_means = ISIMIP.get_chunked_mean(obs_hist, 356)
        years = list(range(1, len(annual_means)))
        # TODO: is that the correct regression?
        r = scipy.stats.linregress(years, annual_means)
        # TODO: do we really wanna do these significance-tests
        if r.pvalue < 0.05:  # detrend preserving multi-year mean value
            trend = r.slope * (years - np.mean(unique_years))
        else:  # do not detrend because trend is insignificant
            trend = np.zeros(years.size, dtype=x.dtype)
        trend_daily_resolution = np.repeat(trend, 365)

    # TODO: we need to know resolution for that
    def step3(self, obs, cm_hist, cm_future):
        trend = None
        if variable in ["psl", "rlds", "tas"]:
            pass

        return obs, cm_hist, cm_future, trend

    def step4(self, obs, cm_hist, cm_future):
        if self.lower_bound > -np.inf and self.lower_threshold > -np.inf:
            # TODO: Is this how to construct a power law that is increasing towards the left bound? Also which power?
            cm_future = np.where(
                cm_future <= self.lower_threshold,
                self.lower_bound
                + (1 - np.random.power(a=self.powerlaw_exponent_step4))
                / (self.lower_threshold - self.lower_bound),
                cm_future,
            )
        if self.upper_bound < np.inf and self.upper_threshold < np.inf:
            cm_future = np.where(
                cm_future >= self.upper_threshold,
                self.upper_threshold
                + np.random.power(a=self.powerlaw_exponent_step4)
                / (self.upper_bound - self.upper_threshold),
                cm_future,
            )

        return obs, cm_hist, cm_future

    # Generate pseudo-future observations. Here x = x_obs_hist = obs_hist
    def step5(self, obs_hist, cm_hist, cm_future):

        # Compute p = F_obs_hist(x) with x in obs_hist
        ecdf_obs_hist = ECDF(obs_hist)
        p = ecdf_obs_hist(obs_hist)

        # Compute iecdfs
        iecdf_obs_hist = IECDF(obs_hist)
        iecdf_cm_hist = IECDF(cm_hist)
        iecdf_cm_future = IECDF(cm_future)

        # Compute q-vals
        q_obs_hist = obs_hist  # = iecdf_obs_hist(p), appears in eq. 7
        q_cm_future = iecdf_cm_future(p)
        q_cm_hist = iecdf_cm_hist(p)

        if self.trend_preservation == "additive":
            delta_add = q_cm_future - q_cm_hist
            return obs_hist + delta_add
        elif self.trend_preservation == "multiplicative":
            delta_star_mult = np.where(q_cm_hist == 0, 1, q_cm_future / q_cm_hist)
            delta_mult = np.maximum(0.01, np.minimum(100, delta_star_mult))
            return obs_hist * delta_mult
        elif self.trend_preservation == "mixed":
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
        elif self.trend_preservation == "bounded":
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
                "Wrong value for self.trend_preservation. Needs to be one of ['additive', 'multiplicative', 'mixed', 'bounded']"
            )

    def _step6_values_between_thresholds(
        self, obs_hist_sorted, obs_future_sorted, cm_hist_sorted, cm_future_sorted
    ):

        # Calculate cdf-fits
        fit_cm_future = self.distribution.fit(cm_future_sorted)

        fit_obs_hist = self.distribution.fit(obs_hist_sorted)
        fit_obs_future = self.distribution.fit(obs_future_sorted)
        fit_cm_hist = self.distribution.fit(cm_hist_sorted)

        # Get the cdf-vals and interpolate if there are unequal sample sizes (following Switanek 2017):
        cdf_vals_cm_future = ISIMIP.apply_cdf_thresholding(
            self.distribution.cdf(cm_future_sorted, *fit_obs_hist)
        )

        cdf_vals_obs_hist = ISIMIP.interp_cdf_values_on_len_cm_future(
            ISIMIP.apply_cdf_thresholding(
                self.distribution.cdf(obs_hist_sorted, *fit_obs_hist)
            ),
            len(cdf_vals_cm_future),
        )
        cdf_vals_cm_hist = ISIMIP.interp_cdf_values_on_len_cm_future(
            ISIMIP.apply_cdf_thresholding(
                self.distribution.cdf(cm_hist_sorted, *fit_cm_hist)
            ),
            len(cdf_vals_cm_future),
        )

        # tas exception
        if self.variable == "tas":
            mapped_vals = self.distribution.ppf(cdf_vals_cm_future, *fit_obs_future)
            return mapped_vals

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

    # Core of the isimip-method: parametric quantile mapping
    def step6(self, obs_hist, obs_future, cm_hist, cm_future):

        # Sort arrays to apply parametric quantile mapping (values of equal rank are mapped together). Save sort-order of cm_future for backsorting
        cm_future_argsort = np.argsort(cm_future)
        cm_future_sorted = cm_future[cm_future_argsort]

        obs_hist_sorted = np.sort(obs_hist)
        obs_future_sorted = np.sort(obs_future)
        cm_hist_sorted = np.sort(cm_hist)

        # TODO: change lower and upper threshold to none is not given. Then precompute indices below
        if self.lower_threshold > -np.inf or self.upper_threshold < np.inf:

            def get_P_obs_future(P_obs_hist, P_cm_hist, P_cm_future):
                if np.isclose(P_cm_hist, P_cm_future):
                    return P_obs_hist
                elif P_cm_hist > P_cm_future:
                    return P_obs_hist * P_cm_future / P_cm_hist
                else:
                    return 1 - (1 - P_obs_hist) * (1 - P_cm_future) / (1 - P_cm_hist)

            # TODO: Optimize via non-recomputing indices
            nr_indices_to_set_to_lower_bound = 0
            if self.lower_threshold > -np.inf:
                P_obs_hist = (obs_hist_sorted < self.lower_threshold).sum() / len(
                    obs_hist_sorted
                )
                P_cm_hist = (cm_hist_sorted < self.lower_threshold).sum() / len(
                    cm_hist_sorted
                )
                P_cm_future = (cm_future_sorted < self.lower_threshold).sum() / len(
                    cm_future_sorted
                )

                P_obs_future = get_P_obs_future(P_obs_hist, P_cm_hist, P_cm_future)

                # TODO: round (3.5 --> 4) returns other value than np.rint (3.5 --> 3)
                nr_indices_to_set_to_lower_bound = round(
                    len(cm_future_sorted) * P_obs_future
                )

            nr_indices_to_set_to_higher_bound = 0
            if self.upper_threshold < np.inf:
                P_obs_hist = (obs_hist_sorted > self.upper_threshold).sum() / len(
                    obs_hist_sorted
                )
                P_cm_hist = (cm_hist_sorted > self.upper_threshold).sum() / len(
                    cm_hist_sorted
                )
                P_cm_future = (cm_future_sorted > self.upper_threshold).sum() / len(
                    cm_future_sorted
                )

                P_obs_future = get_P_obs_future(P_obs_hist, P_cm_hist, P_cm_future)

                # TODO: round (3.5 --> 4) returns other value than np.rint (3.5 --> 3)
                nr_indices_to_set_to_higher_bound = round(
                    len(cm_future_sorted) * P_obs_future
                )

            cm_future_sorted[0:nr_indices_to_set_to_lower_bound] = self.lower_bound
            cm_future_sorted[
                len(cm_future_sorted) - nr_indices_to_set_to_higher_bound :
            ] = self.upper_bound
            cm_future_sorted[
                nr_indices_to_set_to_lower_bound : (
                    len(cm_future_sorted) - nr_indices_to_set_to_higher_bound
                )
            ] = self._step6_values_between_thresholds(
                obs_hist_sorted[
                    (obs_hist_sorted > self.lower_threshold)
                    & (obs_hist_sorted < self.upper_threshold)
                ],
                obs_future_sorted[
                    (obs_future_sorted > self.lower_threshold)
                    & (obs_future_sorted < self.upper_threshold)
                ],
                cm_hist_sorted[
                    (cm_hist_sorted > self.lower_threshold)
                    & (cm_hist_sorted < self.upper_threshold)
                ],
                cm_future_sorted[
                    nr_indices_to_set_to_lower_bound : (
                        len(cm_future_sorted) - nr_indices_to_set_to_higher_bound
                    )
                ],
            )

            reverse_sorting_idx = np.argsort(cm_future_argsort)
            return cm_future_sorted[reverse_sorting_idx]
        else:
            # Return values inserted back at correct locations
            reverse_sorting_idx = np.argsort(cm_future_argsort)
            cm_future_sorted = self._step6_values_between_thresholds(
                obs_hist_sorted, obs_future_sorted, cm_hist_sorted, cm_future_sorted
            )
            return cm_future_sorted[reverse_sorting_idx]

    def step7(cm_future, trend):
        if variable in ["psl", "rlds", "tas"]:
            pass
        return cm_future

    def step8(cm_future, scale):
        if variable == "rsds":
            pass
        return cm_future

    def apply_location(self, obs, cm_hist, cm_future):
        print(self.reasonable_physical_range)
        if self.reasonable_physical_range is not None:
            if np.any(
                (obs < self.reasonable_physical_range[0])
                | (obs > self.reasonable_physical_range[1])
            ):
                raise ValueError(
                    "Values of obs lie outside the reasonable physical range of %s"
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

        # Steps
        obs, cm_hist, cm_future, scale = self.step1(obs, cm_hist, cm_future)
        obs, cm_hist, cm_future = self.step2(obs, cm_hist, cm_future)
        obs, cm_hist, cm_future, trend = self.step3(obs, cm_hist, cm_future)
        obs, cm_hist, cm_future = self.step4(obs, cm_hist, cm_future)
        obs_future = self.step5(obs, cm_hist, cm_future)
        cm_future = self.step6(obs, obs_future, cm_hist, cm_future)
        cm_future = self.step7(cm_future, trend)
        cm_future = self.step8(cm_future, scale)

        return cm_future
