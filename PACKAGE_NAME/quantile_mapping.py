import warnings
from typing import Optional, Union

import attrs
import numpy as np
import scipy

from .debiaser import Debiaser
from .math_helpers import (
    fit_precipitation_censored_gamma,
    fit_precipitation_hurdle_model,
    quantile_mapping_precipitation_censored_gamma,
    quantile_mapping_precipitation_hurdle_model,
)
from .variable_distribution_match import standard_distributions


@attrs.define
class QuantileMapping(Debiaser):

    delta_type: str = attrs.field(
        validator=attrs.validators.in_(["additive", "multiplicative", "no_delta"])
    )
    distribution: Union[
        scipy.stats.rv_continuous,
        scipy.stats.rv_discrete,
        scipy.stats.rv_histogram,
        None,
    ] = attrs.field(
        validator=attrs.validators.instance_of(
            (
                scipy.stats.rv_continuous,
                scipy.stats.rv_discrete,
                scipy.stats.rv_histogram,
                None,
            )
        )
    )
    variable: str = attrs.field(default="unknown", eq=False)
    precip_model_type: str = attrs.field(
        default="censored", validator=attrs.validators.in_(["censored", "hurdle"])
    )
    precip_hurdle_randomization: bool = False
    precip_censoring_value: float = 0.1

    def __attrs_post_init__(self):
        if (
            self.variable in ["precip", "precipitation"]
            and self.precip_model_type == "censored"
        ):
            warnings.warn(
                "Only the gamma distribution is supported for a censored precipitation model"
            )

    @classmethod
    def from_variable(cls, variable, delta_type):
        if variable not in standard_distributions.keys():
            raise ValueError(
                "variable needs to be one of %s" % standard_distributions.keys()
            )
        return cls(
            delta_type=delta_type,
            distribution=standard_distributions.get(variable),
            variable=variable,
        )

    def apply_location_precip_hurdle(self, obs, cm_hist, cm_future):
        fit_obs = fit_precipitation_hurdle_model(obs, self.distribution)
        fit_cm_hist = fit_precipitation_hurdle_model(cm_hist, self.distribution)

        if self.delta_type == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return (
                quantile_mapping_precipitation_hurdle_model(
                    cm_future - delta,
                    fit_cm_hist,
                    fit_obs,
                    self.distribution,
                    self.distribution,
                    self.precip_hurdle_randomization,
                )
                + delta
            )
        elif self.delta_type == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return (
                quantile_mapping_precipitation_hurdle_model(
                    cm_future / delta,
                    fit_cm_hist,
                    fit_obs,
                    self.distribution,
                    self.distribution,
                    self.precip_hurdle_randomization,
                )
                * delta
            )
        elif self.delta_type == "no_delta":
            return quantile_mapping_precipitation_hurdle_model(
                cm_future,
                fit_cm_hist,
                fit_obs,
                self.distribution,
                self.distribution,
                self.precip_hurdle_randomization,
            )
        else:
            raise ValueError(
                "self.delta_type has wrong value. Needs to be one of ['additive', 'multiplicative', 'no_delta']"
            )

    def apply_location_precip_censored(self, obs, cm_hist, cm_future):
        fit_obs = fit_precipitation_censored_gamma(obs, self.precip_censoring_value)
        fit_cm_hist = fit_precipitation_censored_gamma(
            cm_hist, self.precip_censoring_value
        )

        if self.delta_type == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return (
                quantile_mapping_precipitation_censored_gamma(
                    cm_future - delta, self.precip_censoring_value, fit_cm_hist, fit_obs
                )
                + delta
            )
        elif self.delta_type == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return (
                quantile_mapping_precipitation_censored_gamma(
                    cm_future / delta, self.precip_censoring_value, fit_cm_hist, fit_obs
                )
                * delta
            )
        elif self.delta_type == "no_delta":
            return quantile_mapping_precipitation_censored_gamma(
                cm_future, self.precip_censoring_value, fit_cm_hist, fit_obs
            )
        else:
            raise ValueError(
                "self.delta_type has wrong value. Needs to be one of ['additive', 'multiplicative', 'no_delta']"
            )

    def standard_qm(self, x, fit_cm_hist, fit_obs):
        return self.distribution.ppf(self.distribution.cdf(x, *fit_cm_hist), *fit_obs)

    def apply_location_standard(self, obs, cm_hist, cm_future):
        fit_obs = self.distribution.fit(cm_hist)
        fit_cm_hist = self.distribution.fit(cm_hist)

        if self.delta_type == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return self.standard_qm(cm_future - delta, fit_cm_hist, fit_obs) + delta
        elif self.delta_type == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return self.standard_qm(cm_future / delta, fit_cm_hist, fit_obs) * delta
        elif self.delta_type == "no_delta":
            return self.standard_qm(cm_future, fit_cm_hist, fit_obs)
        else:
            raise ValueError(
                "self.delta_type has wrong value. Needs to be one of ['additive', 'multiplicative', 'no_delta']"
            )

    def apply_location(self, obs, cm_hist, cm_future):
        if self.variable in ["precipitation", "precip"]:
            if self.precip_model_type == "censored":
                return self.apply_location_precip_censored(obs, cm_hist, cm_future)
            elif self.precip_model_type == "hurdle":
                return self.apply_location_precip_censored(obs, cm_hist, cm_future)
            else:
                raise ValueError(
                    "Invalid self.precip_model_type. Needs to be one of ['censored', 'hurdle']"
                )
        else:
            return self.apply_location_standard(obs, cm_hist, cm_future)

    """
    def cache_location(self, obs, cm_hist, cm_future):
        fit_obs = self.distribution.fit(cm_hist)
        fit_cm_hist = self.distribution.fit(cm_future)
        return lambda obs, cm_hist, cm_future: self.distribution.ppf(
            self.distribution.cdf(cm_future, *fit_cm_hist), *fit_obs
        )
    """
