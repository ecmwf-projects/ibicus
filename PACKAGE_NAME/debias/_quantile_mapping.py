# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Union

import attrs
import numpy as np
import scipy
import scipy.stats

from .._variables import (
    Variable,
    map_standard_precipitation_method,
    map_variable_str_to_variable_class,
)
from ..utils import StatisticalModel
from ._debiaser import Debiaser


@attrs.define
class QuantileMapping(Debiaser):

    delta_type: str = attrs.field(validator=attrs.validators.in_(["additive", "multiplicative", "no_delta"]))
    distribution: Union[
        scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel, None
    ] = attrs.field(
        validator=attrs.validators.instance_of(
            (scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel, None)
        )  # Why none? TODO
    )
    variable: str = attrs.field(default="unknown", eq=False)

    @classmethod
    def from_variable(
        cls,
        variable: Union[str, Variable],
        delta_type: str,
        precipitation_model_type: str = "censored",
        precipitation_amounts_distribution=scipy.stats.gamma,
        precipitation_censoring_value: float = 0.1,
        precipitation_hurdle_model_randomization: bool = True,
    ):
        if isinstance(variable, Variable):
            return cls(
                delta_type=delta_type,
                distribution=variable.method,
                variable=variable.name,
            )
        else:
            variable = map_variable_str_to_variable_class(variable)
            if variable.name == "Precipitation":
                variable.method = map_standard_precipitation_method(
                    precipitation_model_type,
                    precipitation_amounts_distribution,
                    precipitation_censoring_value,
                    precipitation_hurdle_model_randomization,
                )
            return cls(
                delta_type=delta_type,
                distribution=variable.method,
                variable=variable.name,
            )

    def _standard_qm(self, x, fit_cm_hist, fit_obs):
        return self.distribution.ppf(self.distribution.cdf(x, *fit_cm_hist), *fit_obs)

    def apply_location(self, obs, cm_hist, cm_future):
        fit_obs = self.distribution.fit(cm_hist)
        fit_cm_hist = self.distribution.fit(cm_hist)

        if self.delta_type == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return self._standard_qm(cm_future - delta, fit_cm_hist, fit_obs) + delta
        elif self.delta_type == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return self._standard_qm(cm_future / delta, fit_cm_hist, fit_obs) * delta
        elif self.delta_type == "no_delta":
            return self._standard_qm(cm_future, fit_cm_hist, fit_obs)
        else:
            raise ValueError(
                "self.delta_type has wrong value. Needs to be one of ['additive', 'multiplicative', 'no_delta']"
            )
