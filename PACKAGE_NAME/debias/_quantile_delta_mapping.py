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

from ..utils import (
    StatisticalModel,
    ecdf,
    gen_PrecipitationGammaLeftCensoredModel,
    threshold_cdf_vals,
)
from ..variables import (
    Precipitation,
    Temperature,
    Variable,
    map_standard_precipitation_method,
    map_variable_str_to_variable_class,
)
from ._debiaser import Debiaser

default_settings = {
    Temperature: {"distribution": scipy.stats.norm},
    Precipitation: {"distribution": gen_PrecipitationGammaLeftCensoredModel(censoring_value=0.05)},
}

# Reference Cannon et al. 2015
@attrs.define
class QuantileDeltaMapping(Debiaser):

    distribution: Union[
        scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel
    ] = attrs.field(
        validator=attrs.validators.instance_of(
            (scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel)
        )
    )
    time_window_length: int = attrs.field(validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)])
    variable: str = attrs.field(default="unknown", eq=False)

    # Calculation parameters
    ecdf_method: str = attrs.field(
        default="linear_interpolation",
        validator=attrs.validators.in_(["kernel_density", "linear_interpolation", "step_function"]),
    )
    cdf_threshold: int = attrs.field(default="1e-10", validator=attrs.validators.instance_of(float))

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], time_window_length: int = 50, **kwargs):
        """
        Instanciates the class from a variable: either a string referring to a standard variable name or a Variable object.

        Parameters
        ----------
        variable : Union[str, Variable]
            String or Variable object referring to standard meteorological variable for which default settings can be used.
        time_window_length: int
            Length of moving time window to fit ECDFs.
        **kwargs:
            All other class attributes that shall be set and where the standard values for variable shall be overwritten.
        """
        if not isinstance(variable, Variable):
            variable = map_variable_str_to_variable_class(variable)

        parameters = {
            **default_settings[variable],
            "time_window_length": time_window_length,
            "variable": variable.name,
        }
        return cls(**{**parameters, **kwargs})

    @classmethod
    def for_precipitation(
        cls,
        time_window_length=50,
        precipitation_model_type: str = "censored",
        precipitation_amounts_distribution: scipy.stats.rv_continuous = scipy.stats.gamma,
        precipitation_censoring_value: float = 0.1,
        precipitation_hurdle_model_randomization: bool = True,
        **kwargs
    ):
        """
        Instanciates the class to a precipitation-debiaser. This allows granular setting of available precipitation models without needing to explicitly specify the precipitation censored model for example.

        Parameters
        ----------
        time_window_length: int
            Length of moving time window to fit ECDFs.
        precipitation_model_type: str
            One of ["censored", "hurdle", "ignore_zeros"]. Model type to be used. See utils.gen_PrecipitationGammaLeftCensoredModel, utils.gen_PrecipitationHurdleModel and utils.gen_PrecipitationIgnoreZeroValuesModel for more details.
        precipitation_amounts_distribution: scipy.stats.rv_continuous
            Distribution used for precipitation amounts. For the censored model only scipy.stats.gamma is possible.
        precipitation_censoring_value: float
            The censoring-value if a censored precipitation model is used.
        precipitation_hurdle_model_randomization: bool
            Whether when computing the cdf-values for a hurdle model randomization shall be used. See utils.gen_PrecipitationHurdleModel for more details
        **kwargs:
            All other class attributes that shall be set and where the standard values shall be overwritten.

        """
        variable = Precipitation

        method = map_standard_precipitation_method(
            precipitation_model_type,
            precipitation_amounts_distribution,
            precipitation_censoring_value,
            precipitation_hurdle_model_randomization,
        )
        parameters = {
            **default_settings[variable],
            "time_window_length": time_window_length,
            "distribution": method,
            "variable": variable.name,
        }
        return cls(**{**parameters, **kwargs})

    def apply_location(self, obs: np.ndarray, cm_hist: np.ndarray, cm_future: np.ndarray) -> np.ndarray:
        """Applies QuantileDeltaMapping at one location and returns the debiased timeseries."""
        fit_obs = self.distribution.fit(obs)
        fit_cm_hist = self.distribution.fit(cm_hist)

        # Q: What is the more efficient way to chunk and then return chunks of equal size to write a new array.
        # That, solution below or totally different one?
        # Test: time measurements are quite variable. More testing necessary. Which solution is cleaner?
        debiased_cm_list_of_time_windows = []
        for cm_future_time_window in np.array_split(cm_future, self.time_window_length, axis=0):
            tau_t = threshold_cdf_vals(
                ecdf(cm_future_time_window, cm_future_time_window, method=self.ecdf_method),
                cdf_threshold=self.cdf_threshold,
            )
            debiased_cm_list_of_time_windows.append(
                cm_future_time_window
                * self.distribution.ppf(tau_t, *fit_obs)
                / self.distribution.ppf(tau_t, *fit_cm_hist)
            )
        debiased_cm = np.concatenate(debiased_cm_list_of_time_windows)

        # Alternative:
        """debiased_cm = np.empty(cm_future.shape[0])
        for time_chunk in np.array_split(
            np.array(range(cm_future.shape[0])), self.time_window_length, axis=0
        ):
            ecdf_cm_future_time_window = ECDF(cm_future[time_chunk])
            tau_t = ecdf_cm_future_time_window(cm_future[time_chunk])

            debiased_cm[time_chunk] = (
                cm_future[time_chunk]
                * self.distribution.ppf(tau_t, *fit_obs)
                / self.distribution.ppf(tau_t, *fit_cm_hist)
            )"""

        return debiased_cm
