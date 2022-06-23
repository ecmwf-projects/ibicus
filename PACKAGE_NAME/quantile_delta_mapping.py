from typing import Union

import attrs
import numpy as np
import scipy
from matplotlib import docstring
from statsmodels.distributions.empirical_distribution import ECDF

from .debiaser import Debiaser
from .variable_distribution_match import standard_distributions


# Reference Cannon et al. 2015
# TODO: add docstring
# TODO: check correctness of time window implementation and precipitation-case
@attrs.define
class QuantileDeltaMapping(Debiaser):

    distribution: Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram] = attrs.field(
        validator=attrs.validators.instance_of(
            (
                scipy.stats.rv_continuous,
                scipy.stats.rv_discrete,
                scipy.stats.rv_histogram,
            )
        )
    )
    time_window_length: int = attrs.field(validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)])
    variable: str = attrs.field(default="unknown", eq=False)

    @classmethod
    def from_variable(cls, variable, time_window_length=50):
        if variable not in standard_distributions.keys():
            raise ValueError("variable needs to be one of %s" % standard_distributions.keys())
        return cls(
            distribution=standard_distributions.get(variable),
            time_window_length=time_window_length,
            variable=variable,
        )

    def apply_location(self, obs, cm_hist, cm_future):

        fit_obs = self.distribution.fit(cm_hist)
        fit_cm_hist = self.distribution.fit(cm_hist)

        # Q: What is the more efficient way to chunk and then return chunks of equal size to write a new array.
        # That, solution below or totally different one?
        # Test: time measurements are quite variable. More testing necessary. Which solution is cleaner?
        debiased_cm_list_of_time_windows = []
        for cm_future_time_window in np.array_split(cm_future, self.time_window_length, axis=0):
            ecdf_cm_future_time_window = ECDF(cm_future_time_window)
            tau_t = ecdf_cm_future_time_window(cm_future_time_window)
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
