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
from scipy.signal import detrend
from scipy.stats import norm

from ..utils import interp_sorted_cdf_vals_on_given_length, threshold_cdf_vals
from ..variables import Variable, map_variable_str_to_variable_class, pr, tas
from ._debiaser import Debiaser

default_settings = {
    tas: {"mapping_type": "absolute"},
    pr: {"mapping_type": "relative"},
}


# Reference Switanek et al. 2017
@attrs.define
class ScaledDistributionMapping(Debiaser):

    mapping_type: str = attrs.field(validator=attrs.validators.in_(["absolute", "relative"]))
    variable: str = attrs.field(default="unknown", eq=False)

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        """
        Instanciates the class from a variable: either a string referring to a standard variable name or a Variable object.

        Parameters
        ----------
        variable : Union[str, Variable]
            String or Variable object referring to standard meteorological variable for which default settings can be used.
        **kwargs:
            All other class attributes that shall be set and where the standard values for variable shall be overwritten.
        """
        if not isinstance(variable, Variable):
            variable = map_variable_str_to_variable_class(variable)

        parameters = {
            **default_settings[variable],
            "variable": variable.name,
        }
        return cls(**{**parameters, **kwargs})

    def apply_location_temp(self, obs, cm_hist, cm_future):

        # Step 1
        obs_detrended = detrend(obs, type="constant")
        cm_hist_detrended = detrend(cm_hist, type="constant")
        cm_future_detrended = detrend(cm_future, type="constant")

        # Step 2
        fit_obs_detrended = norm.fit(obs_detrended)
        fit_cm_hist_detrended = norm.fit(cm_hist_detrended)
        fit_cm_future_detrended = norm.fit(cm_future_detrended)

        argsort_cm_future = np.argsort(cm_future_detrended)

        cdf_vals_obs_detrended_thresholded = threshold_cdf_vals(np.sort(norm.cdf(obs_detrended, *fit_obs_detrended)))
        cdf_vals_cm_hist_detrended_thresholded = threshold_cdf_vals(
            np.sort(norm.cdf(cm_hist_detrended, *fit_cm_hist_detrended))
        )
        cdf_vals_cm_future_detrended_thresholded = threshold_cdf_vals(
            norm.cdf(cm_future_detrended, *fit_cm_future_detrended)[argsort_cm_future]
        )

        # interpolate cdf-values for obs and mod to the length of the scenario
        cdf_vals_obs_detrended_thresholded_intpol = interp_sorted_cdf_vals_on_given_length(
            cdf_vals_obs_detrended_thresholded, cm_future.size
        )
        cdf_vals_cm_hist_detrended_thresholded_intpol = interp_sorted_cdf_vals_on_given_length(
            cdf_vals_cm_hist_detrended_thresholded, cm_future.size
        )

        # Step 3
        scaling = (
            (
                norm.ppf(cdf_vals_cm_future_detrended_thresholded, *fit_cm_future_detrended)
                - norm.ppf(cdf_vals_cm_future_detrended_thresholded, *fit_cm_hist_detrended)
            )
            * fit_obs_detrended[1]
            / fit_cm_hist_detrended[1]
        )

        # Step 4
        recurrence_interval_obs = 1 / (0.5 - np.abs(cdf_vals_obs_detrended_thresholded_intpol - 0.5))
        recurrence_interval_cm_hist = 1 / (0.5 - np.abs(cdf_vals_cm_hist_detrended_thresholded_intpol - 0.5))
        recurrence_interval_cm_future = 1 / (0.5 - np.abs(cdf_vals_cm_future_detrended_thresholded - 0.5))

        # Step 5
        recurrence_interval_scaled = np.maximum(
            1,
            recurrence_interval_obs * recurrence_interval_cm_future / recurrence_interval_cm_hist,
        )
        cdf_scaled = threshold_cdf_vals(
            0.5
            + np.sign(cdf_vals_obs_detrended_thresholded_intpol - 0.5) * np.abs(0.5 - 1 / recurrence_interval_scaled)
        )

        # Step 6
        bias_corrected = norm.ppf(cdf_scaled, *fit_obs_detrended) + scaling

        # Step 7
        reverse_sorting_idx = np.argsort(argsort_cm_future)
        return bias_corrected[reverse_sorting_idx] + (cm_future - cm_future_detrended)

    def apply_location(self, obs, cm_hist, cm_future):
        if self.mapping_type == "absolute":
            return self.apply_location_temp(obs, cm_hist, cm_future)
        elif self.mapping_type == "relative":
            pass
        else:
            raise ValueError('self.mapping_type needs too be one of ["absolute", "relative"].')

    def absolute_sdm_location(self, obs_data, mod_data, sce_data, **kwargs):
        from scipy.signal import detrend
        from scipy.stats import norm

        cdf_threshold = kwargs.get("cdf_threshold", 0.99999)

        obs_len = len(obs_data)
        mod_len = len(mod_data)

        obs_mean = obs_data.mean()
        mod_mean = mod_data.mean()

        # detrend the data
        obs_detrended = detrend(obs_data, type="constant")
        mod_detrended = detrend(mod_data, type="constant")

        obs_norm = norm.fit(obs_detrended)
        mod_norm = norm.fit(mod_detrended)

        obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
        mod_cdf = norm.cdf(np.sort(mod_detrended), *mod_norm)
        obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
        mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)

        sce_len = len(sce_data)
        sce_mean = sce_data.mean()

        sce_detrended = detrend(sce_data, type="constant")
        sce_diff = sce_data - sce_detrended
        sce_argsort = np.argsort(sce_detrended)

        sce_norm = norm.fit(sce_detrended)
        sce_cdf = norm.cdf(np.sort(sce_detrended), *sce_norm)
        sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

        # interpolate cdf-values for obs and mod to the length of the
        # scenario
        obs_cdf_intpol = np.interp(np.linspace(1, obs_len, sce_len), np.linspace(1, obs_len, obs_len), obs_cdf)
        mod_cdf_intpol = np.interp(np.linspace(1, mod_len, sce_len), np.linspace(1, mod_len, mod_len), mod_cdf)

        # adapt the observation cdfs
        # split the tails of the cdfs around the center
        obs_cdf_shift = obs_cdf_intpol - 0.5
        mod_cdf_shift = mod_cdf_intpol - 0.5
        sce_cdf_shift = sce_cdf - 0.5
        obs_inverse = 1.0 / (0.5 - np.abs(obs_cdf_shift))
        mod_inverse = 1.0 / (0.5 - np.abs(mod_cdf_shift))
        sce_inverse = 1.0 / (0.5 - np.abs(sce_cdf_shift))

        # Using adapted_cdf2 as below I get different results more in line with my code.
        # Difference between my cdf_scaled and the adapted_cdf is pretty much the same
        # as between adapted_cdf and adapted_cdf2
        adapted_cdf2 = 0.5 + np.sign(obs_cdf_shift) * np.abs(
            0.5 - 1.0 / np.maximum(1, obs_inverse * sce_inverse / mod_inverse)
        )
        adapted_cdf2 = np.maximum(np.minimum(adapted_cdf2, cdf_threshold), 1 - cdf_threshold)

        adapted_cdf = np.sign(obs_cdf_shift) * (1.0 - 1.0 / (obs_inverse * sce_inverse / mod_inverse))
        adapted_cdf[adapted_cdf < 0] += 1.0
        adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

        print(np.sort(np.abs(adapted_cdf - adapted_cdf2)))
        adapted_cdf = adapted_cdf2

        xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) + obs_norm[-1] / mod_norm[-1] * (
            norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm)
        )
        xvals -= xvals.mean()
        xvals += obs_mean + (sce_mean - mod_mean)

        correction = np.zeros(sce_len)
        correction[sce_argsort] = xvals
        correction += sce_diff - sce_mean
        return correction


"""
Several questions / issues with absolute_sdm:
- I can calculate the adapted_cdf as:

adapted_cdf2 = .5 + np.sign(obs_cdf_shift)*np.abs(.5 - 1. / (obs_inverse * sce_inverse / mod_inverse))
adapted_cdf2 = np.maximum(np.minimum(adapted_cdf2, cdf_threshold), 1 - cdf_threshold)

which corresponds to what is written in the publication.
However the results I get are different from the adapted cdf above:

adapted_cdf = np.sign(obs_cdf_shift) * (1. - 1. / (obs_inverse * sce_inverse / mod_inverse))
adapted_cdf[adapted_cdf < 0] += 1.
adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

which one is the right one?


- Why is adapted_cdf sorted again in:

xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) \
            + obs_norm[-1] / mod_norm[-1] \
            * (norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm))

to then be reinserted again with:

correction[sce_argsort] = xvals

wouldn't that sorting above just undo the effect of sce_argsort.
Or is the most important thing backinserting to make sure the biggest value is inserted into the previous biggest value.


- Why is the mean of the new values subtracted here and new means are added:

xvals -= xvals.mean()
xvals += obs_mean + (sce_mean - mod_mean)

correction = np.zeros(sce_len)
correction[sce_argsort] = xvals
correction += sce_diff - sce_mean

In the publication it only says that the trend is reinserted again which would be just correction += sce_diff.
But here several means are added and subtracted which isn't named in the paper. Also woulnd't it amount to:

correction = correction + sce_diff + obs_mean - mod_mean

What does this factor obs_mean - mod_mean do here?
This looks like some kind of delta change, but shouln't it already be accounted for in the rest of the methodology?
"""
