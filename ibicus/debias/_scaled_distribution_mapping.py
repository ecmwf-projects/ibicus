# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from typing import Union

import attrs
import numpy as np
import scipy.stats
from scipy.signal import detrend

from ..utils import (
    StatisticalModel,
    interp_sorted_cdf_vals_on_given_length,
    threshold_cdf_vals,
)
from ..variables import Variable, pr, tas, tasmax, tasmin
from ._debiaser import Debiaser

# ----- Default settings for debiaser ----- #
default_settings = {
    tas: {"mapping_type": "absolute", "distribution": scipy.stats.norm},
    pr: {
        "mapping_type": "relative",
        "distribution": scipy.stats.gamma,
        "pr_lower_threshold": 0.1 / 86400,
        "distribution_fit_kwargs": {"floc": 0},
    },
}
experimental_default_settings = {
    tasmin: {"mapping_type": "absolute", "distribution": scipy.stats.norm},
    tasmax: {"mapping_type": "absolute", "distribution": scipy.stats.norm},
}


# ----- Debiaser ----- #


@attrs.define(slots=False)
class ScaledDistributionMapping(Debiaser):
    """
    |br| Implements Scaled Distribution Matching (SDM) based on Switanek et al. 2017.

    SDM is conceptually similar to QDM, and in the same ‘family’ as CDFt and ECDFM. It is a parametric quantile mapping approach that also attempts to be trend preserving in all quantiles. In addition to the quantile mapping the method also contains an event likelihood adjustment.


    SDM scales the observed distribution by changes in magnitude and additionally likelihood of events -- either multiplicatively (for precipitation) or additively (for temperature).

    Let cm refer to climate model output, obs to observations and hist/future to whether the data was collected from the reference period or is part of future projections.
    Let :math:`F` be a parametric CDF.

    1. Temperature (``tas``): absolute scaled distribution mapping

    First CDFs are fitted to both historical and future climate model values as well as observations. The default settings for ``tas`` use a normal distribution. Then the scaling is calculated as:

    .. math:: \\text{scaling} = [F^{-1}_{\\text{cm_fut}}(F_{\\text{cm_fut}}(x_{\\text{cm_fut}})) - F^{-1}_{\\text{cm_hist}}(F_{\\text{cm_fut}}(x_{\\text{cm_fut}}))] * \\frac{\\sigma_\\text{obs}}{\\sigma_\\text{cm_hist}}

    where :math:`\\sigma_\\text{obs}` and :math:`\\sigma_\\text{cm_hist}` refers to the standard deviation of a normal distribution fitted to obs and cm_hist. Given the CDFs for obs, cm_hist, cm_fut then recurrence intervals for all three are calculated as:

    .. math:: \\text{RI} = \\frac{1}{0.5 - \\|CDF - 0.5\\|}

    and a scaled recurrence interval (RI) and CDF as:

    .. math:: \\text{RI}_{\\text{scaled}} = \\text{max}\\left(1, \\frac{\\text{RI}_{\\text{obs}} \\cdot \\text{RI}_{\\text{cm_fut}}}{\\text{RI}_{\\text{cm_hist}}}\\right)

    .. math:: \\text{CDF}_{\\text{scaled}} = 0.5 + \\text{sgn}(\\text{CDF}_{\\text{obs}} - 0.5) \\cdot \\left|  0.5 - \\frac{1}{\\text{RI}_{\\text{scaled}}} \\right|.

    Then the adjusted values are given as follows:

    .. math:: F^{-1}_{\\text{obs}}(\\text{CDF}_{\\text{scaled}}) + \\text{scaling}

    |br|

    2. Precipitation (``pr``): relative scaled distribution mapping

    For precipitation first in obs, cm_hist and cm_fut all values below a given threshold are set to zero rain. Let :math:`\\text{# rain}` design the number of rain days and :math:`\\text{# total}` the total number of days. The bias corrected number of rain days is calculated as:

    .. math:: \\text{# rain}_{\\text{bc}} = \\text{# rain}_{\\text{cm_fut}} \\cdot \\frac{\\text{# rain}_{\\text{obs}} / \\text{# total}_{\\text{obs}}}{\\text{# rain}_{\\text{cm_hist}} / \\text{# rain}_{\\text{cm_hist}}}

    Using all values bigger than the threshold a scaling is calculated as:

    .. math:: \\text{scaling} = \\frac{F^{-1}_{\\text{cm_fut}}(F_{\\text{cm_fut}}(x_{\\text{cm_fut}}))}{F^{-1}_{\\text{cm_hist}}(F_{\\text{cm_fut}}(x_{\\text{cm_fut}}))}

    :math:`\\text{RI}_{\\text{scaled}}` and :math:`\\text{CDF}_{\\text{scaled}}` are calculated similarly as for temperature. The final bias corrected precipitation values on rainy days are then given as:

    .. math:: F^{-1}_{\\text{obs}}(\\text{CDF}_{\\text{scaled}}) \\cdot \\text{scaling}

    The :math:`\\text{# total}_{\\text{bc}} - \\text{# rain}_{\\text{bc}}` days with the smallest precipitation values in cm_fut are then set to zero and all bias corrected rain values are inserted at the correct locations, starting with the biggest one.

    .. warning:: The relative SDM method does not currently allow correcting the number of precipitation days in cm_fut upwards, so to convert dry into rainy days. Should the calculated expected number of rainy days be higher than what is given inside the future climate model then the number of rainy days is left unadjusted. The method focuses on the biggest precipitation values, so this should not be an issue for most applications. However if such a correction is required this method might not be appropriate.


    **Reference**:

    - Switanek, M. B., Troch, P. A., Castro, C. L., Leuprecht, A., Chang, H.-I., Mukherjee, R., & Demaria, E. M. C. (2017). Scaled distribution mapping: a bias correction method that preserves raw climate model projected changes. In Hydrology and Earth System Sciences (Vol. 21, Issue 6, pp. 2649–2666). Copernicus GmbH. https://doi.org/10.5194/hess-21-2649-2017.

    |br|
    **Usage information:**

    - Default settings exist for: ``["pr", "tas", "tasmin", "tasmax"]``.

    - :py:func:`apply` requires: no additional arguments except ``obs``, ``cm_hist``, ``cm_future``.

    - Next to :py:func:`from_variable` a :py:func:`for_precipitation`-method exists to help you initialise the debiaser for :py:data:`pr`.

    - The method has been developed for daily data, however application on data in other time specifications (monthly etc.) is possible.

    |br|
    **Examples:**

    >>> debiaser = ScaledDistributionMapping.from_variable("tas")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    |br|

    Attributes
    ----------
    distribution : Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel]
        Method used for the fit to the historical and future climate model outputs as well as the observations.
        Usually a distribution in ``scipy.stats.rv_continuous``, but can also be an empirical distribution as given by ``scipy.stats.rv_histogram`` or a more complex statistical model as wrapped by the :py:class:`ibicus.utils.StatisticalModel` class.
    mapping_type : str
        One of ``["absolute", "relative"]``. Type of SDM used. Default are "absolute" for ``tas`` and ``"relative"`` for ``pr``.
    pr_lower_threshold : float
        Lower threshold used for setting precipitation values to zero in relative SDM. Only used if ``mapping_type = "relative"``.
    distribution_fit_kwargs : dict
        Dict of additional arguments passed to the ``distribution.fit``-method. Useful for fixing certain parameters of a distribution. Default: ``{}`` (empty dict).
    cdf_threshold : float
        Threshold to round CDF-values away from zero and one. Default: ``1e-10``.
    variable : str
        Variable for which the debiasing is done. Default: ``"unknown"``.
    """

    # Core algorithm
    distribution: Union[
        scipy.stats.rv_continuous,
        scipy.stats.rv_discrete,
        scipy.stats.rv_histogram,
        StatisticalModel,
    ] = attrs.field(
        validator=attrs.validators.instance_of(
            (
                scipy.stats.rv_continuous,
                scipy.stats.rv_discrete,
                scipy.stats.rv_histogram,
                StatisticalModel,
            )
        )
    )
    mapping_type: str = attrs.field(
        validator=attrs.validators.in_(["absolute", "relative"])
    )

    # pr and relative sdm
    pr_lower_threshold: float = attrs.field(
        default=0.1 / 86400, validator=attrs.validators.instance_of(float)
    )

    # Computation
    distribution_fit_kwargs: dict = attrs.field(
        default={}, validator=attrs.validators.instance_of(dict)
    )
    cdf_threshold: float = attrs.field(
        default=1e-10, validator=attrs.validators.instance_of(float)
    )

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super()._from_variable(
            cls, variable, default_settings, experimental_default_settings, **kwargs
        )

    @classmethod
    def for_precipitation(cls, pr_lower_threshold=0.1 / 86400, **kwargs):
        """
        Instanciates the class to a precipitation-debiaser. This allows an easier setting of the lower precipitation threshold (``pr_lower_threshold``) under which precipitation is assumed zero.

        Parameters
        ----------
        pr_lower_threshold : float = 0.1/86400
            Lower precipitation threshold under which precipitation is assumed zero.
        **kwargs:
            All other class attributes that shall be set and where the standard values shall be overwritten.

        """
        return cls.from_variable("pr", pr_lower_threshold=pr_lower_threshold, **kwargs)

    def apply_location_relative_sdm(self, obs, cm_hist, cm_future):

        # Preparation: sort arrays
        obs = np.sort(obs)
        cm_hist = np.sort(cm_hist)

        argsort_cm_future = np.argsort(cm_future)
        cm_future = cm_future[argsort_cm_future]

        # Step 1

        # Get mask for rainy days
        mask_rainy_days_obs = obs >= self.pr_lower_threshold
        mask_rainy_days_cm_hist = cm_hist >= self.pr_lower_threshold
        mask_rainy_days_cm_future = cm_future >= self.pr_lower_threshold

        # Get rainy day values
        rainy_days_obs = obs[mask_rainy_days_obs]
        rainy_days_cm_hist = cm_hist[mask_rainy_days_cm_hist]
        rainy_days_cm_future = cm_future[mask_rainy_days_cm_future]

        if (
            rainy_days_obs.size == 0
            or rainy_days_cm_hist.size == 0
            or rainy_days_cm_future.size == 0
        ):
            raise ValueError(
                "No values bigger than pr_lower_threshold in either obs, cm_hist or cm_future. Bias adjustment not possible."
            )

        # Set non rainy days to zero
        obs[np.logical_not(mask_rainy_days_obs)] = 0
        cm_hist[np.logical_not(mask_rainy_days_cm_hist)] = 0

        # Calculate the expected bias corrected number of rainy days in cm_future
        expected_nr_rainy_days_cm_future = round(
            mask_rainy_days_cm_future.sum()
            * (mask_rainy_days_obs.sum() / mask_rainy_days_obs.size)
            / (mask_rainy_days_cm_hist.sum() / mask_rainy_days_cm_hist.size)
        )

        if expected_nr_rainy_days_cm_future > mask_rainy_days_cm_future.sum():
            warnings.warn(
                """The relative ScaledDistributionMapping does not currently support adjusting the number of rainy days upwards: so to transform dry days into rainy ones in cm_future.
                The number of dry and rainy days is left unadjusted.""",
                stacklevel=2,
            )
            expected_nr_rainy_days_cm_future = mask_rainy_days_cm_future.sum()

        # Step 2

        # Fit distribution
        fit_rainy_days_obs = self.distribution.fit(
            rainy_days_obs, **self.distribution_fit_kwargs
        )
        fit_rainy_days_cm_hist = self.distribution.fit(
            rainy_days_cm_hist, **self.distribution_fit_kwargs
        )
        fit_rainy_days_cm_future = self.distribution.fit(
            rainy_days_cm_future, **self.distribution_fit_kwargs
        )

        # Calculate CDF vals
        cdf_vals_rainy_days_obs_thresholded = threshold_cdf_vals(
            self.distribution.cdf(rainy_days_obs, *fit_rainy_days_obs),
            self.cdf_threshold,
        )
        cdf_vals_rainy_days_cm_hist_thresholded = threshold_cdf_vals(
            self.distribution.cdf(rainy_days_cm_hist, *fit_rainy_days_cm_hist),
            self.cdf_threshold,
        )
        cdf_vals_rainy_days_cm_future_thresholded = threshold_cdf_vals(
            self.distribution.cdf(rainy_days_cm_future, *fit_rainy_days_cm_future),
            self.cdf_threshold,
        )

        # Interpolate CDF vals of obs and cm_hist onto length of cm_future
        cdf_vals_rainy_days_obs_thresholded_intpol = (
            interp_sorted_cdf_vals_on_given_length(
                cdf_vals_rainy_days_obs_thresholded,
                cdf_vals_rainy_days_cm_future_thresholded.size,
            )
        )
        cdf_vals_rainy_days_cm_hist_thresholded_intpol = (
            interp_sorted_cdf_vals_on_given_length(
                cdf_vals_rainy_days_cm_hist_thresholded,
                cdf_vals_rainy_days_cm_future_thresholded.size,
            )
        )

        # Step 3

        scaling = self.distribution.ppf(
            cdf_vals_rainy_days_cm_future_thresholded, *fit_rainy_days_cm_future
        ) / self.distribution.ppf(
            cdf_vals_rainy_days_cm_future_thresholded, *fit_rainy_days_cm_hist
        )

        # Step 4

        recurrence_interval_obs = 1 / (1 - cdf_vals_rainy_days_obs_thresholded_intpol)
        recurrence_interval_cm_hist = 1 / (
            1 - cdf_vals_rainy_days_cm_hist_thresholded_intpol
        )
        recurrence_interval_cm_future = 1 / (
            1 - cdf_vals_rainy_days_cm_future_thresholded
        )

        # Step 5

        recurrence_interval_scaled = np.maximum(
            1,
            recurrence_interval_obs
            * recurrence_interval_cm_future
            / recurrence_interval_cm_hist,
        )
        cdf_scaled = threshold_cdf_vals(1 - 1 / recurrence_interval_scaled)

        # Step 6

        bc_initial = self.distribution.ppf(cdf_scaled, *fit_rainy_days_obs) * scaling
        cm_future[: cm_future.size - expected_nr_rainy_days_cm_future] = 0
        cm_future[cm_future.size - expected_nr_rainy_days_cm_future :] = bc_initial[
            bc_initial.size - expected_nr_rainy_days_cm_future :
        ]

        # Step 7
        reverse_sorting_idx = np.argsort(argsort_cm_future)
        return cm_future[reverse_sorting_idx]

    def apply_location_absolute_sdm(self, obs, cm_hist, cm_future):

        # Step 1
        obs_detrended = detrend(obs, type="constant")
        cm_hist_detrended = detrend(cm_hist, type="constant")
        cm_future_detrended = detrend(cm_future, type="constant")

        # Step 2
        fit_obs_detrended = self.distribution.fit(obs_detrended)
        fit_cm_hist_detrended = self.distribution.fit(cm_hist_detrended)
        fit_cm_future_detrended = self.distribution.fit(cm_future_detrended)

        argsort_cm_future = np.argsort(cm_future_detrended)

        cdf_vals_obs_detrended_thresholded = threshold_cdf_vals(
            np.sort(self.distribution.cdf(obs_detrended, *fit_obs_detrended))
        )
        cdf_vals_cm_hist_detrended_thresholded = threshold_cdf_vals(
            np.sort(self.distribution.cdf(cm_hist_detrended, *fit_cm_hist_detrended))
        )
        cdf_vals_cm_future_detrended_thresholded = threshold_cdf_vals(
            self.distribution.cdf(cm_future_detrended, *fit_cm_future_detrended)[
                argsort_cm_future
            ]
        )

        # interpolate cdf-values for obs and mod to the length of the scenario
        cdf_vals_obs_detrended_thresholded_intpol = (
            interp_sorted_cdf_vals_on_given_length(
                cdf_vals_obs_detrended_thresholded, cm_future.size
            )
        )
        cdf_vals_cm_hist_detrended_thresholded_intpol = (
            interp_sorted_cdf_vals_on_given_length(
                cdf_vals_cm_hist_detrended_thresholded, cm_future.size
            )
        )

        # Step 3
        scaling = (
            (
                self.distribution.ppf(
                    cdf_vals_cm_future_detrended_thresholded, *fit_cm_future_detrended
                )
                - self.distribution.ppf(
                    cdf_vals_cm_future_detrended_thresholded, *fit_cm_hist_detrended
                )
            )
            * fit_obs_detrended[1]
            / fit_cm_hist_detrended[1]
        )

        # Step 4
        recurrence_interval_obs = 1 / (
            0.5 - np.abs(cdf_vals_obs_detrended_thresholded_intpol - 0.5)
        )
        recurrence_interval_cm_hist = 1 / (
            0.5 - np.abs(cdf_vals_cm_hist_detrended_thresholded_intpol - 0.5)
        )
        recurrence_interval_cm_future = 1 / (
            0.5 - np.abs(cdf_vals_cm_future_detrended_thresholded - 0.5)
        )

        # Step 5
        recurrence_interval_scaled = np.maximum(
            1,
            recurrence_interval_obs
            * recurrence_interval_cm_future
            / recurrence_interval_cm_hist,
        )
        cdf_scaled = threshold_cdf_vals(
            0.5
            + np.sign(cdf_vals_obs_detrended_thresholded_intpol - 0.5)
            * np.abs(0.5 - 1 / recurrence_interval_scaled)
        )

        # Step 6
        bias_corrected = self.distribution.ppf(cdf_scaled, *fit_obs_detrended) + scaling

        # Step 7
        trend = cm_future - cm_future_detrended
        reverse_sorting_idx = np.argsort(argsort_cm_future)
        return bias_corrected[reverse_sorting_idx] + trend

    def apply_location(self, obs, cm_hist, cm_future):
        if self.mapping_type == "absolute":
            return self.apply_location_absolute_sdm(obs, cm_hist, cm_future)
        elif self.mapping_type == "relative":
            return self.apply_location_relative_sdm(obs, cm_hist, cm_future)
        else:
            raise ValueError(
                'self.mapping_type needs too be one of ["absolute", "relative"].'
            )
