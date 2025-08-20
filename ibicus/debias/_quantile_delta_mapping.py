# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from logging import warning
from typing import Optional, Union

import attrs
import numpy as np
import scipy

from ..utils import (
    StatisticalModel,
    ecdf,
    gen_PrecipitationGammaLeftCensoredModel,
    iecdf,
    threshold_cdf_vals,
)
from ..variables import Variable, hurs, pr, psl, rlds, sfcwind, tas, tasmax, tasmin
from ._running_window_debiaser import SeasonalAndFutureRunningWindowDebiaser

# ----- Default settings for debiaser ----- #
default_settings = {
    tas: {
        "distribution": None,
        "trend_preservation": "absolute",
        "mapping_type": "nonparametric",
    },
    pr: {
        "distribution": None,
        "mapping_type": "nonparametric",
        "trend_preservation": "relative",
        "censor_values_to_zero": True,
        "censoring_threshold": 0.05 / 86400,
    },
}
experimental_default_settings = {
    hurs: {
        "distribution": None,
        "trend_preservation": "relative",
        "mapping_type": "nonparametric",
    },
    psl: {
        "distribution": None,
        "trend_preservation": "absolute",
        "mapping_type": "nonparametric",
    },
    rlds: {
        "distribution": None,
        "trend_preservation": "absolute",
        "mapping_type": "nonparametric",
    },
    sfcwind: {
        "distribution": None,
        "trend_preservation": "relative",
        "mapping_type": "nonparametric",
    },
    tasmin: {
        "distribution": None,
        "trend_preservation": "absolute",
        "mapping_type": "nonparametric",
    },
    tasmax: {
        "distribution": None,
        "trend_preservation": "absolute",
        "mapping_type": "nonparametric",
    },
}


# ----- Debiaser ----- #


@attrs.define(slots=False)
class QuantileDeltaMapping(SeasonalAndFutureRunningWindowDebiaser):
    """
    |br| Implements Quantile Delta Mapping based on Cannon et al. 2015.

    QDM is a parametric quantile mapping method that also attempts to be trend-preserving. It extends ECDFM such that the two quantile mappings defined there are not only added but also divided by each other to create multiplicative correction. Furthermore it includes both a running window over the year: to account for seasonality, as well as one over the future period to account for changes in trends.


    Let cm refer to climate model output, obs to observations and hist/future to whether the data was collected from the reference period or is part of future projections.

    Similar to the Equidistant CDF Matching Method based on Li et al. 2010 and implemented in :py:class:`ECDFM`, this method bias corrects the future climate model data directly using reference period observations. This is then multiplied by the quotient of future climate model data and a quantile mapping between past and future climate model data to account for the change from past to future period in all quantiles.

    In mathematical terms, the transformation can be written as:

    .. math:: x_{\\text{cm_fut, bc}} (t) = x_{\\text{cm_fut}} (t) \\cdot \\frac{F^{-1}_{\\text{obs}}(\\hat F_{\\text{cm_fut}}^{(t)}(x_{\\text{cm_fut}}(t)))}{F^{-1}_{\\text{cm_hist}}(\\hat F_{\\text{cm_fut}}^{(t)}(x_{\\text{cm_fut}}))}

    This variation preserves relative trends. If absolute trends are to be preserved, the transformation is equivalent to the transformation introduced in Li et al 2010:

    .. math:: x_{\\text{cm_fut, bc}} (t) = x_{\\text{cm_fut}} (t) + F^{-1}_{\\text{obs}}(\\hat F_{\\text{cm_fut}}^{(t)}(x_{\\text{cm_fut}}(t))) - F^{-1}_{\\text{cm_hist}}(\\hat F_{\\text{cm_fut}}^{(t)} ( x_{\\text{cm_fut}})).

    Hereby :math:`\\hat F_{\\text{cm_fut}}^{(t)}` is the empirical CDF of future climate model values in a window around t. :math:`F^{-1}_{\\text{obs}}` is the inverse CDF estimated by fitting a distribution to observations. :math:`F^{-1}_{\\text{cm_hist}}` is the inverse CDF estimated by fitting a distribution to the historical climate model run.

    Delta Quantile Mapping is approximately trend preserving in all quantiles because the absolute :math:`\\Delta_{\\text{cm}}^{\\text{abs}}` or relative change :math:`\\Delta_{\\text{cm}}^{\\text{rel}}` is calculated and applied for each quantile individually.

    Running window:

    - ``running_window_over_year``: controls whether the methodology and mapping is applied on a running window over the year to account for seasonality. ``running_window_over_year_length`` and ``running_window_over_year_step_length`` control the length (how many days are included in the running window) and step length (by how far the window is shifted and how many days inside are debiased) respectively. |brr|
    - ``running_window_mode_over_years_of_cm_future`` controls whether a running window is used to estimate the empirical CDF :math:`\\hat F_{\\text{cm_fut}}^{(t)}(x_{\\text{cm_fut}}(t))` and time-dependent quantiles or if this is done statically on the whole future climate model run. ``running_window_over_years_of_cm_future_length`` and ``running_window_over_years_of_cm_future_step_length control`` the length and step length of this window respectively. Estimating this information in a running window has the advantage of accounting for changes in trends.

    If both running windows are active then first the running window inside the year is used to account for seasonality. Values are chunked according to this one. Afterwards the running window over years is used and values are further split up. This is just a choice made for computational efficiency and the order of running window application/chunking does not matter.


    .. warning:: Currently only uneven sizes are allowed for window length and window step length. This allows symmetrical windows of the form [window_center - window length//2, window_center + window length//2`] given an arbitrary window center. This affects both within year and over year window.


    **References**:

    - Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes? In Journal of Climate (Vol. 28, Issue 17, pp. 6938–6959). American Meteorological Society. https://doi.org/10.1175/jcli-d-14-00754.1

    |br|
    **Usage information:**

    - Default settings exist for: ``["hurs", "pr", "psl", "rlds", "sfcWind", "tas", "tasmin", "tasmax"]``.

    - :py:func:`apply` requires: time arguments ``time_obs``, ``time_cm_hist``, and ``time_cm_future`` next to ``obs``, ``cm_hist`` and ``cm_future``. These are just 1d numpy-arrays of dates (multiple formats are possible as long as they as convertible to numpy or datetime dates) specifying the date for each value/timestep in ``obs``, ``cm_hist`` and ``cm_future``. If they are not specified they are inferred, assuming the first observation in all three observation/climate value arrays is on a 1st of January.

    - Next to :py:func:`from_variable` a :py:func:`for_precipitation`-method exists to help you initialise the debiaser for :py:data:`pr`.

    - The debiaser has been developed for and assumes daily data, although application on data using other time specifications (monthly etc.) is possible by setting ``running_window_mode_within_year = False``, modifying the running window arguments for the running window over the years of cm_future, and specifying the time arguments in :py:func:`apply`.

    |br|
    **Examples:**

    Running without dates (they are inferred assuming the first value in ``obs``, ``cm_hist`` and ``cm_future`` always corresponds to a January 1st):

    >>> debiaser = QuantileDeltaMapping.from_variable("tas")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    Running with dates and initialising using :py:func:`for_precipitation`:

    >>> debiaser = QuantileDeltaMapping.for_precipitation(censoring_threshold = 0.1/86400)
    >>> debiaser.apply(obs, cm_hist, cm_future, time_obs = time_obs, time_cm_hist = time_cm_hist, time_cm_future = time_cm_future)

    |br|

    Attributes
    ----------
    distribution : Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel]
        Distribution or statistical model used to compute the CDF :math:`F` of observations and historical climate model values.
        Usually a distribution in scipy.stats.rv_continuous, but can also be an empirical distribution as given by scipy.stats.rv_histogram or a more complex statistical model as wrapped by the :py:class:`ibicus.utils.StatisticalModel` class.
        Only relevant if ``mapping_type = "parametric"``.
    mapping_type : str
        One of ``["parametric", "nonparametric"]``. Whether all CDF mappings are done using parametric CDFs or using nonparametric density estimation. Default: ``"nonparametric"``.
    trend_preservation : str
        One of ``["absolute", "relative"]``. If ``"absolute"`` then absolute trend preservation is used, if ``"relative"`` then relative trend preservation is used. |brr|

    censor_values_to_zero : bool
        Whether values below a censoring threshold shall be censored to zero. Only relevant for precipitation. Default: ``False``.
    censoring_threshold : float
        Threshold below which values shall be censored to zero if ``censor_values_to_zero = True``. Relevant mainly for precipitation. |br|
        If it is used (so ``censor_values_to_zero = True``) one needs to make sure that the distribution fits to censored data, knows the correct ``censoring_threshold`` and assumes all observations under the specified censoring_threshold are zero/censored. |br|
        If the standard for_precipitation and from_variable methods are used to construct the class this is ensured by default. However if this parameter is changed manually or own distributions for precipitation are specified problems can arise. |brr|

    running_window_mode : bool
        Whether QuantileDeltaMapping is used in running window over the year to account for seasonality. If ``running_window_mode = False`` then QuantileDeltaMapping is applied on the whole period. Default: ``True``.
    running_window_length : int
        Length of the running window in days: how many values are used to calculate the bias adjustment transformation. Only relevant if ``running_window_mode = True``. Default: ``91``.
    running_window_step_length : int
        Step length of the running window in days: how many values are bias adjusted inside the running window and by how far it is moved. Only relevant if ``running_window_mode = True``. Default: ``31``.

    running_window_mode_over_years_of_cm_future : bool
        Controls whether the methodology is applied on a running time window, running over the years of the future climate model. This helps to smooth discontinuities in the preserved trends. Default: ``True``.
    running_window_over_years_of_cm_future_length : int
        Length of the running window in years: how many years are used to define the future climate (default: ``31`` years). Only relevant if ``running_window_mode_over_years_of_cm_future = True``.
    running_window_over_years_of_cm_future_step_length : int
        Step length of the running window in years: how many years are bias adjusted inside the running window (default: ``1`` years). Only relevant if ``running_window_mode_over_years_of_cm_future = True``.

    variable : str
        Variable for which the debiasing is done. Default: ``"unknown"``. |brr|

    ecdf_method : str
        One of ``["kernel_density", "linear_interpolation", "step_function"]``. Method used to calculate the empirical CDF. Default: ``"linear_interpolation"``.
    cdf_threshold : Optional[float]
        Threshold for the CDF-values to round away from 0 and 1. Default: None. It is then set to ``1 / (self.running_window_within_year_length * self.running_window_over_years_of_cm_future_length + 1)``

    """

    distribution: Union[
        scipy.stats.rv_continuous,
        scipy.stats.rv_discrete,
        scipy.stats.rv_histogram,
        StatisticalModel,
        None,
    ] = attrs.field(
        validator=attrs.validators.instance_of(
            (
                scipy.stats.rv_continuous,
                scipy.stats.rv_discrete,
                scipy.stats.rv_histogram,
                StatisticalModel,
                type(None),
            )
        )
    )
    trend_preservation: str = attrs.field(
        validator=attrs.validators.in_(["absolute", "relative"])
    )
    mapping_type: str = attrs.field(
        default="nonparametric",
        validator=attrs.validators.in_(["parametric", "nonparametric"]),
    )

    # Relevant for precipitation
    censor_values_to_zero: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    censoring_threshold: float = attrs.field(
        default=0.05 / 86400,
        validator=attrs.validators.instance_of(float),
        converter=float,
    )

    # Running window mode
    running_window_mode: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )
    running_window_length: int = attrs.field(
        default=91,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )
    running_window_step_length: int = attrs.field(
        default=31,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )

    # Running window over years
    running_window_mode_over_years_of_cm_future: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )
    running_window_over_years_of_cm_future_length: int = attrs.field(
        default=31,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )
    running_window_over_years_of_cm_future_step_length: int = attrs.field(
        default=1, validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)]
    )

    # Calculation parameters
    ecdf_method: str = attrs.field(
        default="linear_interpolation",
        validator=attrs.validators.in_(
            ["kernel_density", "linear_interpolation", "step_function"]
        ),
    )
    cdf_threshold: Optional[float] = attrs.field(
        default=None, validator=attrs.validators.instance_of((float, type(None)))
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if self.cdf_threshold is None:
            self.cdf_threshold = 1 / (
                self.running_window_length
                * self.running_window_over_years_of_cm_future_length
                + 1
            )

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        if (variable == "pr" or variable == pr) and (
            censoring_threshold := kwargs.pop("censoring_threshold", None)
        ):
            return QuantileDeltaMapping.for_precipitation(
                censoring_threshold=censoring_threshold, **kwargs
            )
        return super()._from_variable(
            cls, variable, default_settings, experimental_default_settings, **kwargs
        )

    @classmethod
    def for_precipitation(
        cls,
        mapping_type="nonparametric",
        censoring_threshold: float = 0.05 / 86400,
        **kwargs,
    ):
        """
        Instanciates the class to a precipitation-debiaser.

        Parameters
        ----------
        censoring_threshold: float
            The censoring-value under which precipitation amounts are assumed zero/censored.
        **kwargs:
            All other class attributes that shall be set and where the standard values shall be overwritten.

        """
        if mapping_type == "nonparametric":
            if distribution := kwargs.pop("distribution", None):
                warning("distribution is ignored for nonparametric mapping.")
                distribution = None
        elif mapping_type == "parametric":
            if distribution := kwargs.pop("distribution", None):
                warning(
                    "If specifying an own precipitation distribution make sure that the .fit-methods fits to censored data and assumes all observations under the specified censoring_threshold are zero/censored."
                )
            else:
                distribution = gen_PrecipitationGammaLeftCensoredModel(
                    censoring_threshold=censoring_threshold, censor_in_ppf=False
                )
        else:
            raise ValueError(
                'mapping_type must be one of ["parametric", "nonparametric"]'
            )

        return super()._from_variable(
            cls,
            "pr",
            default_settings,
            censoring_threshold=censoring_threshold,
            distribution=distribution,
            mapping_type=mapping_type,
            **kwargs,
        )

    # ----- Main application functions ----- #

    def _get_obs_and_cm_hist_fits(self, obs: np.ndarray, cm_hist: np.ndarray):
        fit_obs = self.distribution.fit(obs)
        fit_cm_hist = self.distribution.fit(cm_hist)

        return fit_obs, fit_cm_hist

    def apply_on_seasonal_and_future_window(
        self,
        obs: np.ndarray,
        cm_hist: np.ndarray,
        cm_future: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Applies QuantileDeltaMapping at one location and returns the debiased timeseries.
        """

        if self.censor_values_to_zero:
            # Shuffle zeros values and values below the threshold prior to computation
            # Shuffling values below the threshold (instead of just zero) helps with issues due to very small climate model precipitation values, which can lead to very large bias correction values.
            mask = obs <= self.censoring_threshold
            obs[mask] = np.random.rand(len(obs[mask])) * self.censoring_threshold

            mask = cm_hist <= self.censoring_threshold
            cm_hist[mask] = (
                np.random.rand(len(cm_hist[mask])) * self.censoring_threshold
            )

            mask = cm_future <= self.censoring_threshold
            cm_future[mask] = (
                np.random.rand(len(cm_future[mask])) * self.censoring_threshold
            )

        tau_m_p = threshold_cdf_vals(
            ecdf(cm_future, cm_future, method=self.ecdf_method),
            cdf_threshold=self.cdf_threshold,
        )

        if self.mapping_type == "nonparametric":
            x_obs_to_cm_fut = iecdf(obs, tau_m_p)
            inv_F_cm_hist_tau = iecdf(cm_hist, tau_m_p)
        elif self.mapping_type == "parametric":
            fit_obs, fit_cm_hist = self._get_obs_and_cm_hist_fits(obs, cm_hist)

            x_obs_to_cm_fut = self.distribution.ppf(tau_m_p, *fit_obs)
            inv_F_cm_hist_tau = self.distribution.ppf(tau_m_p, *fit_cm_hist)
        else:
            raise ValueError(
                'self.mapping_type needs to be one of ["parametric", "nonparametric"]'
            )

        if self.trend_preservation == "absolute":
            bias_corrected_vals = x_obs_to_cm_fut + cm_future - inv_F_cm_hist_tau
        elif self.trend_preservation == "relative":
            bias_corrected_vals = x_obs_to_cm_fut * cm_future / inv_F_cm_hist_tau
        else:
            raise ValueError(
                'self.trend_preservation needs to be one of ["absolute", "relative"]'
            )

        if self.censor_values_to_zero:
            bias_corrected_vals[bias_corrected_vals < self.censoring_threshold] = 0

        return bias_corrected_vals
