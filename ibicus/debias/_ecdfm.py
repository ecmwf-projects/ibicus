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
import scipy.stats

from ..utils import PrecipitationHurdleModelGamma, StatisticalModel, threshold_cdf_vals
from ..variables import (
    Variable,
    hurs,
    map_standard_precipitation_method,
    pr,
    psl,
    rlds,
    sfcwind,
    tas,
    tasmax,
    tasmin,
)
from ._running_window_debiaser import SeasonalAndFutureRunningWindowDebiaser

# ----- Default settings for debiaser ----- #
default_settings = {
    tas: {"distribution": scipy.stats.beta},
    pr: {
        "distribution": PrecipitationHurdleModelGamma,
        "censor_values_to_zero": True,
        "censoring_threshold": 0.1 / 86400,
    },
}
experimental_default_settings = {
    hurs: {"distribution": scipy.stats.beta},
    psl: {"distribution": scipy.stats.beta},
    rlds: {"distribution": scipy.stats.beta},
    sfcwind: {"distribution": scipy.stats.gamma},
    tasmin: {"distribution": scipy.stats.beta},
    tasmax: {"distribution": scipy.stats.beta},
}


# ----- Debiaser ----- #


@attrs.define(slots=False)
class ECDFM(SeasonalAndFutureRunningWindowDebiaser):
    """
    |br| Implements Equidistant CDF Matching (ECDFM) based on Li et al. 2010.

    ECDFM is a parametric quantile mapping method that attempts to be trend-preserving in all quantiles. ECDFM applies quantilewise correction by adding the difference between a quantile mapping of observations and future values and a quantile mapping of historical climate model values to the future climate model ones.


    Let cm refer to climate model output, obs to observations and hist/future to whether the data was collected from the reference period or is part of future projections.
    Let :math:`F_{\\text{cm_hist}}` be the cdf fitted as a parametric distribution to climate model output data in the reference period. The future climate projections :math:`x_{\\text{cm_fut}}` are then mapped to:

    .. math:: x_{\\text{cm_fut}} \\rightarrow x_{\\text{cm_fut}} - F^{-1}_{\\text{cm_hist}}(F_{\\text{cm_fut}}(x_{\\text{cm_fut}})) + F^{-1}_{\\text{obs}}(F_{\\text{cm_fut}}(x_{\\text{cm_fut}}))

    The difference between the future climate model data and the future climate model data quantile mapped to historical climate model data (de facto future model data bias corrected to historical model data) is added to a quantile mapping bias correction between observations and future climate model data.
    In essence, this method says that future climate model data can be bias corrected directly with reference period observations, if the quantile specific difference between present-day and future climate model simulations is taken into account.
    This allows for changes in higher moments in the climate model, compared to standard Quantile Mapping where just the mean is assumed to change in future climate.

    .. note:: The method was originally developed with monthly data. Whilst it can be used for daily data, care needs to be taken when doing so. The method can also be seen as an additive QuantileDeltaMapping and the latter might be simpler to use for daily data.

    .. note:: As opposed to most other publications, Li et al. use a  4-parameter beta distribution (:py:data:`scipy.stats.beta`) for ``tas`` instead of a normal distribution. This can be slow for the fit at times. Consider modifying the ``distribution`` parameter for ``tas``.

    For precipitation a distribution or model is needed that accounts for mixed zero and positive value character. Default is a precipitation hurdle model (see :py:class:`ibicus.utils.gen_PrecipitationHurdleModel`). However also different ones are possible. :py:func:`for_precipitation` helps with the initialisation of different precipitation methods.


    **Reference:**

    - Li, H., Sheffield, J., and Wood, E. F. (2010), Bias correction of monthly precipitation and temperature fields from Intergovernmental Panel on Climate Change AR4 models using equidistant quantile matching, J. Geophys. Res., 115, D10101, doi:10.1029/2009JD012882.

    |br|
    **Usage information:**

    - Default settings exist for: ``["hurs", "pr", "psl", "rlds", "sfcWind", "tas", "tasmin", "tasmax"]``.

    - :py:func:`apply` requires: no additional arguments except ``obs``, ``cm_hist``, ``cm_future``.

    - Next to :py:func:`from_variable` a :py:func:`for_precipitation`-method exists to help you initialise the debiaser for :py:data:`pr`.

    - The debiaser has been developed for monthly data, however it works with data in any time specification (daily, monthly, etc.).

    |br|
    **Examples:**

    Initialising using :py:class:`from_variable`:

    >>> debiaser = ECDFM.from_variable("tas")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    Initialising using :py:class:`for_precipitation`:

    >>> debiaser = ECDFM.for_precipitation(model_type = "hurdle")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    |br|

    Attributes
    ----------
    distribution : Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel]
        Method used for the fit to the historical and future climate model outputs as well as the observations.
        Usually a distribution in ``scipy.stats.rv_continuous``, but can also be an empirical distribution as given by ``scipy.stats.rv_histogram`` or a more complex statistical model as wrapped by the :py:class:`ibicus.utils.StatisticalModel` class.
    cdf_threshold : float
        Threshold to round CDF-values away from zero and one. Default: ``1e-10``. |brr|

    censor_values_to_zero : bool
        Whether values below a censoring threshold shall be censored to zero. Only relevant for precipitation. Default: ``False``.
    censoring_threshold : float
        Threshold below which values shall be censored to zero if ``censor_values_to_zero = True``. Relevant mainly for precipitation. |br|
        If it is used (so ``censor_values_to_zero = True``) one needs to make sure that the distribution fits to censored data, knows the correct ``censoring_threshold`` and assumes all observations under the specified censoring_threshold are zero/censored. |br|
        If the standard for_precipitation and from_variable methods are used to construct the class this is ensured by default. However if this parameter is changed manually or own distributions for precipitation are specified problems can arise. |brr|

    running_window_mode : bool
        Whether DeltaChange is used in running window over the year to account for seasonality. If ``running_window_mode = False`` then DeltaChange is applied on the whole period. Default: ``False``.
    running_window_length : int
        Length of the running window in days: how many values are used to calculate the bias adjustment transformation. Only relevant if ``running_window_mode = True``. Default: ``91``.
    running_window_step_length : int
        Step length of the running window in days: how many values are bias adjusted inside the running window and by how far it is moved. Only relevant if ``running_window_mode = True``. Default: ``31``.

    running_window_mode_over_years_of_cm_future : bool
        Controls whether the methodology is applied on a running time window, running over the years of the future climate model. This helps to smooth discontinuities in the preserved trends. Default: ``False``.
    running_window_over_years_of_cm_future_length : int
        Length of the running window in years: how many years are used to define the future climate (default: ``31`` years). Only relevant if ``running_window_mode_over_years_of_cm_future = True``.
    running_window_over_years_of_cm_future_step_length : int
        Step length of the running window in years: how many years are bias adjusted inside the running window (default: ``9`` years). Only relevant if ``running_window_mode_over_years_of_cm_future = True``.

    variable : str
        Variable for which the debiasing is done. Default: "unknown".
    """

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
    cdf_threshold: float = attrs.field(
        default=1e-10, validator=attrs.validators.instance_of(float)
    )

    # Relevant for precipitation
    censor_values_to_zero: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    censoring_threshold: float = attrs.field(
        default=0.1 / 86400,
        validator=attrs.validators.instance_of(float),
        converter=float,
    )

    # Running window mode
    running_window_mode: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
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
        default=False, validator=attrs.validators.instance_of(bool)
    )
    running_window_over_years_of_cm_future_length: int = attrs.field(
        default=31,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )
    running_window_over_years_of_cm_future_step_length: int = attrs.field(
        default=1, validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)]
    )

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super()._from_variable(
            cls, variable, default_settings, experimental_default_settings, **kwargs
        )

    @classmethod
    def for_precipitation(
        cls,
        model_type: str = "hurdle",
        amounts_distribution: scipy.stats.rv_continuous = scipy.stats.gamma,
        censoring_threshold: float = 0.1 / 86400,
        hurdle_model_randomization: bool = True,
        hurdle_model_kwds_for_distribution_fit={"floc": 0, "fscale": None},
        **kwargs
    ):
        """
        Instanciates the class to a precipitation-debiaser. This allows granular setting of available precipitation models without needing to explicitly specify the precipitation censored model for example.

        Parameters
        ----------
        model_type : str
            One of ``["censored", "hurdle", "ignore_zeros"]``. Model type to be used. See :py:class:`ibicus.utils.gen_PrecipitationGammaLeftCensoredModel`, :py:class:`ibicus.utils.gen_PrecipitationHurdleModel` and :py:class:`ibicus.utils.gen_PrecipitationIgnoreZeroValuesModel` for more details.
        amounts_distribution : scipy.stats.rv_continuous
            Distribution used for precipitation amounts. For the censored model only :py:data:`scipy.stats.gamma` is possible.
        censoring_threshold : float
            The censoring-value if a censored precipitation model is used.
        hurdle_model_randomization : bool
            Whether when computing the cdf-values for a hurdle model randomization shall be used. See :py:class:`ibicus.utils.gen_PrecipitationHurdleModel` for more details.
        hurdle_model_kwds_for_distribution_fit : dict
            Dict of parameters used for the distribution fit inside a hurdle model. Default: ``{"floc": 0, "fscale": None} location of distribution is fixed at zero (``floc = 0``) to stabilise Gamma distribution fits in scipy.
        **kwargs:
            All other class attributes that shall be set and where the standard values shall be overwritten.

        """
        variable = pr

        method = map_standard_precipitation_method(
            model_type,
            amounts_distribution,
            censoring_threshold,
            hurdle_model_randomization,
            hurdle_model_kwds_for_distribution_fit,
        )
        parameters = {
            "distribution": method,
            "variable": variable.name,
            "censor_values_to_zero": True,
            "censoring_threshold": censoring_threshold,
        }
        return cls(**{**parameters, **kwargs})

    def apply_on_seasonal_and_future_window(
        self, obs: np.ndarray, cm_hist: np.ndarray, cm_future: np.ndarray, **kwargs
    ):

        if self.censor_values_to_zero:
            obs[obs < self.censoring_threshold] = 0
            cm_hist[cm_hist < self.censoring_threshold] = 0
            cm_future[cm_future < self.censoring_threshold] = 0

        fit_obs = self.distribution.fit(obs)
        fit_cm_hist = self.distribution.fit(cm_hist)
        fit_cm_future = self.distribution.fit(cm_future)

        quantile_cm_future = threshold_cdf_vals(
            self.distribution.cdf(cm_future, *fit_cm_future), self.cdf_threshold
        )

        bias_corrected_vals = (
            cm_future
            + self.distribution.ppf(quantile_cm_future, *fit_obs)
            - self.distribution.ppf(quantile_cm_future, *fit_cm_hist)
        )

        if self.censor_values_to_zero:
            bias_corrected_vals[bias_corrected_vals < self.censoring_threshold] = 0

        return bias_corrected_vals
