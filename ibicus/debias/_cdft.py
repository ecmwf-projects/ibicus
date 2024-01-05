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

from ..utils import (
    RunningWindowOverYears,
    create_array_of_consecutive_dates,
    ecdf,
    iecdf,
    year,
)
from ..variables import (
    Variable,
    hurs,
    pr,
    psl,
    rlds,
    rsds,
    sfcwind,
    tas,
    tasmax,
    tasmin,
    tasrange,
    tasskew,
)
from ._running_window_debiaser import RunningWindowDebiaser

default_settings = {
    tas: {
        "delta_shift": "additive",
    },
    pr: {"delta_shift": "additive", "SSR": True},
    tasmin: {"delta_shift": "additive"},
    tasmax: {"delta_shift": "additive"},
}
experimental_default_settings = {
    hurs: {"delta_shift": "multiplicative"},
    psl: {"delta_shift": "additive"},
    rlds: {"delta_shift": "additive"},
    rsds: {"delta_shift": "multiplicative"},
    sfcwind: {"delta_shift": "multiplicative"},
    tasrange: {"delta_shift": "additive"},
    tasskew: {"delta_shift": "multiplicative"},
}


# ----- Debiaser ----- #


@attrs.define(slots=False)
class CDFt(RunningWindowDebiaser):
    """
    |br| Implements CDF-t based on Michelangeli et al. 2009, Vrac et al. 2012 and Famien et al. 2018, as well as Vrac et al. 2016 for precipitation.

    CDFt is a non-parametric quantile mapping method that attempts to be trend-preserving in all quantiles. CDFt applies a concatenation between a quantile mapping of future and historical climate model data and a quantile mapping of the future climate model with historical observations. It also includes a running window over the future period to account for changes in the simulated trend.


    Let cm refer to climate model output, obs to observations and hist/future to whether the data was collected from the reference period or is part of future projections.
    In this methodology, all cdfs are estimated empirically. Let :math:`F` therefore be an empirical cdf.
    The future climate projections :math:`x_{\\text{cm_fut}}` are then mapped using a QQ-mapping between :math:`F_{\\text{cm_fut}}` and :math:`F_{\\text{obs_fut}}`, with:

    .. math:: F_{\\text{obs_fut}} := F_{\\text{obs_hist}}(F^{-1}_{\\text{cm_hist}}(F_{\\text{cm_fut}})).

    This means that :math:`x_{\\text{cm_fut}}` is mapped using the following formula:

    .. math:: x_{\\text{cm_fut}} \\rightarrow F^{-1}_{\\text{obs_fut}}(F_{\\text{cm_fut}}(x_{\\text{cm_fut}})) = F^{-1}_{\\text{cm_fut}}(F_{\\text{cm_hist}}(F^{-1}_{\\text{obs_hist}}(F_{\\text{cm_fut}}(x_{\\text{cm_fut}}))))

    Because an empirical CDF will not be able to map values outside its fitted range, a delta shift is applied to the future and historical climate model prior to fitting empirical CDFs. This ensures that the data is approximately in the same range. This delta shift can be additive:

    .. math:: x_{\\text{cm_fut}} \\rightarrow x_{\\text{cm_fut}} + \\bar x_{\\text{obs}} - \\bar x_{\\text{cm_hist}}
    .. math:: x_{\\text{cm_hist}} \\rightarrow x_{\\text{cm_hist}} + \\bar x_{\\text{obs}} - \\bar x_{\\text{cm_hist}}

    or multiplicative:

    .. math:: x_{\\text{cm_fut}} \\rightarrow x_{\\text{cm_fut}} \\cdot \\frac{\\bar x_{\\text{obs}}}{\\bar x_{\\text{cm_hist}}}
    .. math:: x_{\\text{cm_hist}} \\rightarrow x_{\\text{cm_hist}} \\cdot \\frac{\\bar x_{\\text{obs}}}{\\bar x_{\\text{cm_hist}}}

    Here :math:`\\bar x` stands for the mean over all x-values.

    After this shift by the absolute or relative mean bias between cm_hist and obs are applied to both cm_fut and cm_hist, the cm_fut values are mapped as shown above using the QQ-mapping between :math:`F_{\\text{cm_fut}}` and :math:`F_{\\text{obs_fut}}`.

    - If ``SSR = True`` then Stochastic Singularity Removal (SSR) based on Vrac et al. 2016 is used to correct the precipitation occurrence in addition to amounts (default setting for ``pr``). All zero values are first replaced by uniform draws between 0 and a small threshold (the minimum positive value of observation and model data). Then CDFt-mapping is used and afterwards all observations under the threshold are set to zero again.
    - If ``running_window_mode_over_years_of_cm_future = True`` (default) then the method is used in a running window mode, running over the values of the future climate model. This helps to smooth discontinuities.
    - If ``running_window_mode_within_year = True`` (default) then the method is used in a running window mode, running over the year to account for seasonalities.
    - If ``apply_by_month = True`` (default: False) then CDF-t uses a running window within the year (`running_window_mode_within_year`) with a window length and step length of 31, so broadly applies it by month following Famien et al. 2018 to take seasonality into account. If  ``apply_by_month = False`` and ``running_window_mode_within_year = False`` the method is applied to the whole year.

    .. warning:: Currently only uneven sizes are allowed for window length and window step length. This allows symmetrical windows of the form [window_center - window length//2, window_center + window length//2] given an arbitrary window center.

    **References**:

    - Michelangeli, P.-A., Vrac, M., & Loukos, H. (2009). Probabilistic downscaling approaches: Application to wind cumulative distribution functions. In Geophysical Research Letters (Vol. 36, Issue 11). American Geophysical Union (AGU). https://doi.org/10.1029/2009gl038401
    - Famien, A. M., Janicot, S., Ochou, A. D., Vrac, M., Defrance, D., Sultan, B., & Noël, T. (2018). A bias-corrected CMIP5 dataset for Africa using the CDF-t method – a contribution to agricultural impact studies. In Earth System Dynamics (Vol. 9, Issue 1, pp. 313–338). Copernicus GmbH. https://doi.org/10.5194/esd-9-313-2018
    - Vrac, M., Drobinski, P., Merlo, A., Herrmann, M., Lavaysse, C., Li, L., & Somot, S. (2012). Dynamical and statistical downscaling of the French Mediterranean climate: uncertainty assessment. In Natural Hazards and Earth System Sciences (Vol. 12, Issue 9, pp. 2769–2784). Copernicus GmbH. https://doi.org/10.5194/nhess-12-2769-2012
    - Vrac, M., Noël, T., & Vautard, R. (2016). Bias correction of precipitation through Singularity Stochastic Removal: Because occurrences matter. In Journal of Geophysical Research: Atmospheres (Vol. 121, Issue 10, pp. 5237–5258). American Geophysical Union (AGU). https://doi.org/10.1002/2015jd024511

    |br|
    **Usage information:**

    - Default settings exist for: ``["hurs", "pr", "psl", "rlds", "rsds", "sfcWind", "tas", "tasmin", "tasmax", "tasrange", "tasskew"]``.

    - :py:func:`apply` requires: time arguments ``time_obs``, ``time_cm_hist``, and ``time_cm_future`` next to ``obs``, ``cm_hist`` and ``cm_future``. These are just 1d numpy-arrays of dates (multiple formats are possible as long as they as convertible to numpy or datetime dates) specifying the date for each value/timestep in ``obs``, ``cm_hist`` and ``cm_future``. If they are not specified they are inferred, assuming the first observation in all three observation/climate value arrays is on a 1st of January.

    - The debiaser has been developed for and assumes daily data, however application on data using other time specifications (monthly etc.) is possible by setting ``apply_by_month = False``, modifying the running window arguments and specifying the time arguments in :py:func:`apply`.

    |br|
    **Examples:**

    Running without dates (they are inferred assuming the first value in ``obs``, ``cm_hist`` and ``cm_future`` always corresponds to a January 1st):

    >>> debiaser = CDFt.from_variable("pr")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    Running with dates:

    >>> debiaser = CDFt.from_variable("pr")
    >>> debiaser.apply(obs, cm_hist, cm_future, time_obs = time_obs, time_cm_hist = time_cm_hist, time_cm_future = time_cm_future)

    |br|

    Attributes
    ----------
    SSR : bool
        If Stochastic Singularity Removal (SSR) following Vrac et al. 2016 is applied to adjust the number of zero values (only relevant for ``pr``).
    delta_shift : str
        One of ``["additive", "multiplicative", "no_shift"]``. Type of shift applied to the data prior to fitting empirical distributions.

    running_window_mode : bool
        Controls whether CDF-t is applied in a running window over the year to account for seasonality. Default: ``True``.
    running_window_length : int
        Length of the running window over the year in days (default: 31 days): the amount of days used to calculate the bias adjustment transformation. Only relevant if ``running_window_mode = True``.
    running_window_step_length : int
        Step length of the running window over the year in days (default 31 days): the amount of days that are bias adjusted/how far the running window is moved. Only relevant if ``running_window_mode = True``. |brr|

    running_window_mode_over_years_of_cm_future : bool
        Whether CDF-t is used in running window mode, running over the values of the future climate model to help smooth discontinuities. Default: ``True``.
    running_window_over_years_of_cm_future_length : int
        Length of the running window in years: how many values are used to calculate the empirical CDF. Only relevant if ``running_window_mode_over_years_of_cm_future = True``. Default: ``17``.
    running_window_over_years_of_cm_future_step_length : int
        Step length of the running window in years: how many values are bias adjusted inside the running window. Only relevant if ``running_window_mode_over_years_of_cm_future = True``. Default: ``9``.

    apply_by_month : bool
        Whether CDF-t is applied month by month (default) to account for seasonality. This is equivalent to a running window within the year with length 31 and step length 31. Default: ``Faöse``.

    variable : str
        Variable for which the debiasing is done. Default: ``"unknown"``.
    ecdf_method : str
        One of ``["kernel_density", "linear_interpolation", "step_function"]``. Method to calculate the empirical CDF. Default: ``"linear_interpolation"``.
    iecdf_method : str
        One of ``["inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased"]``. Method to calculate the inverse empirical CDF (empirical quantile function). Default: ``"linear"``.
    """

    # CDFt parameters
    SSR: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    delta_shift: str = attrs.field(
        default="additive",
        validator=attrs.validators.in_(["additive", "multiplicative", "no_shift"]),
    )

    # Iteration parameters
    apply_by_month: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )

    # Running window mode
    running_window_mode: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )
    running_window_length: int = attrs.field(
        default=31,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )
    running_window_step_length: int = attrs.field(
        default=31,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )

    # Running window mode over years
    running_window_mode_over_years_of_cm_future: bool = attrs.field(
        default=True, validator=attrs.validators.instance_of(bool)
    )
    running_window_over_years_of_cm_future_length: int = attrs.field(
        default=17, validator=attrs.validators.instance_of(int)
    )
    running_window_over_years_of_cm_future_step_length: int = attrs.field(
        default=9, validator=attrs.validators.instance_of(int)
    )

    # Calculation parameters
    ecdf_method: str = attrs.field(
        default="linear_interpolation",
        validator=attrs.validators.in_(
            ["kernel_density", "linear_interpolation", "step_function"]
        ),
    )
    iecdf_method: str = attrs.field(
        default="linear",
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

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if self.running_window_mode_over_years_of_cm_future:
            self.running_window_over_years_of_cm_future = RunningWindowOverYears(
                window_length_in_years=self.running_window_over_years_of_cm_future_length,
                window_step_length_in_years=self.running_window_over_years_of_cm_future_step_length,
            )

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super()._from_variable(
            cls, variable, default_settings, experimental_default_settings, **kwargs
        )

    # ----- Helpers: CDFt application -----#

    def _apply_CDFt_mapping(self, obs, cm_hist, cm_future):
        if self.delta_shift == "additive":
            shift = np.mean(obs) - np.mean(cm_hist)
            cm_hist = cm_hist + shift
            cm_future = cm_future + shift
        elif self.delta_shift == "multiplicative":
            shift = np.mean(obs) / np.mean(cm_hist)
            cm_hist = cm_hist * shift
            cm_future = cm_future * shift
        elif self.delta_shift == "no_shift":
            pass
        else:
            raise ValueError(
                'self.delta_shift needs to be one of ["additive", "multiplicative", "no_shift"]'
            )

        return iecdf(
            x=cm_future,
            p=ecdf(
                x=cm_hist,
                y=iecdf(
                    x=obs,
                    p=ecdf(x=cm_future, y=cm_future, method=self.ecdf_method),
                    method=self.iecdf_method,
                ),
                method=self.ecdf_method,
            ),
            method=self.iecdf_method,
        )

    @staticmethod
    def _get_threshold(obs, cm_hist, cm_future):
        positive_values = np.concatenate(
            [obs[obs > 0], cm_hist[cm_hist > 0], cm_future[cm_future > 0]]
        )
        return positive_values.min() if positive_values.size > 0 else 0

    @staticmethod
    def _randomize_zero_values_between_zero_and_threshold(x, threshold):
        return np.where(
            x == 0, np.random.uniform(low=0, high=threshold, size=x.size), x
        )

    @staticmethod
    def _set_values_below_threshold_to_zero(x, threshold):
        return np.where(x < threshold, 0, x)

    @staticmethod
    def _apply_SSR_steps_before_adjustment(obs, cm_hist, cm_future):
        threshold = CDFt._get_threshold(obs, cm_hist, cm_future)

        obs = CDFt._randomize_zero_values_between_zero_and_threshold(obs, threshold)
        cm_hist = CDFt._randomize_zero_values_between_zero_and_threshold(
            cm_hist, threshold
        )
        cm_future = CDFt._randomize_zero_values_between_zero_and_threshold(
            cm_future, threshold
        )
        return obs, cm_hist, cm_future, threshold

    @staticmethod
    def _apply_SSR_steps_after_adjustment(cm_future, threshold):
        cm_future = CDFt._set_values_below_threshold_to_zero(cm_future, threshold)
        return cm_future

    def _apply_debiasing_steps(
        self, obs: np.ndarray, cm_hist: np.ndarray, cm_future: np.ndarray
    ):
        # Precipitation
        if self.SSR:
            (
                obs,
                cm_hist,
                cm_future,
                threshold,
            ) = CDFt._apply_SSR_steps_before_adjustment(obs, cm_hist, cm_future)

        cm_future = self._apply_CDFt_mapping(obs, cm_hist, cm_future)

        # Precipitation
        if self.SSR:
            cm_future = CDFt._apply_SSR_steps_after_adjustment(cm_future, threshold)

        return cm_future

    def apply_on_window(
        self,
        obs: np.ndarray,
        cm_hist: np.ndarray,
        cm_future: np.ndarray,
        time_obs: np.ndarray = None,
        time_cm_hist: np.ndarray = None,
        time_cm_future: np.ndarray = None,
    ):
        if self.running_window_mode_over_years_of_cm_future:
            if time_cm_future is None:
                warnings.warn(
                    """CDFt runs without time-information for cm_future. This information is inferred, assuming the first observation is on a January 1st.""",
                    stacklevel=2,
                )
                time_cm_future = create_array_of_consecutive_dates(cm_future.size)

            years_cm_future = year(time_cm_future)

            debiased_cm_future = np.empty_like(cm_future)
            for (
                years_to_debias,
                years_in_window,
            ) in self.running_window_over_years_of_cm_future.use(years_cm_future):
                mask_years_in_window = RunningWindowOverYears.get_if_in_chosen_years(
                    years_cm_future, years_in_window
                )
                mask_years_to_debias = RunningWindowOverYears.get_if_in_chosen_years(
                    years_cm_future, years_to_debias
                )
                mask_years_in_window_to_debias = (
                    RunningWindowOverYears.get_if_in_chosen_years(
                        years_cm_future[mask_years_in_window], years_to_debias
                    )
                )

                debiased_cm_future[mask_years_to_debias] = self._apply_debiasing_steps(
                    obs=obs,
                    cm_hist=cm_hist,
                    cm_future=cm_future[mask_years_in_window],
                )[mask_years_in_window_to_debias]

            return debiased_cm_future

        else:
            return self._apply_debiasing_steps(obs, cm_hist, cm_future)
