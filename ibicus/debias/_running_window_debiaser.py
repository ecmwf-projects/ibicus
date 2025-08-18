import warnings
from abc import abstractmethod
from typing import Optional

import attrs
import numpy as np

from ..utils import (
    RunningWindowOverDaysOfYear,
    RunningWindowOverYears,
    check_time_information_and_raise_error,
    create_array_of_consecutive_dates,
    day_of_year,
    infer_and_create_time_arrays_if_not_given,
    year,
)
from ._debiaser import Debiaser


@attrs.define(slots=False, kw_only=True)
class SeasonalRunningWindowDebiaser(Debiaser):
    """
    A generic debiaser meant for subclassing which applies methods in a running window over the year to account for seasonality. Provides functionality for individual debiasers and a unified interface to apply bias adjustment.

    Have a look at the :py:class:`Debiaser` parent class for the general structure. In order to subclass the :py:class:`SeasonalRunningWindowDebiaser`-class, the proposed debiaser needs to implement the :py:func:`from_variable` and :py:func:`apply_on_seasonal_window` functions:

    - :py:func:`apply_on_window`: this applies an initialised debiaser at one location and in one window. Arguments are 1d-vectors of obs, cm_hist, and cm_future representing observations, and climate model values during the reference (cm_hist) and future period (cm_future) as well as time-information ``time_obs``, ``time_cm_hist`` and ``time_cm_future``) as 1d-numpy arrays corresponding to ``obs``, ``cm_hist`` ``and cm_future``. Additionally time a``kwargs`` passed to the debiaser :py:func:`apply`-function are passed down to the :py:func:`apply_location`-function.

    - :py:func:`from_variable`: initialises a debiaser with default arguments given a climatic variable either as ``str`` or member of the :py:class:`Variable`-class. ``kwargs`` are meant to overwrite default arguments for this variable. Given a `dict` of default arguments: with variables of the :py:class:`Variable` class as keys and `dict` of default arguments as values the :py:func:`_from_variable`-function can be used.

    The :py:func:`apply` function, maps the debiaser's :py:func:`apply_window` function over windows and locations. This allows to always initialise and apply debiasers follows:

    >>> debiaser = LinearScaling.from_variable("tas", running_window_mode = True) # LinearScaling is a child-class of Debiaser
    >>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future, time_obs = time_obs, time_cm_hist = time_cm_hist, time_cm_future = time_cm_future)

    Attributes
    ----------
    variable : str
        Variable that is meant to be debiased, by an initialisation of the debiaser. Default: ``"unknown"``.
    reasonable_physical_range : Optional[list]
        Reasonable physical range of the variable to debias in the form ``[lower_bound, upper_bound]``. It is checked against and warnings are raise if values fall outside the range. Default: ``None``.

    running_window_mode : bool
        Whether the bias adjustment method is used in running window over the year to account for seasonality. If ``running_window_mode = False`` then the method is applied on the whole period. Default: ``False``.
    running_window_length : int
        Length of the running window in days: how many values are used to calculate the bias adjustment transformation. Only relevant if ``running_window_mode = True``. Default: ``31``.
    running_window_step_length : int
        Step length of the running window in days: how many values are bias adjusted inside the running window and by how far it is moved. Only relevant if ``running_window_mode = True``. Default: ``1``.
    """

    # Running window mode
    running_window_mode: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    running_window_length: int = attrs.field(
        default=31,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )
    running_window_step_length: int = attrs.field(
        default=1,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )

    def __attrs_post_init__(self):
        if self.running_window_step_length > self.running_window_length:
            raise ValueError(
                "Running window step length (how many days are bias adjusted / how far the window is moved) needs to be equal or smaller than the running window length (how many days are used for calculating the bias adjustment transformation)"
            )
        if self.running_window_mode:
            self.running_window = RunningWindowOverDaysOfYear(
                window_length_in_days=self.running_window_length,
                window_step_length_in_days=self.running_window_step_length,
            )

    @abstractmethod
    def apply_on_seasonal_window(obs, cm_hist, cm_future, **kwargs):
        """
        Applies the debiaser at one location and on one window.

        Parameters
        ----------
        obs : np.ndarray
            1-dimensional numpy array of observations of the meteorological variable at one location.
        cm_hist : np.ndarray
            1-dimensional numpy array of values of the historical climate model run (run during the same or a similar period as observations) at one location.
        cm_future : np.ndarray
            1-dimensional numpy array of values of a climate model to debias (future run) at one location.

        Returns
        -------
        np.ndarray
            1-dimensional numpy array containing the debiased climate model values for the future run (cm_future).
        """
        pass

    def apply_location(
        self,
        obs: np.ndarray,
        cm_hist: np.ndarray,
        cm_future: np.ndarray,
        time_obs: Optional[np.ndarray] = None,
        time_cm_hist: Optional[np.ndarray] = None,
        time_cm_future: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies the debiaser at one location.

        Parameters
        ----------
        obs : np.ndarray
            1-dimensional numpy array of observations of the meteorological variable at one location.
        cm_hist : np.ndarray
            1-dimensional numpy array of values of the historical climate model run (run during the same or a similar period as observations) at one location.
        cm_future : np.ndarray
            1-dimensional numpy array of values of a climate model to debias (future run) at one location.

        Returns
        -------
        np.ndarray
            1-dimensional numpy array containing the debiased climate model values for the future run (cm_future).
        """

        if self.running_window_mode:
            if time_obs is None or time_cm_hist is None or time_cm_future is None:
                warnings.warn(
                    """Debiaser runs without time-information for at least one of obs, cm_hist or cm_future.
                        This information is inferred, assuming the first observation is on a January 1st. Observations are chunked according to the assumed time information.
                        This might lead to slight numerical differences to the run with time information, however the debiasing is not fundamentally changed.""",
                    stacklevel=2,
                )

                (
                    time_obs,
                    time_cm_hist,
                    time_cm_future,
                ) = infer_and_create_time_arrays_if_not_given(
                    obs, cm_hist, cm_future, time_obs, time_cm_hist, time_cm_future
                )

            check_time_information_and_raise_error(
                obs, cm_hist, cm_future, time_obs, time_cm_hist, time_cm_future
            )

            days_of_year_obs = day_of_year(time_obs)
            days_of_year_cm_hist = day_of_year(time_cm_hist)
            days_of_year_cm_future = day_of_year(time_cm_future)

            debiased_cm_future = np.empty_like(cm_future)

            # Iteration over year to account for seasonality
            for (
                window_center,
                indices_bias_corrected_values,
            ) in self.running_window.use(days_of_year_cm_future):
                indices_window_obs = self.running_window.get_indices_vals_in_window(
                    days_of_year_obs, window_center
                )
                indices_window_cm_hist = self.running_window.get_indices_vals_in_window(
                    days_of_year_cm_hist, window_center
                )
                indices_window_cm_future = (
                    self.running_window.get_indices_vals_in_window(
                        days_of_year_cm_future, window_center
                    )
                )

                mask_vals_to_adjust_in_window = (
                    RunningWindowOverDaysOfYear.get_mask_vals_to_adjust_in_window(
                        indices_window_cm_future, indices_bias_corrected_values
                    )
                )

                debiased_cm_future[indices_bias_corrected_values] = (
                    self.apply_on_seasonal_window(
                        obs=obs[indices_window_obs],
                        cm_hist=cm_hist[indices_window_cm_hist],
                        cm_future=cm_future[indices_window_cm_future],
                        time_obs=time_obs[indices_window_obs],
                        time_cm_hist=time_cm_hist[indices_window_cm_hist],
                        time_cm_future=time_cm_future[indices_window_cm_future],
                    )[mask_vals_to_adjust_in_window]
                )
            return debiased_cm_future

        else:
            return self.apply_on_seasonal_window(
                obs,
                cm_hist,
                cm_future,
                time_obs=time_obs,
                time_cm_hist=time_cm_hist,
                time_cm_future=time_cm_future,
            )


@attrs.define(slots=False, kw_only=True)
class SeasonalAndFutureRunningWindowDebiaser(SeasonalRunningWindowDebiaser):
    """
    A generic debiaser meant for subclassing which applies methods both in a running window over the year to account for seasonality and in a running window over the future period to smooth trends. Provides functionality for individual debiasers and a unified interface to apply bias adjustment.

    Have a look at the :py:class:`Debiaser` parent class for the general structure. In order to subclass the :py:class:`SeasonalAndFutureRunningWindowDebiaser`-class, the proposed debiaser needs to implement the :py:func:`from_variable` and :py:func:`apply_on_seasonal_and_future_window` functions:

    - :py:func:`apply_on_double_window`: this applies an initialised debiaser at one location and in one window. Arguments are 1d-vectors of obs, cm_hist, and cm_future representing observations, and climate model values during the reference (cm_hist) and future period (cm_future) as well as time-information ``time_obs``, ``time_cm_hist`` and ``time_cm_future``) as 1d-numpy arrays corresponding to ``obs``, ``cm_hist`` ``and cm_future``. Additionally time a``kwargs`` passed to the debiaser :py:func:`apply`-function are passed down to the :py:func:`apply_location`-function.

    - :py:func:`from_variable`: initialises a debiaser with default arguments given a climatic variable either as ``str`` or member of the :py:class:`Variable`-class. ``kwargs`` are meant to overwrite default arguments for this variable. Given a `dict` of default arguments: with variables of the :py:class:`Variable` class as keys and `dict` of default arguments as values the :py:func:`_from_variable`-function can be used.

    The :py:func:`apply` function, maps the debiaser's :py:func:`apply_window` function over windows and locations. This allows to always initialise and apply debiasers follows:

    >>> debiaser = QuantileMapping.from_variable("tas", running_window_mode_over_years_of_cm_future = True) # QuantileMapping is a child-class of Debiaser
    >>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future, time_obs = time_obs, time_cm_hist = time_cm_hist, time_cm_future = time_cm_future)

    Attributes
    ----------
    variable : str
        Variable that is meant to be debiased, by an initialisation of the debiaser. Default: ``"unknown"``.
    reasonable_physical_range : Optional[list]
        Reasonable physical range of the variable to debias in the form ``[lower_bound, upper_bound]``. It is checked against and warnings are raise if values fall outside the range. Default: ``None``.

    running_window_mode_over_years_of_cm_future : bool
        Controls whether the methodology is applied on a running time window, running over the years of the future climate model. This helps to smooth discontinuities in the preserved trends. Default: ``False``.
    running_window_over_years_of_cm_future_length : int
        Length of the running window in years: how many years are used to define the future climate (default: ``31`` years). Only relevant if ``running_window_mode_over_years_of_cm_future = True``.
    running_window_over_years_of_cm_future_step_length : int
        Step length of the running window in years: how many years are bias adjusted inside the running window (default: ``9`` years). Only relevant if ``running_window_mode_over_years_of_cm_future = True``.
    """

    # Running window over future
    running_window_mode_over_years_of_cm_future: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    running_window_over_years_of_cm_future_length: int = attrs.field(
        default=31,
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
    )
    running_window_over_years_of_cm_future_step_length: int = attrs.field(
        default=9, validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)]
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if self.running_window_mode_over_years_of_cm_future:
            self.running_window_over_years_of_cm_future = RunningWindowOverYears(
                window_length_in_years=self.running_window_over_years_of_cm_future_length,
                window_step_length_in_years=self.running_window_over_years_of_cm_future_step_length,
            )

    @abstractmethod
    def apply_on_seasonal_and_future_window(obs, cm_hist, cm_future, **kwargs):
        """
        Applies the debiaser at one location and on one window.

        Parameters
        ----------
        obs : np.ndarray
            1-dimensional numpy array of observations of the meteorological variable at one location.
        cm_hist : np.ndarray
            1-dimensional numpy array of values of the historical climate model run (run during the same or a similar period as observations) at one location.
        cm_future : np.ndarray
            1-dimensional numpy array of values of a climate model to debias (future run) at one location.

        Returns
        -------
        np.ndarray
            1-dimensional numpy array containing the debiased climate model values for the future run (cm_future).
        """
        pass

    def apply_on_seasonal_window(
        self,
        obs: np.ndarray,
        cm_hist: np.ndarray,
        cm_future: np.ndarray,
        time_obs: np.ndarray,
        time_cm_hist: np.ndarray,
        time_cm_future: np.ndarray,
    ):

        if self.running_window_mode_over_years_of_cm_future:
            if time_cm_future is None:
                warnings.warn(
                    """QuantileDeltaMapping runs without time-information for cm_future. This information is inferred, assuming the first observation is on a January 1st.""",
                    stacklevel=2,
                )
                time_cm_future = create_array_of_consecutive_dates(cm_future.size)

            years_cm_future = year(time_cm_future)

            debiased_cm_future = np.empty_like(cm_future)

            # Iteration over years of cm_future to account for trends
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

                debiased_cm_future[mask_years_to_debias] = (
                    self.apply_on_seasonal_and_future_window(
                        obs=obs,
                        cm_hist=cm_hist,
                        cm_future=cm_future[mask_years_in_window],
                        time_obs=time_obs,
                        time_cm_hist=time_cm_hist,
                        time_cm_future=time_cm_future[mask_years_in_window],
                    )[mask_years_in_window_to_debias]
                )

            return debiased_cm_future

        else:
            return self.apply_on_seasonal_and_future_window(
                obs=obs,
                cm_hist=cm_hist,
                cm_future=cm_future,
                time_obs=time_obs,
                time_cm_hist=time_cm_hist,
                time_cm_future=time_cm_future,
            )
