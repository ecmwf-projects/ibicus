# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from typing import Optional, Union

import attrs
import numpy as np

from ..utils import (
    RunningWindowOverDaysOfYear,
    check_time_information_and_raise_error,
    day_of_year,
    get_mask_for_unique_subarray,
    infer_and_create_time_arrays_if_not_given,
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
)
from ._debiaser import Debiaser

# ----- Default settings for debiaser ----- #
default_settings = {
    tas: {"delta_type": "additive"},
    pr: {"delta_type": "multiplicative"},
    tasmin: {"delta_type": "additive"},
    tasmax: {"delta_type": "additive"},
}
experimental_default_settings = {
    hurs: {"delta_type": "multiplicative"},
    psl: {"delta_type": "additive"},
    rlds: {"delta_type": "additive"},
    rsds: {"delta_type": "multiplicative"},
    sfcwind: {"delta_type": "multiplicative"},
}


# ----- Debiaser ----- #


@attrs.define(slots=False)
class LinearScaling(Debiaser):
    """
    |br| Implements debiasing via linear scaling based on Maraun 2016.

    Linear scaling corrects a climate model by the difference in the mean of observations and the mean of the climate model on the reference period, either additively or multiplicatively.


    The present day model bias is calculated and then either subtracted or divided from the future climate model data.

    Let :math:`x_{\\text{obs}}` be the observed timeseries :math:`x_{\\text{cm_hist}}` the simulated historical one and :math:`x_{\\text{cm_fut}}` the simulated future one (climate model historical and future run). Then additive linear scaling adjusts :math:`x_{\\text{cm_fut}}` as follows:

    .. math:: x_{\\text{cm_fut}} \\rightarrow x_{\\text{cm_fut}} - (\\bar x_{\\text{cm_hist}} - \\bar x_{\\text{obs}})

    and multiplicative scaling:

    .. math:: x_{\\text{cm_fut}} \\rightarrow x_{\\text{cm_fut}} \\cdot \\frac{\\bar x_{\\text{obs}}}{\\bar x_{\\text{cm_hist}}}.

    Here :math:`\\bar x` stands for the mean over all x-values.

    Multiplicative scaling is classically used for precipitation (`pr`) and additive scaling for temperature (`tas`). Additive scaling amounts to a simple mean bias correction, whilst multiplicative one adjusts both mean and variance, but keeps their ration constant (Maraun 2016).

    **References:**

    - Maraun, D. Bias Correcting Climate Change Simulations - a Critical Review. Curr Clim Change Rep 2, 211–220 (2016). https://doi.org/10.1007/s40641-016-0050-x

    |br|
    **Usage information:**

    - Default settings exist for: ``["hurs", "pr", "psl", "rlds", "rsds", "sfcWind", "tas", "tasmin", "tasmax"]``.

    - :py:func:`apply` requires: no additional arguments except ``obs``, ``cm_hist``, ``cm_future``.

    - The debiaser works with data in any time specification (daily, monthly, etc.).

    |br|
    **Examples:**

    >>> debiaser = DeltaChange.from_variable("tas")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    |br|

    Attributes
    ----------
    delta_type : str
        One of ``["additive", "multiplicative"]``. Determines whether additive or multiplicative scaling is used.

    running_window_mode : bool
        Iteration: Whether LinearScaling is used in running window mode to account for seasonalities. If ``running_window_mode = False`` then LinearScaling is applied on the whole period. Default: ``True``.
    running_window_length : int
        Iteration: Length of the running window in days: how many values are used to the debiased climate model values. Only relevant if ``running_window_mode = True``. Default: ``31``.
    running_window_step_length : int
        Iteration: Step length of the running window in days: how many values are debiased inside the running window. Only relevant if ``running_window_mode = True``. Default: ``1``.

    variable : str
        Variable for which the debiasing is used
    """

    delta_type: str = attrs.field(
        validator=attrs.validators.in_(["additive", "multiplicative"])
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

    def __attrs_post_init__(self):
        if self.running_window_mode:
            self.running_window = RunningWindowOverDaysOfYear(
                window_length_in_days=self.running_window_length,
                window_step_length_in_days=self.running_window_step_length,
            )

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super()._from_variable(
            cls, variable, default_settings, experimental_default_settings, **kwargs
        )

    def _apply_on_within_year_window(self, obs, cm_hist, cm_future):
        if self.delta_type == "additive":
            return cm_future - (np.mean(cm_hist) - np.mean(obs))
        elif self.delta_type == "multiplicative":
            return cm_future * (np.mean(obs) / np.mean(cm_hist))
        else:
            raise ValueError(
                'self.delta_type needs to be one of ["additive", "multiplicative"].'
            )

    def apply_location(
        self,
        obs: np.ndarray,
        cm_hist: np.ndarray,
        cm_future: np.ndarray,
        time_obs: Optional[np.ndarray] = None,
        time_cm_hist: Optional[np.ndarray] = None,
        time_cm_future: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.running_window_mode:
            if time_obs is None or time_cm_hist is None or time_cm_future is None:
                warnings.warn(
                    """LinearScaling runs without time-information for at least one of obs, cm_hist or cm_future.
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

            years_cm_future = year(time_cm_future)

            days_of_year_obs = day_of_year(time_obs)
            days_of_year_cm_hist = day_of_year(time_cm_hist)
            days_of_year_cm_future = day_of_year(time_cm_future)

            debiased_cm_future = np.empty_like(cm_future)

            # Iteration over year to account for seasonality
            for (
                window_center,
                indices_bias_corrected_values,
            ) in self.running_window.use(days_of_year_cm_future, years_cm_future):
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

                debiased_cm_future[
                    indices_bias_corrected_values
                ] = self._apply_on_within_year_window(
                    obs=obs[indices_window_obs],
                    cm_hist=cm_hist[indices_window_cm_hist],
                    cm_future=cm_future[indices_window_cm_future],
                )[
                    np.logical_and(
                        np.in1d(
                            indices_window_cm_future, indices_bias_corrected_values
                        ),
                        get_mask_for_unique_subarray(indices_window_cm_future),
                    )
                ]
            return debiased_cm_future
        else:
            return self._apply_on_within_year_window(obs, cm_hist, cm_future)
