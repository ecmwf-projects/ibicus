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

from ..variables import Variable, pr, tas
from ._debiaser import Debiaser

default_settings = {
    tas: {"delta_type": "additive"},
    pr: {"delta_type": "multiplicative"},
}


@attrs.define
class DeltaChange(Debiaser):
    """
    Class DeltaChange representing debiasing via so-called delta change scaling following Maraun 2016 as reference.

    This is technically not a debiasing of a climate model because the future climate model output gets not directly transformed. Instead it only uses the models capturing of climate change to modify historical observations.

    Let :math:`x_{\\text{obs}}` be the observed timeseries :math:`x_{\\text{cm_hist}}` the simulated historical one and :math:`x_{\\text{cm_fut}}` the simulated future one (climate model historical and future run). Then in delta change a timeseries of future climate is generated as:

    .. math::  x_{\\text{obs}} +  (\\bar x_{\\text{cm_fut}} - \\bar x_{\\text{cm_hist}})

    and for multiplicative change:

    .. math:: x_{\\text{obs}} * \\frac{\\bar x_{\\text{cm_fut}}}{\\bar x_{\\text{cm_hist}}}.

    Here :math:`\\bar x` stands for the mean over all x-values.

    Multiplicative change is classically used for precipitation and additive scaling for temperature.

    **References**:

    - Maraun, D. Bias Correcting Climate Change Simulations - a Critical Review. Curr Clim Change Rep 2, 211–220 (2016). https://doi.org/10.1007/s40641-016-0050-x

    ...

    Attributes
    ----------
    variable : str
        Variable for which the debiasing is used
    delta_type : str
        One of ["additive", "multiplicative"]. Determines whether additive or multiplicative scaling is used.


    Methods
    -------
    apply(obs: np.ndarray, cm_hist: np.ndarray, cm_future: np.ndarray) -> np.ndarray
        Applies linear scaling at all given locations on a grid and returns the the debiased timeseries.
    """

    delta_type: str = attrs.field(validator=attrs.validators.in_(["additive", "multiplicative"]))

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super().from_variable(cls, variable, default_settings, **kwargs)

    def apply_location(self, obs: np.ndarray, cm_hist: np.ndarray, cm_future: np.ndarray) -> np.ndarray:
        """Applies delta change at one location and returns the debiased timeseries."""
        if self.delta_type == "additive":
            return obs + (np.mean(cm_future) - np.mean(cm_hist))
        elif self.delta_type == "multiplicative":
            return obs * (np.mean(cm_future) / np.mean(cm_hist))
        else:
            raise ValueError('self.delta_type needs to be one of ["additive", "multiplicative"].')

    def apply(self, obs, cm_hist, cm_future):
        print("----- Running debiasing -----")
        Debiaser.check_inputs_and_convert_if_possible(obs, cm_hist, cm_future)

        output = Debiaser.map_over_locations(
            func=self.apply_location, output_size=obs.shape, obs=obs, cm_hist=cm_hist, cm_future=cm_future
        )
        return output
