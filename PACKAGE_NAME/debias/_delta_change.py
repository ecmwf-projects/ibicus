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

from ..variables import (
    Precipitation,
    Temperature,
    Variable,
    map_variable_str_to_variable_class,
)
from ._debiaser import Debiaser

default_settings = {
    Temperature: {"delta_type": "additive"},
    Precipitation: {"delta_type": "multiplicative"},
}


@attrs.define
class DeltaChange(Debiaser):
    """
    Class DeltaChange representing debiasing via so-called delta change scaling following Maraun 2016 as reference.

    This is technically not a debiasing of a climate model because the future climate model output gets not directly transformed. Instead it only uses the models capturing of climate change to modify historical observations.

    Let :math:`y` be the observed timeseries :math:`x_{hist}` the simulated historical one and :math:`x_{fut}` the simulated future one (climate model historical and future run). Then in delta change a timeseries of future climate is generated as:

    .. math::  y +  - (\\text{mean}(x_{fut}) - \\text{mean}(x_{hist}))

    and for multiplicative change:

    .. math:: y * \\frac{\\text{mean}(x_{fut})}{\\text{mean}(x_{hist})}.


    Multiplicative change is classically used for precipitation and additive scaling for temperature.

    References:
    Maraun, D. Bias Correcting Climate Change Simulations - a Critical Review. Curr Clim Change Rep 2, 211–220 (2016). https://doi.org/10.1007/s40641-016-0050-x

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
        Debiaser.check_inputs(obs, cm_hist, cm_future)

        output = Debiaser.map_over_locations(
            func=self.apply_location, output_size=obs.shape, obs=obs, cm_hist=cm_hist, cm_future=cm_future
        )
        return output
