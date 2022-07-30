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

from ..variables import Variable, map_variable_str_to_variable_class, pr, tas
from ._debiaser import Debiaser

default_settings = {
    tas: {"delta_type": "additive"},
    pr: {"delta_type": "multiplicative"},
}


@attrs.define
class LinearScaling(Debiaser):
    """
    Class LinearScaling representing debiasing via so-called linear scaling following Maraun 2016 as reference.

    In linear scaling the present day model bias is either subtracted or divided from the future climate model time-series.
    Let :math:`y` be the observed timeseries :math:`x_{hist}` the simulated historical one and :math:`x_{fut}` the simulated future one (climate model historical and future run). Then additive linear scaling adjusts :math:`x_{fut}` as follows:

    .. math:: x_{fut} \\rightarrow x_{fut} - (\\text{mean}(x_{hist}) - \\text{mean}(y))

    and multiplicative scaling:

    .. math:: x_{fut} \\rightarrow x_{fut} * \\frac{\\text{mean}(y)}{\\text{mean}(x_{hist})}.


    Multiplicative scaling is classically used for precipitation and additive scaling for temperature. Additive scaling amounts to a simple mean bias correction, whilst multiplicative one adjusts both mean and variance, but keeps their ration constant (Maraun 2016).


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
        return super().from_variable(cls, default_settings, variable, **kwargs)

    def apply_location(self, obs: np.ndarray, cm_hist: np.ndarray, cm_future: np.ndarray) -> np.ndarray:
        """Applies linear scaling at one location and returns the debiased timeseries."""
        if self.delta_type == "additive":
            return cm_future - (np.mean(cm_hist) - np.mean(obs))
        elif self.delta_type == "multiplicative":
            return cm_future * (np.mean(obs) / np.mean(cm_hist))
        else:
            raise ValueError('self.delta_type needs to be one of ["additive", "multiplicative"].')
