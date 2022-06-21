# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
linear_scaling MODULE - implements debiasing using linear scaling.
"""

import warnings

import numpy as np

from .debiaser import Debiaser

standard_delta_types = {
    "temp": "additive",
    "pr": "multiplicative",
    "temp": "additive",
    "temp": "additive",
}


class LinearScaling(Debiaser):
    """
    Class LinearScaling representing debiasing via so-called linear scaling following Maraun 2016 as reference.

    In linear scaling the present day model bias is either subtracted or divided from the future climate model time-series.
    Let :math:`y` be the observed timeseries :math:`x_{hist}` the simulated historical one and :math:`x_{fut}` the simulated future one (climate model historical and future run). Then additive linear scaling adjusts :math:`x_{fut}` as follows:

    .. math:: x_{fut} \\rightarrow x_{fut} - (\\text{mean}(x_{hist}) - \\text{mean}(y))

    and multiplicative scaling:

    .. math:: x_{fut} \\rightarrow x_{fut} * \\frac{\\text{mean}(y)}{\\text{mean}(x_{hist})}.


    Multiplicative scaling is hereby classically used for precipitation and additive scaling for temperature. Additive scaling amounts to a simple mean bias correction, whilst multiplicative one adjusts both mean and variance, but keeps their ration constant (Maraun 2016).


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

    def __init__(self, variable: str = None, delta_type: str = None):

        # Checks
        if variable is not None:
            if variable not in standard_delta_types.keys():
                raise ValueError(
                    "variable needs to be one of %s" % standard_delta_types.keys()
                )

        if delta_type is not None:
            if delta_type not in ["additive", "multiplicative"]:
                raise ValueError(
                    'delta_type needs to be one of ["additive", "multiplicative"].'
                )

        # Parameter setting
        if variable is not None:
            if delta_type is not None:
                if delta_type != standard_delta_types.get(variable):
                    warnings.warn(
                        "Given delta type for variable is different from standard one."
                    )
                self.variable = variable
                self.delta_type = delta_type
            else:
                self.variable = variable
                self.delta_type = standard_delta_types.get(variable)
        else:
            if delta_type is not None:
                self.variable = "unknown"
                self.delta_type = delta_type
            else:
                raise ValueError(
                    "At least one of variable, delta_type needs to be specified."
                )

    def apply_location(
        self, obs: np.ndarray, cm_hist: np.ndarray, cm_future: np.ndarray
    ) -> np.ndarray:
        """Applies linear scaling at one location and returns the debiased timeseries."""
        if self.delta_type == "additive":
            return cm_future - (np.mean(cm_hist) - np.mean(obs))
        elif self.delta_type == "multiplicative":
            return cm_future * (np.mean(obs) / np.mean(cm_hist))
        else:
            raise ValueError(
                'self.delta_type needs to be one of ["additive", "multiplicative"].'
            )
