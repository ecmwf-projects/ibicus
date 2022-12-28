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
    variable : str
        Variable for which the debiasing is used
    """

    delta_type: str = attrs.field(
        validator=attrs.validators.in_(["additive", "multiplicative"])
    )

    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super()._from_variable(
            cls, variable, default_settings, experimental_default_settings, **kwargs
        )

    def apply_location(
        self, obs: np.ndarray, cm_hist: np.ndarray, cm_future: np.ndarray
    ) -> np.ndarray:
        if self.delta_type == "additive":
            return cm_future - (np.mean(cm_hist) - np.mean(obs))
        elif self.delta_type == "multiplicative":
            return cm_future * (np.mean(obs) / np.mean(cm_hist))
        else:
            raise ValueError(
                'self.delta_type needs to be one of ["additive", "multiplicative"].'
            )
