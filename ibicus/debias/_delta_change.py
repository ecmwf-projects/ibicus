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

from ..utils import get_library_logger
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
class DeltaChange(Debiaser):
    """
    |br| Implements 'delta change' method based on Maraun 2016 as reference.

    This is technically not a bias correction method because the future climate model output is not transformed. Instead, the delta change method applies the climate change trend from the model to historical observations, therefore generating modified observations rather than a modified model output.
    So the output by :py:func:`apply` from this method has the same number of timesteps as the obs data, and not the same number as cm_fut like other debiasers.

    Let :math:`x_{\\text{obs}}` be the observed timeseries :math:`x_{\\text{cm_hist}}` the simulated historical one and :math:`x_{\\text{cm_fut}}` the simulated future one (climate model historical and future run).
    For an additive change a future timeseries is generated as:

    .. math::  x_{\\text{obs}} +  (\\bar x_{\\text{cm_fut}} - \\bar x_{\\text{cm_hist}})

    and for multiplicative change:

    .. math:: x_{\\text{obs}} \\cdot \\frac{\\bar x_{\\text{cm_fut}}}{\\bar x_{\\text{cm_hist}}}.

    Here :math:`\\bar x` stands for the mean over all x-values.

    Multiplicative change is typically used for precipitation and additive scaling for temperature.

    **References**:

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
            return obs + (np.mean(cm_future) - np.mean(cm_hist))
        elif self.delta_type == "multiplicative":
            return obs * (np.mean(cm_future) / np.mean(cm_hist))
        else:
            raise ValueError(
                'self.delta_type needs to be one of ["additive", "multiplicative"].'
            )

    def apply(
        self,
        obs,
        cm_hist,
        cm_future,
        progressbar=True,
        parallel=False,
        nr_processes=4,
        failsafe=False,
        **kwargs
    ):
        logger = get_library_logger()
        logger.info("----- Running debiasing for variable: %s -----" % self.variable)

        obs, cm_hist, cm_future = self._check_inputs_and_convert_if_possible(
            obs, cm_hist, cm_future
        )

        if parallel:
            output = Debiaser.parallel_map_over_locations(
                self.apply_location,
                output_size=obs.shape,
                obs=obs,
                cm_hist=cm_hist,
                cm_future=cm_future,
                nr_processes=nr_processes,
                failsafe=failsafe,
                **kwargs,
            )
        else:
            output = Debiaser.map_over_locations(
                self.apply_location,
                output_size=obs.shape,
                obs=obs,
                cm_hist=cm_hist,
                cm_future=cm_future,
                progressbar=progressbar,
                failsafe=failsafe,
                **kwargs,
            )

        self._check_output(output)

        return output
