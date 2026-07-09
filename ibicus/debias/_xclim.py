from typing import Union

import attrs
import numpy as np
import scipy
import scipy.stats

from ..utils import (
    PrecipitationHurdleModelGamma,
    StatisticalModel,
    quantile_map_non_parametically_with_constant_extrapolation,
    threshold_cdf_vals,
)
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
from ._running_window_debiaser import RunningWindowDebiaser
import xarray as xr
from xclim.sdba import Grouper
from xclim.sdba._adjustment import dqm_train, eqm_train, dqm_adjust, qm_adjust
from xclim.sdba.utils import equally_spaced_nodes

# ----- Default settings for debiaser ----- #
default_settings = {
    tas: {
        "kind": "additive",
    },
    pr: {
        "kind": "multiplicative",
    },
}
experimental_default_settings = {
    hurs: {
        "kind": "multiplicative",
    },
    psl: {
        "kind": "additive",
    },
    rlds: {
        "kind": "additive",
    },
    sfcwind: {
        "kind": "multiplicative",
    },
    tasmin: {
        "kind": "additive",
    },
    tasmax: {
        "kind": "additive",
    },
}

# ----- Debiaser ----- #


@attrs.define(slots=False)
class XclimQuantileMapping(RunningWindowDebiaser):
    """
    |br| Implements (detrended) Quantile Mapping based on Cannon et al. 2015 and Maraun 2016.

    Empirical quantile mapping maps every quantile of the climate model distribution to the corresponding quantile in observations during the reference period. Optionally, additive or multiplicative detrending of the mean can be applied to make the method trend preserving in the mean. Most methods build on quantile mapping.


    Let cm refer to climate model output, obs to observations and hist/future to whether the data was collected from the reference period or is part of future projections.
    Let :math:`F` be a CDF. The future climate projections :math:`x_{\\text{cm_fut}}` are then mapped using a QQ-mapping between :math:`F_{\\text{cm_hist}}` and :math:`F_{\\text{obs}}`, so:

    .. math:: x_{\\text{cm_fut}} \\rightarrow F^{-1}_{\\text{obs}}(F_{\\text{cm_hist}}(x_{\\text{cm_fut}}))

    If detrended quantile mapping is used then :math:`x_{\\text{cm_fut}}` is first rescaled and then the mapped value is scaled back either additively or multiplicatively. That means for additive detrending:

    .. math:: x_{\\text{cm_fut}} \\rightarrow F^{-1}_{\\text{obs}}(F_{\\text{cm_hist}}(x_{\\text{cm_fut}} + \\bar x_{\\text{cm_hist}} - \\bar x_{\\text{cm_fut}})) + \\bar x_{\\text{cm_fut}} - \\bar x_{\\text{cm_hist}}

    and for multiplicative detrending.

    .. math:: x_{\\text{cm_fut}} \\rightarrow F^{-1}_{\\text{obs}}\\left(F_{\\text{cm_hist}}\\left(x_{\\text{cm_fut}} \\cdot \\frac{\\bar x_{\\text{cm_hist}}}{\\bar x_{\\text{cm_fut}}}\\right)\\right) \\cdot \\frac{\\bar x_{\\text{cm_fut}}}{\\bar x_{\\text{cm_hist}}}

    Here :math:`\\bar x_{\\text{cm_fut}}` designs the mean of :math:`x_{\\text{cm_fut}}` and similar for :math:`x_{\\text{cm_hist}}`.
    Detrended Quantile Mapping accounts for changes in the projected values and is thus trend-preserving in the mean.

    **References**:

    - Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes? In Journal of Climate (Vol. 28, Issue 17, pp. 6938–6959). American Meteorological Society. https://doi.org/10.1175/jcli-d-14-00754.1
    - Maraun, D. (2016). Bias Correcting Climate Change Simulations - a Critical Review. In Current Climate Change Reports (Vol. 2, Issue 4, pp. 211–220). Springer Science and Business Media LLC. https://doi.org/10.1007/s40641-016-0050-x

    |br|
    **Usage information:**

    - Default settings exist for: ``["hurs", "pr", "psl", "rlds", "sfcWind", "tas", "tasmin", "tasmax"]``.

    - :py:func:`apply` requires: no additional arguments except ``obs``, ``cm_hist``, ``cm_future``.

    - Next to :py:func:`from_variable` a :py:func:`for_precipitation`-method exists to help you initialise the debiaser for :py:data:`pr`.

    |br|
    **Examples:**

    Initialising using :py:class:`from_variable`:

    >>> debiaser = QuantileMapping.from_variable("tas")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    |br|

    Attributes
    ----------
    kind: str
        One of ``["additive", "multiplicative"]``.
    detrending : bool
        Whether to apply detrending or not.
    nquantiles : int
        Number of quantiles to use.
    running_window_mode : bool
        Whether QuantileMapping is used in running window over the year to account for seasonality. If ``running_window_mode = False`` then QuantileMapping is applied on the whole period. Default: ``False``.
    running_window_length : int
        Length of the running window in days: how many values are used to calculate the bias adjustment transformation. Only relevant if ``running_window_mode = True``. Default: ``31``.
    running_window_step_length : int
        Step length of the running window in days: how many values are bias adjusted inside the running window and by how far it is moved. Only relevant if ``running_window_mode = True``. Default: ``1``.
    variable : str
        Variable for which the debiasing is done. Default: ``"unknown"``.
    """
    kind: str = attrs.field(
        default="additive",
        validator=attrs.validators.in_(["additive", "multiplicative"]),
    )
    nquantiles: int = attrs.field(
        default=50,
        validator=attrs.validators.le(100)
    )
    detrending: bool = attrs.field(
        default=True,
        validator=attrs.validators.in_([True, False]),
    )

    # ----- Constructors -----
    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super()._from_variable(
            cls, variable, default_settings, experimental_default_settings, **kwargs
        )

    # ----- Helpers -----
    def apply_on_window(self, obs, cm_hist, cm_future, **kwargs):
        dsIn = xr.Dataset({
            'ref': xr.DataArray(obs, dims=('time',)),
            'hist': xr.DataArray(cm_hist, dims=('time',))
        })

        kind = '+' if self.kind == 'additive' else '*'
        quantiles = equally_spaced_nodes(self.nquantiles).astype(dsIn.ref.dtype)

        if self.detrending:
            dsTr = dqm_train.func(dsIn, dim='time', kind=kind, quantiles=quantiles, adapt_freq_thresh=None)            
            dsTr = dsTr.assign(sim=xr.DataArray(cm_future, dims=('time',))) 
            scen = dqm_adjust.func(dsTr, group=Grouper('time'), interp='linear', kind=kind, extrapolation='constant', detrend=1)
        else:
            dsTr = eqm_train.func(dsIn, dim='time', kind=kind, quantiles=quantiles, adapt_freq_thresh=None)            
            dsTr = dsTr.assign(sim=xr.DataArray(cm_future, dims=('time',))) 
            scen = qm_adjust.func(dsTr, group=Grouper('time'), interp='linear', kind=kind, extrapolation='constant')
        return scen.scen.values