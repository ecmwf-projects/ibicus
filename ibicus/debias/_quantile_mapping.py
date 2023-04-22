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
import scipy
import scipy.stats

from ..utils import PrecipitationHurdleModelGamma, StatisticalModel, threshold_cdf_vals
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
from ._debiaser import Debiaser

# ----- Default settings for debiaser ----- #
default_settings = {
    tas: {"distribution": scipy.stats.norm, "detrending": "additive"},
    pr: {"distribution": PrecipitationHurdleModelGamma, "detrending": "multiplicative"},
}
experimental_default_settings = {
    hurs: {"distribution": scipy.stats.beta, "detrending": "multiplicative"},
    psl: {"distribution": scipy.stats.beta, "detrending": "additive"},
    rlds: {"distribution": scipy.stats.beta, "detrending": "additive"},
    sfcwind: {"distribution": scipy.stats.gamma, "detrending": "multiplicative"},
    tasmin: {"distribution": scipy.stats.beta, "detrending": "additive"},
    tasmax: {"distribution": scipy.stats.beta, "detrending": "additive"},
}

# ----- Debiaser ----- #


@attrs.define(slots=False)
class QuantileMapping(Debiaser):
    """
    |br| Implements (detrended) Quantile Mapping based on Cannon et al. 2015 and Maraun 2016.

    (Parametric) quantile mapping maps every quantile of the climate model distribution to the corresponding quantile in observations during the reference period. Optionally, additive or multiplicative detrending of the mean can be applied to make the method trend preserving in the mean. Most methods build on quantile mapping.


    Let cm refer to climate model output, obs to observations and hist/future to whether the data was collected from the reference period or is part of future projections.
    Let :math:`F` be a CDF. The future climate projections :math:`x_{\\text{cm_fut}}` are then mapped using a QQ-mapping between :math:`F_{\\text{cm_hist}}` and :math:`F_{\\text{obs}}`, so:

    .. math:: x_{\\text{cm_fut}} \\rightarrow F^{-1}_{\\text{obs}}(F_{\\text{cm_hist}}(x_{\\text{cm_fut}}))

    If detrended quantile mapping is used then :math:`x_{\\text{cm_fut}}` is first rescaled and then the mapped value is scaled back either additively or multiplicatively. That means for additive detrending:

    .. math:: x_{\\text{cm_fut}} \\rightarrow F^{-1}_{\\text{obs}}(F_{\\text{cm_hist}}(x_{\\text{cm_fut}} + \\bar x_{\\text{cm_hist}} - \\bar x_{\\text{cm_fut}})) + \\bar x_{\\text{cm_fut}} - \\bar x_{\\text{cm_hist}}

    and for multiplicative detrending.

    .. math:: x_{\\text{cm_fut}} \\rightarrow F^{-1}_{\\text{obs}}\\left(F_{\\text{cm_hist}}\\left(x_{\\text{cm_fut}} \\cdot \\frac{\\bar x_{\\text{cm_hist}}}{\\bar x_{\\text{cm_fut}}}\\right)\\right) \\cdot \\frac{\\bar x_{\\text{cm_fut}}}{\\bar x_{\\text{cm_hist}}}

    Here :math:`\\bar x_{\\text{cm_fut}}` designs the mean of :math:`x_{\\text{cm_fut}}` and similar for :math:`x_{\\text{cm_hist}}`.
    Detrended Quantile Mapping accounts for changes in the projected values and is thus trend-preserving in the mean.

    For precipitation a distribution or model is needed that accounts for the mixed zero and positive value character. Default is a precipitation hurdle model (see :py:class:`ibicus.utils.gen_PrecipitationHurdleModel`). However, other models are also possible, :py:func:`for_precipitation` helps with the initialisation of different precipitation methods.

    **References**:

    - Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes? In Journal of Climate (Vol. 28, Issue 17, pp. 6938–6959). American Meteorological Society. https://doi.org/10.1175/jcli-d-14-00754.1
    - Maraun, D. (2016). Bias Correcting Climate Change Simulations - a Critical Review. In Current Climate Change Reports (Vol. 2, Issue 4, pp. 211–220). Springer Science and Business Media LLC. https://doi.org/10.1007/s40641-016-0050-x

    |br|
    **Usage information:**

    - Default settings exist for: ``["hurs", "pr", "psl", "rlds", "sfcWind", "tas", "tasmin", "tasmax"]``.

    - :py:func:`apply` requires: no additional arguments except ``obs``, ``cm_hist``, ``cm_future``.

    - Next to :py:func:`from_variable` a :py:func:`for_precipitation`-method exists to help you initialise the debiaser for :py:data:`pr`.

    - The debiaser works with data in any time specification (daily, monthly, etc.), although some of the default distributions have the best fit to daily data.

    |br|
    **Examples:**

    Initialising using :py:class:`from_variable`:

    >>> debiaser = QuantileMapping.from_variable("tas")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    Initialising using :py:class:`for_precipitation`:

    >>> debiaser = QuantileMapping.for_precipitation(model_type = "hurdle")
    >>> debiaser.apply(obs, cm_hist, cm_future)

    |br|

    Attributes
    ----------
    distribution : Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel]
        Distribution or statistical model used to compute the CDFs F.
        Usually a distribution in :py:class:`scipy.stats.rv_continuous`, but can also be an empirical distribution as given by :py:class:`scipy.stats.rv_histogram` or a more complex statistical model as wrapped by the :py:class:`ibicus.utils.StatisticalModel` class.
    detrending : str
        One of ``["additive", "multiplicative", "no_detrending"]``. What kind of scaling is applied to the future climate model data before quantile mapping. Default: ``"no_detrending"``.
    cdf_threshold : float
        Threshold to round CDF-values away from zero and one. Default: ``1e-10``.
    variable : str
        Variable for which the debiasing is done. Default: ``"unknown"``.
    """

    distribution: Union[
        scipy.stats.rv_continuous,
        scipy.stats.rv_discrete,
        scipy.stats.rv_histogram,
        StatisticalModel,
    ] = attrs.field(
        validator=attrs.validators.instance_of(
            (
                scipy.stats.rv_continuous,
                scipy.stats.rv_discrete,
                scipy.stats.rv_histogram,
                StatisticalModel,
            )
        )
    )
    detrending: str = attrs.field(
        default="no_detrending",
        validator=attrs.validators.in_(["additive", "multiplicative", "no_detrending"]),
    )
    cdf_threshold: float = attrs.field(
        default=1e-10, validator=attrs.validators.instance_of(float)
    )

    # ----- Constructors -----
    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super()._from_variable(
            cls, variable, default_settings, experimental_default_settings, **kwargs
        )

    @classmethod
    def for_precipitation(
        cls,
        model_type: str = "hurdle",
        amounts_distribution=scipy.stats.gamma,
        censoring_threshold: float = 0.1 / 86400,
        hurdle_model_randomization: bool = True,
        hurdle_model_kwds_for_distribution_fit={"floc": 0, "fscale": None},
        **kwargs
    ):
        """
        Instanciates the class to a precipitation-debiaser. This allows granular setting of available precipitation models without needing to explicitly specify the precipitation censored model for example.

        Parameters
        ----------
        model_type : str
            One of ``["censored", "hurdle", "ignore_zeros"]``. Model type to be used. See :py:class:`ibicus.utils.gen_PrecipitationGammaLeftCensoredModel`, :py:class:`ibicus.utils.gen_PrecipitationHurdleModel` and :py:class:`ibicus.utils.gen_PrecipitationIgnoreZeroValuesModel` for more details.
        amounts_distribution : scipy.stats.rv_continuous
            Distribution used for precipitation amounts. For the censored model only ``scipy.stats.gamma`` is possible.
        censoring_threshold : float
            The censoring-value if a censored precipitation model is used.
        hurdle_model_randomization : bool
            Whether when computing the cdf-values for a hurdle model randomization shall be used. See :py:class:`ibicus.utils.gen_PrecipitationHurdleModel` for more details.
        hurdle_model_kwds_for_distribution_fit : dict
            Dict of parameters used for the distribution fit inside a hurdle model. Default: location of distribution is fixed at zero (``floc = 0``) to stabilise Gamma distribution fits in scipy.
        **kwargs:
            All other class attributes that shall be set and where the standard values shall be overwritten.

        """
        variable = pr

        method = map_standard_precipitation_method(
            model_type,
            amounts_distribution,
            censoring_threshold,
            hurdle_model_randomization,
            hurdle_model_kwds_for_distribution_fit,
        )

        parameters = {
            **default_settings[variable],
            "distribution": method,
            "variable": variable.name,
        }

        return cls(**{**parameters, **kwargs})

    # ----- Helpers -----
    def _standard_qm(self, x, fit_cm_hist, fit_obs):
        return self.distribution.ppf(
            threshold_cdf_vals(
                self.distribution.cdf(x, *fit_cm_hist), self.cdf_threshold
            ),
            *fit_obs
        )

    # ----- Apply location function -----
    def apply_location(self, obs, cm_hist, cm_future):
        fit_obs = self.distribution.fit(obs)
        fit_cm_hist = self.distribution.fit(cm_hist)

        if self.detrending == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return self._standard_qm(cm_future - delta, fit_cm_hist, fit_obs) + delta
        elif self.detrending == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return self._standard_qm(cm_future / delta, fit_cm_hist, fit_obs) * delta
        elif self.detrending == "no_detrending":
            return self._standard_qm(cm_future, fit_cm_hist, fit_obs)
        else:
            raise ValueError(
                "self.detrending has wrong value. Needs to be one of ['additive', 'multiplicative', 'no_detrending']"
            )
