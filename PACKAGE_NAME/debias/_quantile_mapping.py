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

from ..utils import PrecipitationHurdleModelGamma, StatisticalModel
from ..variables import Variable, map_standard_precipitation_method, pr, tas
from ._debiaser import Debiaser

default_settings = {
    tas: {"distribution": scipy.stats.norm, "detrending": "additive"},
    pr: {"distribution": PrecipitationHurdleModelGamma, "detrending": "multiplicative"},
}


@attrs.define
class QuantileMapping(Debiaser):
    """
    Implements (detrended) Quantile Mapping following Cannon et al. 2015 and Maraun 2016.
    Let cm refer to climate model output, obs to observations and hist/future to whether the data was collected from the reference period or is part of future projections.
    Let :math: `F be a CDF. The future climate projections :math: `x_{\text{cm_fut}}` are then mapped using a QQ-mapping between :math: `F_{\text{cm_hist}}` and :math: `F_{\text{obs}}`, so:

    .. math:: x_{\text{cm_fut}} \\rightarrow F^{-1}_{\text{obs}}(F_{\text{cm_hist}}(x_{\text{cm_fut}}))

    If detrended quantile mapping is used then :math: `x_{\text{cm_fut}}` is first rescaled and then the mapped value is scaled back either additively or multiplicatively. That means for additive detrending:

    .. math:: x_{\text{cm_fut}} \\rightarrow F^{-1}_{\text{obs}}(F_{\text{cm_hist}}(x_{\text{cm_fut} + \mean x_{\text{cm_hist}} - \mean x_{\text{cm_fut}}})) + \mean x_{\text{cm_fut}} - \mean x_{\text{cm_hist}}

    and for multiplicative detrending.

    .. math:: x_{\text{cm_fut}} \\rightarrow F^{-1}_{\text{obs}}(F_{\text{cm_hist}}(x_{\text{cm_fut} \cdot \frac{\mean x_{\text{cm_hist}}}{\mean x_{\text{cm_fut}}}})) \cdot \frac{\mean x_{\text{cm_fut}}}{\mean x_{\text{cm_hist}}}

    Here :math: `\mean x_{\text{cm_fut}}` designs the mean of :math: `x_{\text{cm_fut}}` and similar for :math: `x_{\text{cm_hist}}`.
    Detrended Quantile Mapping accounts for changes in the projected values and is thus trend-preserving in the mean.


    Attributes
    ----------
    distribution: Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel]
        Distribution or statistical model used to compute the CDFs F.
        Usually a distribution in scipy.stats.rv_continuous, but can also be an empirical distribution as given by scipy.stats.rv_histogram or a more complex statistical model as wrapped by the StatisticalModel class (see utils).
    detrending: str
        One of ["additive", "multiplicative", "no_detrending"]. Default: "no_detrending". What kind of scaling is applied to the future climate model data before quantile mapping.
    variable: str
        Variable for which the debiasing is done. Default: "unknown".

    References:
    Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes? In Journal of Climate (Vol. 28, Issue 17, pp. 6938–6959). American Meteorological Society. https://doi.org/10.1175/jcli-d-14-00754.1
    Maraun, D. (2016). Bias Correcting Climate Change Simulations - a Critical Review. In Current Climate Change Reports (Vol. 2, Issue 4, pp. 211–220). Springer Science and Business Media LLC. https://doi.org/10.1007/s40641-016-0050-x     Vrac, M., Drobinski, P., Merlo, A., Herrmann, M., Lavaysse, C., Li, L., & Somot, S. (2012). Dynamical and statistical downscaling of the French Mediterranean climate: uncertainty assessment. In Natural Hazards and Earth System Sciences (Vol. 12, Issue 9, pp. 2769–2784). Copernicus GmbH. https://doi.org/10.5194/nhess-12-2769-2012
    """

    distribution: Union[
        scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel
    ] = attrs.field(
        validator=attrs.validators.instance_of(
            (scipy.stats.rv_continuous, scipy.stats.rv_discrete, scipy.stats.rv_histogram, StatisticalModel)
        )
    )
    detrending: str = attrs.field(
        default="no_detrending", validator=attrs.validators.in_(["additive", "multiplicative", "no_detrending"])
    )
    variable: str = attrs.field(default="unknown", eq=False)

    # ----- Constructors -----
    @classmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        return super().from_variable(cls, default_settings, variable, **kwargs)

    @classmethod
    def for_precipitation(
        cls,
        precipitation_model_type: str = "censored",
        precipitation_amounts_distribution=scipy.stats.gamma,
        precipitation_censoring_value: float = 0.1,
        precipitation_hurdle_model_randomization: bool = True,
        precipitation_hurdle_model_kwds_for_distribution_fit={"floc": 0, "fscale": None},
        **kwargs
    ):
        """
        Instanciates the class to a precipitation-debiaser. This allows granular setting of available precipitation models without needing to explicitly specify the precipitation censored model for example.

        Parameters
        ----------
        detrending: str
            One of ["additive", "multiplicative", "no_detrending"]. Type of delta-change used.
        precipitation_model_type: str
            One of ["censored", "hurdle", "ignore_zeros"]. Model type to be used. See utils.gen_PrecipitationGammaLeftCensoredModel, utils.gen_PrecipitationHurdleModel and utils.gen_PrecipitationIgnoreZeroValuesModel for more details.
        precipitation_amounts_distribution: scipy.stats.rv_continuous
            Distribution used for precipitation amounts. For the censored model only scipy.stats.gamma is possible.
        precipitation_censoring_value: float
            The censoring-value if a censored precipitation model is used.
        precipitation_hurdle_model_randomization: bool
            Whether when computing the cdf-values for a hurdle model randomization shall be used. See utils.gen_PrecipitationHurdleModel for more details
        precipitation_hurdle_model_kwds_for_distribution_fit: dict
            Dict of parameters used for the distribution fit inside a hurdle model. Standard: location of distribution is fixed at zero (floc = 0) to stabilise Gamma distribution fits in scipy.
        **kwargs:
            All other class attributes that shall be set and where the standard values shall be overwritten.

        """
        variable = pr

        method = map_standard_precipitation_method(
            precipitation_model_type,
            precipitation_amounts_distribution,
            precipitation_censoring_value,
            precipitation_hurdle_model_randomization,
            precipitation_hurdle_model_kwds_for_distribution_fit,
        )

        parameters = {
            **default_settings[variable],
            "distribution": method,
            "variable": variable.name,
        }

        return cls(**{**parameters, **kwargs})

    # ----- Helpers -----
    def _standard_qm(self, x, fit_cm_hist, fit_obs):
        return self.distribution.ppf(self.distribution.cdf(x, *fit_cm_hist), *fit_obs)

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
