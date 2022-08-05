# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Variable module - Standard variable-definitions"""

from typing import Union

import attrs
import numpy as np
import scipy.stats

import PACKAGE_NAME.utils as utils


@attrs.define(eq=False)
class Variable:
    name: str = attrs.field(default="unknown", validator=attrs.validators.instance_of(str))
    unit: str = attrs.field(default="unknown", validator=attrs.validators.instance_of(str))
    reasonable_physical_range: list = attrs.field(default=None)

    @reasonable_physical_range.validator
    def validate_reasonable_physical_range(self, attribute, value):
        if value is not None:
            if len(value) != 2:
                raise ValueError("reasonable_physical_range should have only a lower and upper physical range")
            if not all(isinstance(elem, (int, float)) for elem in value):
                raise ValueError("reasonable_physical_range needs to be a list of floats")
            if not value[0] < value[1]:
                raise ValueError("lower bounds needs to be smaller than upper bound in reasonable_physical_range")


hurs = Variable(name="Daily mean near-surface relative humidity", unit="%")
pr = Variable(name="Daily mean precipitation", unit="kg m-2 s-1", reasonable_physical_range=[0, np.inf])
# prsn = Variable(name = "Daily mean snowfall flux", unit="kg m-2 s-1")
prsnratio = Variable("Daily mean snowfall flux / Daily mean precipitation", unit="1")
psl = Variable(name="Daily mean sea-level pressure", unit="Pa")
rlds = Variable(name="Daily mean surface downwelling longwave radiation", unit="W m-2")
rsds = Variable(name="Daily mean surface downwelling shortwave radiation", unit="W m-2")
sfcwind = Variable(name="Daily mean near-surface wind speed", unit="m s-1")
tas = Variable(name="Daily mean near-surface air temperature", unit="K", reasonable_physical_range=[0, 400])
tasmin = Variable(name="Daily minimum near-surface air temperature", unit="K", reasonable_physical_range=[0, 400])
tasmax = Variable(name="Daily maximum near-surface air temperature", unit="K", reasonable_physical_range=[0, 400])
tasrange = Variable(
    name="Daily near-surface air temperature range (tasmax-tasmin)", unit="K", reasonable_physical_range=[0, 400]
)
tasskew = Variable(
    name="Daily near-surface air temperature skew (tas-tasmin)/tasrange", unit="1", reasonable_physical_range=[0, 400]
)


str_to_variable_class = {
    "hurs": hurs,
    "pr": pr,
    "prsnratio": prsnratio,
    "ps": psl,
    "psl": psl,
    "rlds": rlds,
    "sfcwind": sfcwind,
    "tas": tas,
    "t2m": tas,
    "tasmin": tasmin,
    "tasmax": tasmax,
    "tasrange": tasrange,
    "tasskew": tasskew,
}


def map_standard_precipitation_method(
    precipitation_model_type: str = "censored",
    precipitation_amounts_distribution=scipy.stats.gamma,
    precipitation_censoring_threshold: float = 0.1,
    precipitation_hurdle_model_randomization: bool = True,
    precipitation_hurdle_model_kwds_for_distribution_fit={"floc": 0, "fscale": None},
):
    if precipitation_model_type == "censored":
        if precipitation_model_type == "censored" and precipitation_amounts_distribution != scipy.stats.gamma:
            raise ValueError("Only the gamma distribution is supported for a censored precipitation model")
        if precipitation_censoring_threshold < 0:
            raise ValueError("precipitation_censoring_threshold needs to be >= 0")
        method = utils.gen_PrecipitationGammaLeftCensoredModel(censoring_threshold=precipitation_censoring_threshold)
    elif precipitation_model_type == "hurdle":
        method = utils.gen_PrecipitationHurdleModel(
            distribution=precipitation_amounts_distribution,
            fit_kwds=precipitation_hurdle_model_kwds_for_distribution_fit,
            cdf_randomization=precipitation_hurdle_model_randomization,
        )
    elif precipitation_model_type == "ignore_zeros":
        method = utils.gen_PrecipitationIgnoreZeroValuesModel(precipitation_amounts_distribution)
    else:
        raise ValueError(
            "precipitation_model_type has wrong value. Needs to be one of ['censored', 'hurdle', 'ignore_zeros']"
        )

    return method


def map_variable_str_to_variable_class(variable_str: str):
    variable_str = variable_str.lower()
    if variable_str not in str_to_variable_class.keys():
        raise ValueError(
            "%s not known as variable. Variable needs to be one of %s"
            % (variable_str, list(str_to_variable_class.keys()))
        )
    return str_to_variable_class.get(variable_str)
