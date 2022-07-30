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


Temperature = Variable(name="Temperature", reasonable_physical_range=[0, 400])
Precipitation = Variable(name="Precipitation", reasonable_physical_range=[0, np.inf])


str_to_variable_class = {
    "pr": Precipitation,
    "precip": Precipitation,
    "precipitation": Precipitation,
    "rainfall": Precipitation,
    "tas": Temperature,
    "temp": Temperature,
    "temperature": Temperature,
}


def map_standard_precipitation_method(
    precipitation_model_type: str = "censored",
    precipitation_amounts_distribution=scipy.stats.gamma,
    precipitation_censoring_value: float = 0.1,
    precipitation_hurdle_model_randomization: bool = True,
    precipitation_hurdle_model_kwds_for_distribution_fit={"floc": 0, "fscale": None},
):
    if precipitation_model_type == "censored":
        if precipitation_model_type == "censored" and precipitation_amounts_distribution != scipy.stats.gamma:
            raise ValueError("Only the gamma distribution is supported for a censored precipitation model")
        if precipitation_censoring_value < 0:
            raise ValueError("precipitation_censoring_value needs to be >= 0")
        method = utils.gen_PrecipitationGammaLeftCensoredModel(censoring_value=precipitation_censoring_value)
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
        raise ValueError("variable_str needs to be one of %s" % str_to_variable_class.keys())
    return str_to_variable_class.get(variable_str)
