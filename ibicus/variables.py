# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
:py:mod:`Variable` module - Standard definitions of climatic variables.

The variables below represent all variables currently recognized by the package and mapped onto default arguments by some of the debiasers using the :py:func:`from_variable` classmethod of a debiaser. However by setting class parameters oneself it is possible to use debiasers for other variables than the one below.

.. autosummary::
    hurs
    pr
    prsnratio
    psl
    rlds
    rsds
    sfcwind
    tas
    tasmin
    tasmax
    tasrange
    tasskew
"""

import attrs
import scipy.stats

import ibicus.utils as utils


@attrs.define(eq=False)
class Variable:
    """
    Provides an abstract interface for climatic variables

    It stores some essential attributes of the variable and is mainly used for internal purposes, however defining new ones is also possible.

    Examples
    --------
    >>> hurs = Variable(name="Daily mean near-surface relative humidity", unit="%").

    Attributes
    ----------
    name : str
        Name of climatic variable.
    unit : str
        Unit of climatic variable.
    reasonable_physical_range : list
        List of upper and lower bound of expectable reasonable physical range of climatic variable.
    """

    name: str = attrs.field(
        default="unknown", validator=attrs.validators.instance_of(str)
    )
    unit: str = attrs.field(
        default="unknown", validator=attrs.validators.instance_of(str)
    )
    reasonable_physical_range: list = attrs.field(default=None)

    @reasonable_physical_range.validator
    def _validate_reasonable_physical_range(self, attribute, value):
        if value is not None:
            if len(value) != 2:
                raise ValueError(
                    "reasonable_physical_range should have only a lower and upper physical range"
                )
            if not all(isinstance(elem, (int, float)) for elem in value):
                raise ValueError(
                    "reasonable_physical_range needs to be a list of floats"
                )
            if not value[0] < value[1]:
                raise ValueError(
                    "lower bounds needs to be smaller than upper bound in reasonable_physical_range"
                )


hurs = Variable(
    name="Daily mean near-surface relative humidity",
    unit="%",
    reasonable_physical_range=[1e-5, 150],
)
"""
Daily mean near-surface relative humidity, unit: %
"""

pr = Variable(
    name="Daily mean precipitation flux",
    unit="kg m-2 s-1",
    reasonable_physical_range=[0, 0.01],
)
"""
Daily mean precipitation flux, unit: kg m-2 s-1
"""

prsn = Variable(
    name="Daily mean snowfall flux",
    unit="kg m-2 s-1",
    reasonable_physical_range=[0, 0.006],
)
"""
Daily mean snowfall flux, unit: kg m-2 s-1
"""

prsnratio = Variable(
    "Daily mean snowfall flux / Daily mean precipitation flux", unit="1"
)
"""
Daily mean snowfall flux / Daily mean precipitation flux, unit: 1
"""

psl = Variable(
    name="Daily mean surface air pressure",
    unit="Pa",
    reasonable_physical_range=[0, 1000000],
)
"""
Daily mean surface air pressure, unit: Pa
"""

rlds = Variable(
    name="Daily mean surface downwelling longwave radiation",
    unit="W m-2",
    reasonable_physical_range=[0, 1000],
)
"""
Daily mean surface downwelling longwave radiation, unit: W m-2
"""

rsds = Variable(
    name="Daily mean surface downwelling shortwave radiation",
    unit="W m-2",
    reasonable_physical_range=[0, 1000],
)
"""
Daily mean surface downwelling shortwave radiation, unit: W m-2
"""

sfcwind = Variable(
    name="Daily mean near-surface wind speed",
    unit="m s-1",
    reasonable_physical_range=[1e-5, 500],
)
"""
Daily mean near-surface wind speed, unit: m s-1
"""

tas = Variable(
    name="Daily mean near-surface air temperature",
    unit="K",
    reasonable_physical_range=[100, 400],
)
"""
Daily mean near-surface air temperature, unit: K
"""

tasmin = Variable(
    name="Daily minimum near-surface air temperature",
    unit="K",
    reasonable_physical_range=[100, 400],
)
"""
Daily minimum near-surface air temperature, unit: K
"""

tasmax = Variable(
    name="Daily maximum near-surface air temperature",
    unit="K",
    reasonable_physical_range=[100, 400],
)
"""
Daily maximum near-surface air temperature, unit: K
"""

tasrange = Variable(
    name="Daily near-surface air temperature range (tasmax-tasmin)",
    unit="K",
    reasonable_physical_range=[1e-5, 100],
)
"""
Daily near-surface air temperature range (tasmax-tasmin), unit: K
"""

tasskew = Variable(
    name="Daily near-surface air temperature skew (tas-tasmin)/tasrange",
    unit="1",
    reasonable_physical_range=[0, 1],
)
"""
Daily near-surface air temperature skew (tas-tasmin)/tasrange, unit: 1
"""


str_to_variable_class = {
    "hurs": hurs,
    "pr": pr,
    "prsn": prsn,
    "prsnratio": prsnratio,
    "ps": psl,
    "psl": psl,
    "rlds": rlds,
    "rsds": rsds,
    "sfcwind": sfcwind,
    "tas": tas,
    "tasmin": tasmin,
    "tasmax": tasmax,
    "tasrange": tasrange,
    "tasskew": tasskew,
}


def map_standard_precipitation_method(
    model_type: str = "censored",
    amounts_distribution=scipy.stats.gamma,
    censoring_threshold: float = 0.1,
    hurdle_model_randomization: bool = True,
    hurdle_model_kwds_for_distribution_fit={"floc": 0, "fscale": None},
):
    if model_type == "censored":
        if model_type == "censored" and amounts_distribution != scipy.stats.gamma:
            raise ValueError(
                "Only the gamma distribution is supported for a censored precipitation model"
            )
        if censoring_threshold < 0:
            raise ValueError("censoring_threshold needs to be >= 0")
        method = utils.gen_PrecipitationGammaLeftCensoredModel(
            censoring_threshold=censoring_threshold
        )
    elif model_type == "hurdle":
        method = utils.gen_PrecipitationHurdleModel(
            distribution=amounts_distribution,
            fit_kwds=hurdle_model_kwds_for_distribution_fit,
            cdf_randomization=hurdle_model_randomization,
        )
    elif model_type == "ignore_zeros":
        method = utils.gen_PrecipitationIgnoreZeroValuesModel(amounts_distribution)
    else:
        raise ValueError(
            "model_type has wrong value. Needs to be one of ['censored', 'hurdle', 'ignore_zeros']"
        )

    return method


def map_variable_str_to_variable_class(variable_str: str):
    variable_str = variable_str.lower()
    if variable_str not in str_to_variable_class.keys():
        raise ValueError(
            "%s is not known as variable. Variable needs to be one of %s"
            % (variable_str, list(str_to_variable_class.keys()))
        )
    return str_to_variable_class.get(variable_str)
