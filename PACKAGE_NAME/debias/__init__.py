# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""
:py:mod:`debias`-module: provides functionality to debias climate models

Basis is the :py:class:`Debiaser` class from which every debiaser inherits and which provides a unified interface to instantiate and apply debiasers.

Debiasers currently implemented: 

.. autosummary::
    Debiaser
    LinearScaling
    QuantileMapping
    ScaledDistributionMapping
    CDFt
    ECDFM
    QuantileDeltaMapping
    ISIMIP

**Usage**

.. testsetup::

        from PACKAGE_NAME.debias import ISIMIP, CDFt, QuantileMapping


For applying debiasing in general three arrays of data are required:

1. observations of the climatic variable: ``obs``.

2. values of the climate model for the climatic variable during the observational/reference or historical period -- to calculate the bias: ``cm_hist``.

3. values of the climate model for the climatic variable during the application or future period -- the values to debias: ``cm_future``. 

Let's generate some pseudo climate data:

>>> import numpy as np
>>> np.random.seed(12345)
>>> obs, cm_hist, cm_future = np.random.normal(loc = 3, size = 40000).reshape((10000, 2, 2)), np.random.normal(loc = 5, size = 40000).reshape((10000, 2, 2)), np.random.normal(loc = 7, size = 40000).reshape((10000, 2, 2))

Every debiaser can be instatiated using :py:func:`from_variable` and a standard abbrevation for a meteorological variable following the cs-convention (TODO: check)::

>>> debiaser = CDFt.from_variable("tas")

This instantiates a debiaser with default settings for `"tas"`. Variables currently supported across debiasers are:: 

["hurs", "pr", "prsnratio", "ps", "psl", "rlds", "rsds", "sfcWind", "tas", "tasmin", "tasmax", "tasrange", "tasskew"]

Applying the initiliased debiaser is as easy as follows. Given some climate model data (cm_hist`, `cm_fut` and observations `obs` it is just:

>>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)

**Modifying parameters**

Each debiaser has settings and parameters to modify its behavior. For each variable supported by a debiaser a set of default settings exists, however it highly encouraged to modify those. This is possible either by setting them in :py:func:`from_variable` or modifying the object attribute:

>>> debiaser = CDFt.from_variable("tas", delta_shift = "no_shift")

or:

>>> debiaser = CDFt.from_variable("tas")
>>> debiaser.delta_shift = "no_shift"

It is also possible to instantiate debiasers directly by setting the necessary parameters. This is useful to apply them to variables without current default settings:

>>> debiaser1 = CDFt()
>>> from scipy.stats import norm
>>> debiaser2 = QuantileMapping(distribution = norm, detrending = "none")

We can also apply those debiasers and compare the results:

>>> debiased_cm_future1 = debiaser1.apply(obs, cm_hist, cm_future)
>>> debiased_cm_future2 = debiaser2.apply(obs, cm_hist, cm_future)
>>> print(debiased_cm1.mean())
>>> print(debiased_cm2.mean())

.. note:: Some default settings for some variable-debiaser combinations might be experimental. A warning message is shown.

When applying we can control the verbosity with: 

>>> debiased_cm_future1 = debiaser1.apply(obs, cm_hist, cm_future, verbosity = "ERROR")

TODO: table debiasers x variables. Also for each debiaser table of variables supported. Possibly communicate experimental default settings here


"""

from ._cdft import CDFt
from ._debiaser import *
from ._delta_change import DeltaChange
from ._ecdfm import ECDFM
from ._isimip import ISIMIP
from ._linear_scaling import LinearScaling
from ._quantile_delta_mapping import QuantileDeltaMapping
from ._quantile_mapping import QuantileMapping
from ._scaled_distribution_mapping import ScaledDistributionMapping

if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
