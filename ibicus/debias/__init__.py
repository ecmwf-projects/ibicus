# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""
The :py:mod:`debias`-module provides the necessary functionality to bias correct climate models.

The foundation of the module is the :py:class:`Debiaser` class from which every debiaser inherits and which provides a unified interface to instantiate and apply debiasers.

The following bias correction methodologies are currently implemented in the package, each based on the respective publication cited.

.. autosummary::
    Debiaser
    LinearScaling
    DeltaChange
    QuantileMapping
    ScaledDistributionMapping
    CDFt
    ECDFM
    QuantileDeltaMapping
    ISIMIP

**Methodology**

For a brief introduction to bias correction, and some issues to pay attention to when applying a bias correction method, have a look at the `'What is bias adjustment?' <../getting_started/whatisdebiasing.html>`_ page.

The general idea behind bias correction is to calibrate an empirical transfer function between simulated and observed distributional parameters that bias adjust the climate model output.
This can be done in a number of different ways. This table in `'Overview' <../getting_started/overview.html#what-is-ibicus>`_ provides an overview of the different methodological choices made by the bias correction methods implemented in the package.
For a detailed description of their methodology, we refer you to the class descriptions and the cited publications.

**Usage**

.. testsetup::

        from ibicus.debias import ISIMIP, CDFt, QuantileMapping


Three types of data are required in order to conduct bias correction for a given climatic variable:

1. Observations / reanalysis data for given historical period: ``obs``.

2. Climate model simulation for same historical period as observations: ``cm_hist``.

3. Climate model simulation for the period that is to be bias corrected, often a future period: ``cm_future``.

Let's generate some pseudo climate data:

>>> import numpy as np
>>> np.random.seed(12345)
>>> obs, cm_hist, cm_future = np.random.normal(loc = 3, size = 40000).reshape((10000, 2, 2)), np.random.normal(loc = 5, size = 40000).reshape((10000, 2, 2)), np.random.normal(loc = 7, size = 40000).reshape((10000, 2, 2))

Every debiaser can be instatiated using :py:func:`from_variable` and a standard abbrevation for a meteorological variable following the CMIP-convention::

>>> debiaser = CDFt.from_variable("tas")

This instantiates a debiaser with default settings for ``"tas"`` (daily mean 2m air surface temperature (K)).

The following code the applies this debiaser, given the data ``obs``, ``cm_hist`` and ``cm_future``.

>>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)


**Variable support**

Variables currently supported across debiasers include::

["hurs", "pr", "prsnratio", "psl", "rlds", "rsds", "sfcWind", "tas", "tasmin", "tasmax", "tasrange", "tasskew"]

However, whereas some bias correction methods such as ISIMIP have explicitly been published and implemented for all these variables
(tasmin and tasmax are not explicitely debiased but can be calculated from tas, tasrange and taskew), other methods have been published only for
specific variables such as precipitation or temperature. Where possible, the authors have introduced informed choices for the default settings of other variables
as well.

The following table provides an overview of which debiasers currently have which default settings for which variables. Crosses in brackets signify so-called
'experimental default settings' that have been chosen by the creators of this package and may not have been evaluated by the peer reviewed literature. It is advised to evaluate those carefully.


+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| Variable   | :py:class:`LinearScaling` | :py:class:`DeltaChange` | :py:class:`QuantileMapping` | :py:class:`ScaledDistributionMapping` | :py:class:`CDFt`  | :py:class:`ECDFM` | :py:class:`QuantileDeltaMapping` | :py:class:`ISIMIP` |
+============+===========================+=========================+=============================+=======================================+===================+===================+==================================+====================+
| hurs       | .. centered:: (x)         |  .. centered:: (x)      |  .. centered:: (x)          |                                       | .. centered:: (x) | .. centered:: (x) | .. centered:: (x)                | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| pr         | .. centered:: x           |  .. centered:: x        |  .. centered:: x            |  .. centered:: x                      | .. centered:: x   | .. centered:: x   | .. centered:: x                  | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| prsnratio  |                           |                         |                             |                                       |                   |                   |                                  | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| psl        | .. centered:: (x)         |  .. centered:: (x)      |  .. centered:: (x)          |                                       | .. centered:: (x) | .. centered:: (x) | .. centered:: (x)                | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| rlds       | .. centered:: (x)         |  .. centered:: (x)      |  .. centered:: (x)          |                                       | .. centered:: (x) | .. centered:: (x) | .. centered:: (x)                | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| rsds       | .. centered:: (x)         |  .. centered:: (x)      |                             |                                       | .. centered:: (x) |                   |                                  | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| sfcWind    | .. centered:: (x)         |  .. centered:: (x)      |  .. centered:: (x)          |                                       | .. centered:: (x) | .. centered:: (x) | .. centered:: (x)                | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| tas        | .. centered:: x           |  .. centered:: x        |  .. centered:: x            |  .. centered:: x                      | .. centered:: x   | .. centered:: x   | .. centered:: x                  | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| tasmin     | .. centered:: x           |  .. centered:: x        |  .. centered:: (x)          |  .. centered:: (x)                    | .. centered:: x   | .. centered:: (x) | .. centered:: (x)                |                    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| tasmax     | .. centered:: x           |  .. centered:: x        |  .. centered:: (x)          |  .. centered:: (x)                    | .. centered:: x   | .. centered:: (x) | .. centered:: (x)                |                    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| tasrange   |                           |                         |                             |                                       | .. centered:: (x) |                   |                                  | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+
| tasskew    |                           |                         |                             |                                       | .. centered:: (x) |                   |                                  | .. centered:: x    |
+------------+---------------------------+-------------------------+-----------------------------+---------------------------------------+-------------------+-------------------+----------------------------------+--------------------+


.. note:: A warning message is shown for variable-debiaser combinations that are still experimental.


**Setting and Modifying parameters**

In addition to these default setting settings, the user can also modify the settings and parameters of each debiaser.
In particular for those default settings that are still experimental, it is highly recommended to try out some alternatives.
This is possible either by setting alternative settings in :py:func:`from_variable` or modifying the object attribute:

>>> debiaser = CDFt.from_variable("tas", delta_shift = "no_shift")

or:

>>> debiaser = CDFt.from_variable("tas")
>>> debiaser.delta_shift = "no_shift"

It is also possible to instantiate debiasers by directly setting the necessary parameters, bypassing :py:func:`from_variable`:

>>> debiaser1 = CDFt()
>>> from scipy.stats import norm
>>> debiaser2 = QuantileMapping(distribution = norm, detrending = "none")

Some debiasers additionally provide a :py:func:`for_precipitation` classmethod to help you initialise the debiaser for precipitation. Debiasing precipitation often requires setting additional arguments like a threshold under which precipitation is assumed zero and the :py:func:`for_precipitation` method helps with that:

>>> debiaser = QuantileMapping.for_precipitation(model_type = "hurdle")
>>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)

The documentation of the individual debiasers provides some information on this.

Some debiasers eg. ISIMIP require date information to be applied in a running window. Dates should then be passed as 1d numpy arrays as keyword arguments:

>>> debiaser = ISIMIP.from_variable("tas")
>>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future, time_obs = time_obs, time_cm_hist = time_cm_hist, time_cm_future = time_cm_future)

Whenever date information is needed this is indicated in the debiaser documentation.

When applying the debiaser we can parallelise the execution by setting ``parallel = True`` and setting the ``nr_processes`` (default: 4). It is also possible to activate/deactive the progressbar using the ``progressbar`` argument and to activate a failsafe mode ``failsafe = True``, continuing execution when encoutnering errors:

>>> debiased_cm_future1 = debiaser1.apply(obs, cm_hist, cm_future, parallel = True, failsafe = True)

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
