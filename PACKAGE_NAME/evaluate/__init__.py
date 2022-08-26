# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# from dictionary import *

"""
:py:mod:`evaluate`-module: provides the necessary functionality to evaluate the bias correction of climate models.


**There are three components to the evaluation module:**

**Testing assumptions of different debiasers** 
    
Different debiasers rely on different assumptions - 
some are parametrics, others non-parametric, some bias correct each day of the year separately, 
others are applied to all days of the year in the same way. 

This components is meant to check some 
of these assumptions and help the user rule out the use of some debiasers that are not fit for
purpose in this specific application.

Different bias correction methods rely on different assumptions, as described above. A detailed overview of assumptions associated with a specific bias correction method can be found in the documentation of each debiaser. For the sake of demonstration, we investigate the goodness of fit in this notebook:

For all parametric methods, distributions are fitted to the data. Default distributions for each variable are specified in the individual debiasers. We assess the following two components:
- Is the fit of the default distribution 'good enough' or should a different distribution be used? (1)
- Is there any seasonality in the data that should be accounted for, for example by applying a 'running window mode' (meaning that the bias correction is fitted separately for different parts of the year, i.e. windows)? (2)







**Evaluating the bias corrected model on a validation period** 

In order to assess the performance of a bias correction method, the bias corrected model
data should be compared to observational / reanalysis data. This component provides
 insight into the correction of marginal biases, as well as temporal, spatial
and spatiotemporal metrics.
 







 
**Investigating whether the climate change trend is preserved** 

Bias correction methods can significantly
modify the trend projected in the climate model simulation (Switanek 2017). If the user does not consider
the simulated trend to be credible, then modifying it can be a good thing to do. However, any trend modification
should always be a concious and informed choice, and it the belief that a bias correction method will improve the
trend should be justified. Otherwise, the trend modification through the application of a bias correction method
should be considered an artifact. Therefore, this section assesses whether a certain method preserves the cliamte
model trend or not. Some methods implemented in this package are explicitly trend preserving, for more details see
the methodologies and descriptions of the individual debiasers.

"""