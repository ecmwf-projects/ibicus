# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# from dictionary import *

"""
The :py:mod:`evaluate`-module: provides a set of functionalities to assess the performance of your bias adjustment method.

Bias adjustment is prone to mis-use and requires careful evaluation, as demonstrated and argued in Maraun et al. 2017.
In particular, the bias adjustment methods implemented in this package operate on a marginal level which means that they correct
distribution of individual variables at individual locations. There is therefore only a subset of climate model biases
that these debiasers will be able to correct. Biases in the temporal or spatial structure of climate models, or the
feedbacks to large-scale weather patterns might not be well corrected.

The :py:mod:`evaluate`-module: attempts to provide the user with the functionality to make an informed decision whether a chosen bias adjustment
method is fit for purpose - whether it corrects marginal, as well as spatial and temporal statistical properties properties in the desired manner,
as well as how it modifies the multivariate structure, if and how it modifies the climate change trend, and how it changes the bias in selected
climate impact metrics.


**There are three components to the evaluation module:**

**1. Evaluating the bias adjusted model on a validation period**

In order to assess the performance of a bias adjustment method, the bias adjusted model
data is compared to observational / reanalysis data. The historical period for which observations exist is therefore split
into to dataset in pre-processing - a reference period, and a validation period.

Both statistical properties such as quantiles or the mean of the bias adjusted variables, as well as tailored threshold metrics of particular relevance to the use-case
can be investigated. A threshold metric is an instance of the class :py:mod:`ThresholdMetric`-class. A number of threshold metrics such as dry days
are pre-defined in the package. The user can modify existing metrics or create new metrics from scratch. Threshold metrics are defined by a variable
they refer to, an absolute threshold value that can also be defined for by location or time period (such as day of year, or season), a name, and whether
the threshold sets a lower, higher, outer or inner bound to the variables of interest. An example of a threshold metric is:

>>> frost_days = ThresholdMetric(name="Frost days (tasmin<0°C)", variable="tasmin", threshold_value=273.13, threshold_type="lower")

The bias before and after bias adjustment in both statistical properties as well as threshold metrics can be evaluated marginally (i.e. location-wise).
Furthermore, the temporal spell length, the spatial extent and the spatiotemporal cluster size of threshold metrics such as hot days can be analysed
and plotted. Spatial and multivariate statistical properties can also be evaluated in an experimental setting.
The following table provides an overview of the different components that can be analysed in each of these two categories:


+----------------+------------------------+-----------------------+
|                | Statistical properties | Threshold metrics     |
+================+========================+=======================+
| Marginal       | x                      |  x                    |
+----------------+------------------------+-----------------------+
| Temporal       |                        |  x (spell length)     |
+----------------+------------------------+-----------------------+
| Spatial        | x (RMSE)               | x (spatial extent)    |
+----------------+------------------------+-----------------------+
| Spatiotemporal |                        |  x (cluster size)     |
+----------------+------------------------+-----------------------+
| Multivariate   | x (correlation)        |  x (joint exceedance) |
+----------------+------------------------+-----------------------+



Within the metrics class, the following functions are available:

.. autosummary::
    metrics.ThresholdMetric.from_quantile
    metrics.ThresholdMetric.calculate_instances_of_threshold_exceedance
    metrics.ThresholdMetric.filter_threshold_exceedances
    metrics.ThresholdMetric.calculate_exceedance_probability
    metrics.ThresholdMetric.calculate_number_annual_days_beyond_threshold
    metrics.ThresholdMetric.calculate_spell_length
    metrics.ThresholdMetric.calculate_spatial_extent
    metrics.ThresholdMetric.calculate_spatiotemporal_clusters

:py:mod:`AccumulativeThresholdMetric`-class is a child class of :py:mod:`ThresholdMetric`-class that adds additional functionalities for variables and metrics where
the total accumulative amount over a given threshold is of interest - this is the case for precipitation, but not for temperature for example. The following functions are added:

.. autosummary::
    metrics.AccumulativeThresholdMetric.calculate_percent_of_total_amount_beyond_threshold
    metrics.AccumulativeThresholdMetric.calculate_annual_value_beyond_threshold
    metrics.AccumulativeThresholdMetric.calculate_intensity_index

For the evaluation of marginal properties, the following functions are currently available:

.. autosummary::
    marginal.calculate_marginal_bias
    marginal.plot_marginal_bias
    marginal.plot_bias_spatial
    marginal.calculate_bias_days_metrics
    marginal.plot_spatiotemporal
    marginal.plot_histogram

The following functions are available to analyse the bias in spatial correlation structure:

.. autosummary::
    correlation.rmse_spatial_correlation_distribution
    correlation.rmse_spatial_correlation_boxplot

To analyse the multivariate correlation structure, as well as joint threshold exceedances:

.. autosummary::
    multivariate.calculate_conditional_joint_threshold_exceedance
    multivariate.plot_conditional_joint_threshold_exceedance
    multivariate.plot_conditional_probability_spatial
    multivariate.calculate_and_spatialplot_multivariate_correlation
    multivariate.plot_correlation_single_location
    multivariate.plot_bootstrap_correlation_replicates


**2. Investigating whether the climate change trend is preserved**

Bias adjustment methods can significantly
modify the trend projected in the climate model simulation (Switanek 2017). If the user does not consider
the simulated trend to be credible, then modifying it can be a good thing to do. However, any trend modification
should always be a concious and informed choice, and it the belief that a bias adjustment method will improve the
trend should be justified. Otherwise, the trend modification through the application of a bias adjustment method
should be considered an artifact.

This component helps the user assess whether a certain method preserves the cliamte
model trend or not. Some methods implemented in this package are explicitly trend preserving, for more details see
the methodologies and descriptions of the individual debiasers.

.. autosummary::
    trend.calculate_future_trend_bias
    trend.plot_future_trend_bias_boxplot
    trend.plot_future_trend_bias_spatial

**3. Testing assumptions of different debiasers**

Different debiasers rely on different assumptions -
some are parametrics, others non-parametric, some bias correct each day or month of the year separately,
others are applied to all days of the year in the same way.

This components enables the user to check some of these assumptions and for example help the user choose an appropriate function
to fit the data to, an appropriate application window (entire year vs each days or month individually) and rule out the use
of some debiasers that are not fit for purpose in a specific application.

The current version of this component can analyse the following two questions?
- Is the fit of the default distribution 'good enough' or should a different distribution be used?
- Is there any seasonality in the data that should be accounted for, for example by applying a 'running window mode' (meaning that the bias adjustment is fitted separately for different parts of the year, i.e. windows)?

The following functions are currently available:

.. autosummary::
    assumptions.calculate_aic
    assumptions.plot_aic
    assumptions.plot_fit_worst_aic
    assumptions.plot_quantile_residuals








"""
