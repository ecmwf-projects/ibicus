# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# from dictionary import *

"""
The :py:mod:`evaluate`-module: provides a set of functionalities to assess the performance of your bias correction method.

Bias correction is prone to mis-use and requires careful evaluation, as demonstrated and argued in Maraun et al. 2017. 
In particular, the bias correction methods implemented in this package operate on a marginal level, that is they correct 
distribution of individual variables at individual locations. There is therefore only a subset of climate model biases 
that these debiasers will be able to correct. Biases in the temporal or spatial structure of climate models, or the 
feedbacks to large-scale weather patterns might not be well corrected.

The :py:mod:`evaluate`-module: attempts to provide the user with the functionality to make an informed decision whether a chosen bias correction
method is fit for purpose - whether it corrects marginal, as well as spatial and temporal statistical properties properties in the desired manner,
as well as how it modifies the multivariate structure, if and how it modifies the climate change trend, and how it changes the bias in selected
climate impact metrics.


**There are three components to the evaluation module:**

**1. Testing assumptions of different debiasers** 
    
Different debiasers rely on different assumptions - 
some are parametrics, others non-parametric, some bias correct each day or month of the year separately, 
others are applied to all days of the year in the same way. 

This components is meant to check some of these assumptions and for example help the user choose an appropriate function
to fit the data to, an appropriate application window (entire year vs each days or month individually) and rule out the use 
of some debiasers that are not fit for purpose in a specific application.

The current version of this component can analyse the following two questions?
- Is the fit of the default distribution 'good enough' or should a different distribution be used?
- Is there any seasonality in the data that should be accounted for, for example by applying a 'running window mode' (meaning that the bias correction is fitted separately for different parts of the year, i.e. windows)?

The following functions are currently available:

.. autosummary::
    assumptions.calculate_aic
    assumptions.plot_aic
    assumptions.plot_fit_worst_aic
    assumptions.plot_quantile_residuals



**2. Evaluating the bias corrected model on a validation period** 

In order to assess the performance of a bias correction method, the bias corrected model
data has to be compared to observational / reanalysis data. The historical period for which observations exist is therefore split
into to dataset in pre-processing - a reference period, and a validation period. 

There are two types of analysis that the evaluation module enables you to conduct: 

1. Statistical properties: this includes the marginal bias of descriptive statistics such as the mean, or 5th and 95th percentile, as well as the difference in spatial and multivariate correlation structure. 
2. Threshold metrics: A threshold metric is an instance of the class :py:mod:`ThresholdMetric`-class and is needs to be one of
   four types: exceedance of the specified threshold value ('higher'), underceedance of the threshold value ('lower'), falling within two specified bounds ('between')
   or falling outside two specified bounds ('outside'). With the functionalities provided as part of :py:mod:`ThresholdMetric`-class, the marginal exceedance probability as well
   as the temporal spell length, the spatial extent and the spatiotemporal cluster size can be analysed. Some threshold metrics are pre-specified, and the user can add further
   metrics in the following way:
       
>>> frost_days = ThresholdMetric(name="Frost days (tasmin<0°C)", variable="tasmin", threshold_value=273.13, threshold_type="lower")

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
    metrics.ThresholdMetric.calculate_instances_of_threshold_exceedance
    metrics.ThresholdMetric.filter_threshold_exceedances
    metrics.ThresholdMetric.calculate_exceedance_probability
    metrics.ThresholdMetric.calculate_number_annual_days_beyond_threshold
    metrics.ThresholdMetric.calculate_spell_length
    metrics.ThresholdMetric.calculate_spatial_extent
    metrics.ThresholdMetric.calculate_spatiotemporal_clusters
    metrics.ThresholdMetric.violinplots_clusters
    
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
    marginal.plot_histogram

The following functions are available to analyse the bias in spatial correlation structure:

.. autosummary::
    correlation.rmse_spatial_correlation_distribution
    correlation.rmse_spatial_correlation_boxplot

To analyse the multivariate correlation structure, as well as joint threshold exceedances:
    
.. autosummary::
    multivariate.calculate_conditional_joint_threshold_exceedance
    multivariate.plot_conditional_joint_threshold_exceedance
    multivariate.calculate_and_spatialplot_multivariate_correlation
    multivariate.plot_correlation_single_location
    multivariate.plot_bootstrap_correlation_replicates
    
 
**3. Investigating whether the climate change trend is preserved** 

Bias correction methods can significantly
modify the trend projected in the climate model simulation (Switanek 2017). If the user does not consider
the simulated trend to be credible, then modifying it can be a good thing to do. However, any trend modification
should always be a concious and informed choice, and it the belief that a bias correction method will improve the
trend should be justified. Otherwise, the trend modification through the application of a bias correction method
should be considered an artifact.

This component helps the user assess whether a certain method preserves the cliamte
model trend or not. Some methods implemented in this package are explicitly trend preserving, for more details see
the methodologies and descriptions of the individual debiasers.

.. autosummary::
    trend.calculate_future_trend_bias
    trend.plot_future_trend_bias_boxplot
    trend.plot_future_trend_bias_spatial








"""