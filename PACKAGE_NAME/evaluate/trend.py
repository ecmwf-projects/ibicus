# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn

from PACKAGE_NAME.variables import *




def _calculate_mean_trend_bias(variable: str, trend_type: str, raw_validate: np.ndarray, raw_future: np.ndarray, bc_validate: np.ndarray, bc_future: np.ndarray) ->  np.ndarray:

    if trend_type == "additive":

        bc_trend = np.mean(bc_future, axis=0) - np.mean(bc_validate, axis=0)
        raw_trend = np.mean(raw_future, axis=0) - np.mean(raw_validate, axis=0)
        bias = 100 * (bc_trend - raw_trend) / raw_trend

    elif trend_type =="multiplicative":
        
        bc_trend = np.mean(bc_future, axis=0) / np.mean(bc_validate, axis=0)
        raw_trend = np.mean(raw_future, axis=0) / np.mean(raw_validate, axis=0)
        bias = 100 * (bc_trend - raw_trend) / raw_trend
        
    else:

        raise ValueError("trend type currently not supported")

    return bias


def _calculate_quantile_trend_bias(variable: str, trend_type: str, quantile: float, raw_validate: np.ndarray, raw_future: np.ndarray, bc_validate: np.ndarray, bc_future: np.ndarray) ->  np.ndarray:

    if trend_type == "additive":

        bc_trend = np.quantile(bc_future, quantile, axis=0) - np.quantile(bc_validate, quantile, axis=0)
        raw_trend = np.quantile(raw_future, quantile, axis=0) - np.quantile(raw_validate, quantile, axis=0)
        bias = 100 * (bc_trend - raw_trend) / raw_trend
        
    elif trend_type =="multiplicative":
        
        if np.quantile(bc_validate, quantile, axis=0)==0 or np.quantile(raw_validate, quantile, axis=0):
            
            raise Warning("Selected quantile is zero, cannot analyse multiplicative trend. Output will be automatically set to zero.")
            
        else:
            
            bc_trend = np.quantile(bc_future, quantile, axis=0) / np.quantile(bc_validate, quantile, axis=0)
            raw_trend = np.quantile(raw_future, quantile, axis=0) / np.quantile(raw_validate, quantile, axis=0)
            bias = 100 * (bc_trend - raw_trend) / raw_trend

    else:

        raise ValueError("trend type currently not supported")

    return bias


def _calculate_metrics_trend_bias(variable: str, metric, raw_validate: np.ndarray, raw_future: np.ndarray, bc_validate: np.ndarray, bc_future: np.ndarray) -> np.ndarray:

    trend_raw = metric.calculate_exceedance_probability(raw_future) - metric.calculate_exceedance_probability(raw_validate)
    
    trend_bc = metric.calculate_exceedance_probability(bc_future) - metric.calculate_exceedance_probability(bc_validate)

    trend_bias = 100 * (trend_bc - trend_raw) / trend_raw

    return trend_bias



def calculate_future_trend_bias(variable: str, metric_collection: np.ndarray, raw_validate: np.ndarray, raw_future: np.ndarray, **debiased_cms) -> np.ndarray:
    
    """
    For each location, calculates the bias in the trend of the bias corrected model compared to the raw climate model for the following metrics: mean, 5% and 95% quantile (default) 
    as well as metrics passed as arguments to function. Trend can be specified as either additive or multiplicative. Function returns numpy array with three columns: 
    [Bias correction method, Metric, Bias value at certain location]

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    metrics: np.ndarray
        1d numpy array of strings containing the keys of the metrics to be analysed. Example: metrics = ['dry', 'wet']
    raw_validate: np.ndarray
        Raw climate data set in validation period
    raw_future: np.ndarray
        Raw climate data set in future period
    debiased_cms: np.ndarray
        Keyword arguments given in format debiaser_name = [debiased_dataset_validation_period, debiased_dataset_future_period]
        Example: QM = [tas_val_debiased_QM, tas_future_debiased_QM]

    """

    trend_bias_data = np.empty((0, 3))

    number_locations = len(np.ndarray.flatten(raw_validate[1, :, :]))

    for k in debiased_cms.keys():

        if len(debiased_cms[k])!=2:
            
            raise TypeError("Array of debiased climate datasets does not have correct length - should be two.")

        mean_bias = _calculate_mean_trend_bias(
            variable, "additive", raw_validate, raw_future, *debiased_cms[k]
        )
        
        lowqn_bias = _calculate_quantile_trend_bias(
            variable, "additive", 0.05, raw_validate, raw_future, *debiased_cms[k]
        )
        
        highqn_bias = _calculate_quantile_trend_bias(
            variable, "additive", 0.95, raw_validate, raw_future, *debiased_cms[k]
        )

        trend_bias_data = np.append(
            trend_bias_data,
            np.transpose(
                np.array(
                    [[k] * number_locations, ["Mean"] * number_locations, np.transpose(np.ndarray.flatten(mean_bias))]
                )
            ),
            axis=0,
        )

        trend_bias_data = np.append(
            trend_bias_data,
            np.transpose(
                np.array(
                    [[k] * number_locations, ["5% qn"] * number_locations, np.transpose(np.ndarray.flatten(lowqn_bias))]
                )
            ),
            axis=0,
        )

        trend_bias_data = np.append(
            trend_bias_data,
            np.transpose(
                np.array(
                    [
                        [k] * number_locations,
                        ["95% qn"] * number_locations,
                        np.transpose(np.ndarray.flatten(highqn_bias)),
                    ]
                )
            ),
            axis=0,
        )
        if len(metric_collection) != 0:
            for m in metric_collection:

                metric_bias = _calculate_metrics_trend_bias(variable, m, raw_validate, raw_future, *debiased_cms[k])

                trend_bias_data = np.append(
                    trend_bias_data,
                    np.transpose(
                        np.array(
                            [
                                [k] * number_locations,
                                [m.name] * number_locations,
                                np.transpose(np.ndarray.flatten(metric_bias)),
                            ]
                        )
                    ),
                    axis=0,
                )

    return trend_bias_data


def plot_future_trend_bias(variable: str, bias_array: np.ndarray):
    
    """
    Accepts ouput given by calculate_future_trend_bias (expects the following three columns: [Bias correction method, Metric, Bias value at certain location]),
    transforms it from a numpy error to a pandas dataframe and creates
    an overview boxplot of the bias in the trend of metrics existing in the input numpy array using seaborn.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    bias_array: np.ndarray
        Numpy array with three columns: [Bias correction method, Metric, Bias value at certain location]
    """
    
    if bias_array.shape[1]!=3:
        raise TypeError("Wrong input data format. Needs three columns Bias correction method, Metric, Bias value at certain location]. Use function calculate_future_trend_bias to generate input.")

    plot_data = pd.DataFrame(bias_array, columns=["Correction Method", "Metric", "Relative change bias (%)"])
    plot_data["Relative change bias (%)"] = pd.to_numeric(plot_data["Relative change bias (%)"])

    fig = plt.figure(figsize=(10, 6))
    ax = seaborn.boxplot(
        y="Relative change bias (%)", x="Metric", data=plot_data, palette="colorblind", hue="Correction Method"
    )
    [ax.axvline(x + 0.5, color="k") for x in ax.get_xticks()]
    fig.suptitle(
        "Bias in climate model trend between historical and future period \n {} ({})".format(
            map_variable_str_to_variable_class(variable).name,  map_variable_str_to_variable_class(variable).unit
        )
    )
