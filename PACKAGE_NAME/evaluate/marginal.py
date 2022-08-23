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
import seaborn

from PACKAGE_NAME.variables import *

def _mean_marginal_bias(obs_data: np.ndarray, cm_data: np.ndarray):
    
    """
    Calculates location-wise percentage bias of mean
    """

    mean_bias = 100 * (np.mean(obs_data, axis=0) - np.mean(cm_data, axis=0)) / np.mean(obs_data, axis=0)

    return mean_bias


def _quantile_marginal_bias(quantile: float, obs_data: np.ndarray, cm_data: np.ndarray):
    
    """
    Calculates location-wise percentage bias of specified quantile
    """

    if quantile<0 or quantile>1:
        raise ValueError('Quantile needs to be between 0 and 1')

    qn_obs = np.quantile(obs_data, quantile, axis=0)

    if np.any(qn_obs) == 0:
        qn_bias = 0 * qn_obs
    else:
        qn_bias = 100 * (qn_obs - np.quantile(cm_data, quantile, axis=0)) / qn_obs

    return qn_bias


def _metrics_marginal_bias(metric, obs_data: np.ndarray, cm_data: np.ndarray):
    
    """
    Calculates location-wise percentage bias of metric specified
    """

    obs_metric = metric.calculate_exceedance_probability(dataset=obs_data)

    cm_metric = metric.calculate_exceedance_probability(dataset=cm_data)

    bias = 100 * (obs_metric - cm_metric) / obs_metric

    return bias

def calculate_marginal_bias(metrics: np.ndarray, obs_data: np.ndarray, **cm_data) -> np.ndarray:
    
    """
    Calculates location-wise percentage bias of different metrics, comparing observations to climate model output during the validation period.
    Default metrics include mean, 5th and 95th perecentile, 
    calculated in _descriptive_statistics_marginal_bias. Additional metrics can be specified in the metrics input argument,
    bias is calculated in _metrics_marginal_bias
    
    Parameters
    ----------
    metrics : np.array
        Array of strings containing the names of the metrics that are to be assessed.
    obs_data : np.ndarray
        observational dataset in validation period
    **cm_data : 
        Keyword arguments of type debiaser_name = debiased_dataset in validation period (example: QM = tas_val_debiased_QM),
        covering all debiasers that are to be compared
    """

    marginal_bias_data = np.empty((0, 3))
    number_locations = len(np.ndarray.flatten(obs_data[1, :, :]))

    for k, cm_data in cm_data.items():

        mean_bias = _mean_marginal_bias(
            obs_data=obs_data, cm_data=cm_data
        )
        lowqn_bias = _quantile_marginal_bias(
            quantile = 0.05, obs_data=obs_data, cm_data=cm_data
        )
        highqn_bias = _quantile_marginal_bias(
            quantile = 0.95, obs_data=obs_data, cm_data=cm_data
        )

        marginal_bias_data = np.append(
            marginal_bias_data,
            np.transpose(
                np.array(
                    [[k] * number_locations, ["Mean"] * number_locations, np.transpose(np.ndarray.flatten(mean_bias))]
                )
            ),
            axis=0,
        )

        marginal_bias_data = np.append(
            marginal_bias_data,
            np.transpose(
                np.array(
                    [[k] * number_locations, ["5% qn"] * number_locations, np.transpose(np.ndarray.flatten(lowqn_bias))]
                )
            ),
            axis=0,
        )

        marginal_bias_data = np.append(
            marginal_bias_data,
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
        if len(metrics) != 0:

            for m in metrics:

                metric_bias = _metrics_marginal_bias(m, obs_data, cm_data)

                marginal_bias_data = np.append(
                    marginal_bias_data,
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

    return marginal_bias_data


def plot_marginal_bias(variable: str, bias_array: np.ndarray):

    """
    Takes numpy array containing the location-wise percentage bias of different metrics and outputs two boxplots,
    one for default descriptive statistics (mean, 5th and 95th quantile) and one for additional metrics.  

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    bias_array : np.ndarray
        Numpy array containing percentage bias for descriptive statistics and specified metrics. Has to be output of
        the function calculate_marginal_bias to be in correct format
    metrics : np.ndarray
        Array of strings containing the names of the metrics that are to be plotted.

    """


    plot_data = pd.DataFrame(bias_array, columns=["Correction Method", "Metric", "Percentage bias"])
    plot_data["Percentage bias"] = pd.to_numeric(plot_data["Percentage bias"])

    plot_data1 = plot_data[plot_data["Metric"].isin(["Mean", "5% qn", "95% qn"])]
    plot_data2 = plot_data[~plot_data["Metric"].isin(["Mean", "5% qn", "95% qn"])]

    fig_width = 2 * plot_data["Metric"].nunique() + 3

    fig, ax = plt.subplots(1, 2, figsize=(fig_width, 6))

    seaborn.violinplot(
        ax=ax[0], y="Percentage bias", x="Metric", data=plot_data1, palette="colorblind", hue="Correction Method"
    )
    [ax[0].axvline(x + 0.5, color="k") for x in ax[0].get_xticks()]

    seaborn.violinplot(
        ax=ax[1], y="Percentage bias", x="Metric", data=plot_data2, palette="colorblind", hue="Correction Method"
    )
    [ax[1].axvline(x + 0.5, color="k") for x in ax[1].get_xticks()]

    fig.suptitle("{} ({}) - Bias".format(map_variable_str_to_variable_class(variable).name, map_variable_str_to_variable_class(variable).unit))


def plot_histogram(variable: str, data_obs: np.ndarray, bin_number=100, **cm_data):

    """
    Plots histogram over entire area. Expects a one-dimensional array as input, so 2d lat-long array has to be flattened using
    for example np.ndarray.flatten. This plot will be more meaningful for smaller areas.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    data_obs : np.ndarray
        Flattened entry of all observed values over the area, numeric entries expected
    bin_number : int
        Number of bins plotted in histogram, set to 100 by default

    """

    number_biascorrections = len(debiased_cm_data.keys())
    figure_length = number_biascorrections * 5
    plot_number = number_biascorrections

    fig, ax = plt.subplots(1, plot_number, figsize=(figure_length, 5))
    fig.suptitle("Distribution {} ({}) over entire area".format(map_variable_str_to_variable_class(variable).name, map_variable_str_to_variable_class(variable).unit))

    i = 0
    for k, cm_data in cm_data.items():

        ax[i].hist(data_obs, bins=bin_number, alpha=0.5, label="Observed")
        ax[i].hist(cm_data, bins=bin_number, alpha=0.5, label="Climate model")
        ax[i].set_title(k)
        ax[i].legend()
        i=i+1

    return fig
