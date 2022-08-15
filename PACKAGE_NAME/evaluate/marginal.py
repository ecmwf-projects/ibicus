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

from PACKAGE_NAME.evaluate import metrics

variable_dictionary = {
    "tas": {
        "distribution": scipy.stats.norm,
        "trend_preservation": "additive",
        "detrending": True,
        "name": "2m daily mean air temperature (K)",
        "high_threshold": 295,
        "low_threshold": 273,
        "unit": "K",
    },
    "pr": {
        "distribution": scipy.stats.gamma,
        "trend_preservation": "mixed",
        "detrending": False,
        "name": "Total precipitation (m/day)",
        "high_threshold": 0.0004,
        "low_threshold": 0.00001,
        "unit": "m/day",
    },
}

metrics_dictionary = {
    "frost": {
        "variable": "tasmin",
        "variablename": "2m daily minimum air temperature (K)",
        "value": 273.15,
        "threshold_sign": "lower",
        "name": "Frost days",
    },
    "mean_warm_day": {
        "variable": "tas",
        "variablename": "2m daily mean air temperature (K)",
        "value": 295,
        "threshold_sign": "higher",
        "name": "Warm days (mean)",
    },
    "mean_cold_day": {
        "variable": "tas",
        "variablename": "2m daily mean air temperature (K)",
        "value": 273,
        "threshold_sign": "lower",
        "name": "Cold days (mean)",
    },
    "dry": {
        "variable": "pr",
        "variablename": "Precipitation",
        "value": 0.000001,
        "threshold_sign": "lower",
        "name": "Dry days (mean)",
    },
    "wet": {
        "variable": "pr",
        "variable_name": "Precipitation",
        "value": 1 / 86400,
        "threshold_sign": "higher",
        "name": "Wet days (daily total precipitation > 1 mm)",
    },
}


def _descriptive_statistics_marginal_bias(obs_data: np.ndarray, cm_data: np.ndarray):

    mean_obs = np.mean(obs_data, axis=0)
    lowpc_obs = np.quantile(obs_data, 0.05, axis=0)
    highpc_obs = np.quantile(obs_data, 0.95, axis=0)

    mean_bias = 100 * (mean_obs - np.mean(cm_data, axis=0)) / mean_obs

    if np.any(lowpc_obs) == 0:
        lowpc_bias = 0 * lowpc_obs
    else:
        lowpc_bias = 100 * (lowpc_obs - np.quantile(cm_data, 0.05, axis=0)) / lowpc_obs

    highpc_bias = 100 * (highpc_obs - np.quantile(cm_data, 0.95, axis=0)) / highpc_obs

    return (mean_bias, lowpc_bias, highpc_bias)


def _metrics_marginal_bias(metric: str, obs_data: np.ndarray, cm_data: np.ndarray):

    obs_metric = metrics.calculate_eot_probability(data=obs_data, threshold_name=metric)

    cm_metric = metrics.calculate_eot_probability(data=cm_data, threshold_name=metric)

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
    metrics: np.array
        Array of strings containing the names of the metrics that are to be assessed.
    obs_data: np.ndarray
        observational dataset in validation period
    **cm_data: 
        Keyword arguments of type debiaser_name = debiased_dataset in validation period (example: QM = tas_val_debiased_QM),
        covering all debiasers that are to be compared
    """

    marginal_bias_data = np.empty((0, 3))
    number_locations = len(np.ndarray.flatten(obs_data[1, :, :]))

    for k, cm_data in cm_data.items():

        mean_bias, lowqn_bias, highqn_bias = _descriptive_statistics_marginal_bias(
            obs_data=obs_data, cm_data=cm_data
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
                                [metrics_dictionary.get(m).get("name")] * number_locations,
                                np.transpose(np.ndarray.flatten(metric_bias)),
                            ]
                        )
                    ),
                    axis=0,
                )

    return marginal_bias_data


def plot_marginal_bias(variable: str, bias_array: np.ndarray, metrics: np.ndarray):

    """
    Takes numpy array containing the location-wise percentage bias of different metrics and outputs two boxplots,
    one for default descriptive statistics (mean, 5th and 95th quantile) and one for additional metrics.  

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    bias_array: np.ndarray
        Numpy array containing percentage bias for descriptive statistics and specified metrics. Has to be output of
        the function calculate_marginal_bias to be in correct format
    metrics: np.array
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

    fig.suptitle("{} - Bias".format(variable_dictionary.get(variable).get("name")))


def plot_histogram(variable: str, data_obs: np.ndarray, bin_number=100, **cm_data):

    """
    Plots histogram over entire area. Expects a one-dimensional array as input, so 2d lat-long array has to be flattened using
    for example np.ndarray.flatten. This plot will be more meaningful for smaller areas.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    data_obs: np.ndarray
        Flattened entry of all observed values over the area, numeric entries expected
    bin_number: int
        Number of bins plotted in histogram, set to 100 by default

    """

    number_biascorrections = len(debiased_cm_data.keys())
    figure_length = number_biascorrections * 5
    plot_number = number_biascorrections

    fig, ax = plt.subplots(1, plot_number, figsize=(figure_length, 5))
    fig.suptitle("Distribution {} over entire area".format(variable_dictionary.get(variable).get("name")))

    i = 0
    for k, cm_data in cm_data.items():

        ax[i].hist(data_obs, bins=bin_number, alpha=0.5, label="Observed")
        ax[i].hist(cm_data, bins=bin_number, alpha=0.5, label="Climate model")
        ax[i].set_title(k)
        ax[i].legend()
        i=i+1

    return fig
