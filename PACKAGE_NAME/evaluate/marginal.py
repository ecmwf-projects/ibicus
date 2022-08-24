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

    if quantile < 0 or quantile > 1:
        raise ValueError("quantile needs to be between 0 and 1")

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


def calculate_marginal_bias(obs_data: np.ndarray, metrics: list = [], **cm_data) -> np.ndarray:
    """
    Return a :py:class:`pd.DataFrame` containing location-wise percentage bias of different metrics: mean, 5th and 95th percentile, as well as metrics specific in `metrics`, comparing observations to climate model output during a validation period.

    Parameters
    ----------
    obs_data : np.ndarray
        observational dataset in validation period
    metrics : list
        Array of strings containing the names of the metrics that are to be assessed.
    **cm_data :
        Keyword arguments of type debiaser_name = debiased_dataset in validation period (example: QM = tas_val_debiased_QM),
        covering all debiasers that are to be compared
    """

    marginal_bias_dfs = []
    number_locations = len(np.ndarray.flatten(obs_data[1, :, :]))

    for cm_data_key, cm_data in cm_data.items():

        mean_bias = _mean_marginal_bias(obs_data=obs_data, cm_data=cm_data)
        marginal_bias_dfs.append(
            pd.DataFrame(
                data={
                    "Correction Method": [cm_data_key] * number_locations,
                    "Metric": ["Mean"] * number_locations,
                    "Percentage bias": mean_bias.flatten(),
                }
            )
        )

        lowqn_bias = _quantile_marginal_bias(quantile=0.05, obs_data=obs_data, cm_data=cm_data)
        marginal_bias_dfs.append(
            pd.DataFrame(
                data={
                    "Correction Method": [cm_data_key] * number_locations,
                    "Metric": ["5% qn"] * number_locations,
                    "Percentage bias": lowqn_bias.flatten(),
                }
            )
        )

        highqn_bias = _quantile_marginal_bias(quantile=0.95, obs_data=obs_data, cm_data=cm_data)
        marginal_bias_dfs.append(
            pd.DataFrame(
                data={
                    "Correction Method": [cm_data_key] * number_locations,
                    "Metric": ["95% qn"] * number_locations,
                    "Percentage bias": highqn_bias.flatten(),
                }
            )
        )

        for m in metrics:

            metric_bias = _metrics_marginal_bias(m, obs_data, cm_data)

            marginal_bias_dfs.append(
                pd.DataFrame(
                    data={
                        "Correction Method": [cm_data_key] * number_locations,
                        "Metric": [m.name] * number_locations,
                        "Percentage bias": metric_bias.flatten(),
                    }
                )
            )

    plot_data = pd.concat(marginal_bias_dfs)
    plot_data["Percentage bias"] = pd.to_numeric(plot_data["Percentage bias"])
    return plot_data


def plot_marginal_bias(variable: str, bias_df: pd.DataFrame):
    """
    Returns boxplots plotting the distribution of location-wise percentage bias of different metrics, passed in `bias_array` and as calculated by `calculate_marginal_bias`.

    Two wo boxplots are created: one for default descriptive statistics (mean, 5th and 95th quantile) and one for additional metrics.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    bias_df : pd.DataFrame
        :py:class:`pd.DataFrame` containing percentage bias for descriptive statistics and specified metrics. Output of :py:func:`calculate_marginal_bias`.
    """

    plot_data1 = bias_df[bias_df["Metric"].isin(["Mean", "5% qn", "95% qn"])]
    plot_data2 = bias_df[~bias_df["Metric"].isin(["Mean", "5% qn", "95% qn"])]

    fig_width = 2 * bias_df["Metric"].nunique() + 3

    fig, ax = plt.subplots(1, 2, figsize=(fig_width, 6))

    seaborn.violinplot(
        ax=ax[0], y="Percentage bias", x="Metric", data=plot_data1, palette="colorblind", hue="Correction Method"
    )
    [ax[0].axvline(x + 0.5, color="k") for x in ax[0].get_xticks()]

    seaborn.violinplot(
        ax=ax[1], y="Percentage bias", x="Metric", data=plot_data2, palette="colorblind", hue="Correction Method"
    )
    [ax[1].axvline(x + 0.5, color="k") for x in ax[1].get_xticks()]

    fig.suptitle(
        "{} ({}) - Bias".format(
            map_variable_str_to_variable_class(variable).name, map_variable_str_to_variable_class(variable).unit
        )
    )


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

    number_biascorrections = len(cm_data.keys())
    figure_length = number_biascorrections * 5
    plot_number = number_biascorrections

    fig, ax = plt.subplots(1, plot_number, figsize=(figure_length, 5), squeeze=True)
    print(ax)
    fig.suptitle(
        "Distribution {} ({}) over entire area".format(
            map_variable_str_to_variable_class(variable).name, map_variable_str_to_variable_class(variable).unit
        )
    )

    i = 0
    for k, cm_data in cm_data.items():

        ax[i].hist(data_obs, bins=bin_number, alpha=0.5, label="Observed")
        ax[i].hist(cm_data, bins=bin_number, alpha=0.5, label="Climate model")
        ax[i].set_title(k)
        ax[i].legend()
        i = i + 1

    return fig
