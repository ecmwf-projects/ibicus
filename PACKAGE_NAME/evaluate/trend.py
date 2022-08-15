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


def calculate_descriptive_statistics_trend_bias(variable, trend_type, raw_validate, raw_future, bc_validate, bc_future):

    if trend_type == "additive":

        bc_trend_mean = np.mean(bc_future, axis=0) - np.mean(bc_validate, axis=0)
        raw_trend_mean = np.mean(raw_future, axis=0) - np.mean(raw_validate, axis=0)
        bias_mean = 100 * (bc_trend_mean - raw_trend_mean) / raw_trend_mean

        bc_trend_lowqn = np.quantile(bc_future, 0.05, axis=0) - np.quantile(bc_validate, 0.05, axis=0)
        raw_trend_lowqn = np.quantile(raw_future, 0.05, axis=0) - np.quantile(raw_validate, 0.05, axis=0)
        bias_lowqn = 100 * (bc_trend_lowqn - raw_trend_lowqn) / raw_trend_lowqn

        bc_trend_highqn = np.quantile(bc_future, 0.95, axis=0) - np.quantile(bc_validate, 0.95, axis=0)
        raw_trend_highqn = np.quantile(raw_future, 0.95, axis=0) - np.quantile(raw_validate, 0.95, axis=0)
        bias_highqn = 100 * (bc_trend_highqn - raw_trend_highqn) / raw_trend_highqn

    else:

        print("trend type currently not supported")

    return (bias_mean, bias_lowqn, bias_highqn)


def calculate_metrics_trend_bias(variable, metric, raw_validate, raw_future, bc_validate, bc_future):

    trend_raw = metrics.calculate_eot_probability(data=raw_future, threshold_name=metric) - metrics.calculate_probability(
        data=raw_validate, threshold_name=metric
    )

    trend_bc = metrics.calculate_eot_probability(data=bc_future, threshold_name=metric) - metrics.calculate_probability(
        data=bc_validate, threshold_name=metric
    )

    trend_bias = 100 * (trend_bc - trend_raw) / trend_raw

    return trend_bias


def calculate_future_trend_bias(variable, metrics, raw_validate, raw_future, **debiased_cms):

    # calculate 2d bias array for each of the metrics chosen, for each of the debiased cms, and append to numpy array

    trend_bias_data = np.empty((0, 3))

    number_locations = len(np.ndarray.flatten(raw_validate[1, :, :]))

    for k in debiased_cms.keys():

        # calculate mean, low quantile and high quantile

        mean_bias, lowqn_bias, highqn_bias = calculate_descriptive_statistics_trend_bias(
            variable, "additive", raw_validate, raw_future, *debiased_cms[k]
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
        if len(metrics) != 0:
            for m in metrics:

                metric_bias = calculate_metrics_trend_bias(variable, m, raw_validate, raw_future, *debiased_cms[k])

                trend_bias_data = np.append(
                    trend_bias_data,
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

    return trend_bias_data


def plot_future_trend_bias(variable, bias_array):

    plot_data = pd.DataFrame(bias_array, columns=["Correction Method", "Metric", "Relative change bias (%)"])
    plot_data["Relative change bias (%)"] = pd.to_numeric(plot_data["Relative change bias (%)"])

    fig = plt.figure(figsize=(10, 6))
    ax = seaborn.boxplot(
        y="Relative change bias (%)", x="Metric", data=plot_data, palette="colorblind", hue="Correction Method"
    )
    [ax.axvline(x + 0.5, color="k") for x in ax.get_xticks()]
    fig.suptitle(
        "Bias in climate model trend between historical and future period \n {}".format(
            variable_dictionary.get(variable).get("name")
        )
    )
