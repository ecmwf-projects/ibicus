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
from pylab import arange
from scipy.ndimage import measurements

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


def calculate_spell_length(metric, min_length, **climate_data):

    spell_length_array = np.empty((0, 3))

    for k in climate_data.keys():

        threshold_data = metrics.calculate_eot_matrix(climate_data[k], metric)
        spell_length = np.array([])

        for i in range(threshold_data.shape[1]):
            for j in range(threshold_data.shape[2]):
                N = 0
                for t in range(threshold_data.shape[0]):
                    if threshold_data[t, i, j] == 1:
                        N = N + 1
                    elif (threshold_data[t, i, j] == 0) and (N != 0):
                        spell_length = np.append(spell_length, N)
                        N = 0

        spell_length = spell_length[spell_length > min_length]

        spell_length_array = np.append(
            spell_length_array,
            np.transpose(
                np.array(
                    [
                        [k] * len(spell_length),
                        [metrics_dictionary.get(metric).get("name")] * len(spell_length),
                        np.transpose(spell_length),
                    ]
                )
            ),
            axis=0,
        )

    plot_data = pd.DataFrame(spell_length_array, columns=["Correction Method", "Metric", "Spell length (days)"])
    plot_data["Spell length (days)"] = pd.to_numeric(plot_data["Spell length (days)"])

    return plot_data


def calculate_spatiotemporal_clusters(metric, **climate_data):

    clusters_array = np.empty((0, 3))

    for k in climate_data.keys():

        threshold_data = metrics.calculate_eot_matrix(dataset=climate_data[k], thresholdname=metric)
        threshold_data_lw, threshold_data_num = measurements.label(threshold_data)
        area = measurements.sum(threshold_data, threshold_data_lw, index=arange(threshold_data_lw.max() + 1))

        clusters_array = np.append(
            clusters_array,
            np.transpose(
                np.array(
                    [[k] * len(area), [metrics_dictionary.get(metric).get("name")] * len(area), np.transpose(area)]
                )
            ),
            axis=0,
        )

    spatiotemporal_clusters = pd.DataFrame(
        clusters_array, columns=["Correction Method", "Metric", "Spatiotemporal cluster size"]
    )
    spatiotemporal_clusters["Spatiotemporal cluster size"] = pd.to_numeric(
        spatiotemporal_clusters["Spatiotemporal cluster size"]
    )

    return spatiotemporal_clusters


def calculate_spatial_clusters(metric, **climate_data):

    clusters_array = np.empty((0, 3))

    for k in climate_data.keys():

        spatial_count = np.array([])

        number_gridpoints = climate_data[k].shape[1] * climate_data[k].shape[2]

        threshold_data = metrics.calculate_eot_matrix(dataset=climate_data[k], thresholdname=metric)

        for i in range(threshold_data.shape[0]):

            count = np.sum(threshold_data[i, :, :]) / number_gridpoints
            spatial_count = np.append(spatial_count, count)

        spatial_count = spatial_count[spatial_count != 0]

        clusters_array = np.append(
            clusters_array,
            np.transpose(
                np.array(
                    [
                        [k] * len(spatial_count),
                        [metrics_dictionary.get(metric).get("name")] * len(spatial_count),
                        np.transpose(spatial_count),
                    ]
                )
            ),
            axis=0,
        )

    spatial_clusters = pd.DataFrame(
        clusters_array, columns=["Correction Method", "Metric", "Spatial extent (% of area)"]
    )
    spatial_clusters["Spatial extent (% of area)"] = pd.to_numeric(spatial_clusters["Spatial extent (% of area)"])

    return spatial_clusters


def calculate_clusters(metric, min_length, **climate_data):

    temporal_data = calculate_spell_length(metric, min_length, **climate_data)
    spatial_data = calculate_spatial_clusters(metric, **climate_data)
    spatiotemporal_data = calculate_spatiotemporal_clusters(metric, **climate_data)

    return (temporal_data, spatial_data, spatiotemporal_data)


def plot_clusters_distribution(thresholdname, plot_data, clustertype):

    seaborn.set_style("white")
    p = seaborn.displot(
        x=plot_data.keys()[2], data=plot_data, kind="kde", palette="colorblind", hue="Correction Method"
    )
    p.fig.subplots_adjust(top=0.9)
    p.fig.suptitle("{} - {} distribution".format(metrics_dictionary.get(thresholdname).get("name"), clustertype))


def plot_clusters_distribution_histograms(plot_data, metric, debiasers, clustertype):

    plot_data_subset = {}

    plot_number = len(debiasers)
    fig_width = plot_number * 5

    for debiaser in debiasers:
        plot_data_subset[debiaser] = plot_data[plot_data["Correction Method"].isin(["obs", "raw", debiaser])]

    fig, ax = plt.subplots(1, plot_number, figsize=(fig_width, 6))

    i = 0
    for debiaser in debiasers:

        seaborn.histplot(
            ax=ax[i], data=plot_data_subset[debiaser], x="Spell length", palette="colorblind", hue="Correction Method"
        )
        ax[i].set_title(debiaser)
        i = i + 1

    fig.suptitle("{} - {} distribution".format(metrics_dictionary.get(metric).get("name"), clustertype))
    return fig


def plot_extent_distributions(temporal_data, spatial_data, spatiotemporal_data):

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    seaborn.violinplot(
        ax=ax[0], data=temporal_data, x="Metric", y="Spell length (days)", palette="colorblind", hue="Correction Method"
    )
    ax[0].set_title("Spell length (days)")

    seaborn.violinplot(
        ax=ax[1],
        data=spatial_data,
        x="Metric",
        y="Spatial extent (% of area)",
        palette="colorblind",
        hue="Correction Method",
    )
    ax[1].set_title("Spatial extent (% of area)")

    seaborn.violinplot(
        ax=ax[2],
        data=spatiotemporal_data,
        x="Metric",
        y="Spatiotemporal cluster size",
        palette="colorblind",
        hue="Correction Method",
    )
    ax[2].set_title("Spatiotemporal cluster size")
