# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import scipy

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


def plot_bias_metrics(threshold_name, data_obs, **cm_data):

    threshold_obs = metrics.calculate_probability(data=data_obs, threshold_name=threshold_name)
    bias = {}

    for k in cm_data.keys():

        bias[k] = (
            100
            * (threshold_obs - metrics.calculate_probability(data=cm_data[k], threshold_name=threshold_name))
            / threshold_obs
        )

    axis_max = max(
        abs(max(np.ndarray.flatten(np.vstack(list(chain(*bias.values())))))),
        abs(min(np.ndarray.flatten(np.vstack(list(chain(*bias.values())))))),
    )
    axis_min = -axis_max

    fig_width = 6 * len(cm_data.keys())
    fig, ax = plt.subplots(1, len(cm_data.keys()), figsize=(fig_width, 5))
    fig.suptitle("{} - percentage bias".format(metrics_dictionary.get(threshold_name).get("name")))

    i = 0
    for k in cm_data.keys():

        plot = ax[i].imshow(bias[k], cmap=plt.get_cmap("coolwarm"), vmin=axis_min, vmax=axis_max)
        ax[i].set_title("{}".format(k))
        fig.colorbar(plot, ax=ax[i])
        i = i + 1

    # return(fig)


def plot_bias_ds(variable, data_obs, **cm_data):

    """
    Calculates the bias of the mean, 10th percentile and 90th percentile between the
    observational and climate model data at each location and plots their spatial distribution. Function is intended to be applied to data in the validation
    period.

    Parameters
    ----------
    variable : str, variable name is standard form (i.e. 'tas', 'pr', etc)
    data_obs : three-dimensional array (time, latitude, longitude) of observational data in validation period, numeric entries expected
    data_raw : three-dimensional array (time, latitude, longitude) of raw (i.e. not bias corrected) climate model data in validation period, numeric entries expected
    **kwargs: three-dimensional array (time, latitude, longitude) of bias corrected data sets. To be given in the form bias_correction_name = bias_corrected_dataset,
    the latter being of the same form as data_obs and data_raw, numeric entries expected.
    """

    fig_width = 6 * len(cm_data.keys())
    plot_number = len(cm_data.keys())

    mean_obs = np.mean(data_obs, axis=0)
    lowpc_obs = np.quantile(data_obs, 0.1, axis=0)
    highpc_obs = np.quantile(data_obs, 0.9, axis=0)

    bias_mean = {}
    bias_lowpc = {}
    bias_highpc = {}

    for k in cm_data.keys():

        bias_mean[str(k)] = 100 * (mean_obs - np.mean(cm_data[k], axis=0)) / mean_obs
        bias_lowpc[str(k)] = 100 * (lowpc_obs - np.quantile(cm_data[k], 0.1, axis=0)) / lowpc_obs
        bias_highpc[str(k)] = 100 * (highpc_obs - np.quantile(cm_data[k], 0.9, axis=0)) / highpc_obs

    arrays_max = max(
        max(np.ndarray.flatten(np.vstack(list(chain(*bias_mean.values()))))),
        max(np.ndarray.flatten(np.vstack(list(chain(*bias_lowpc.values()))))),
        max(np.ndarray.flatten(np.vstack(list(chain(*bias_highpc.values()))))),
    )

    arrays_min = min(
        min(np.ndarray.flatten(np.vstack(list(chain(*bias_mean.values()))))),
        min(np.ndarray.flatten(np.vstack(list(chain(*bias_lowpc.values()))))),
        min(np.ndarray.flatten(np.vstack(list(chain(*bias_highpc.values()))))),
    )

    axis_max = max(abs(arrays_max), abs(arrays_min))
    axis_min = -axis_max

    fig, ax = plt.subplots(3, plot_number, figsize=(fig_width, 15))

    fig.suptitle("{} - Bias".format(variable_dictionary.get(variable).get("name")))

    i = 0
    for k in cm_data.keys():

        plot1 = ax[0, i].imshow(bias_mean[k], cmap=plt.get_cmap("coolwarm"), vmin=axis_min, vmax=axis_max)
        ax[0, i].set_title("% bias of mean \n {}".format(k))
        fig.colorbar(plot1, ax=ax[0, i])

        plot2 = ax[1, i].imshow(bias_lowpc[k], cmap=plt.get_cmap("coolwarm"), vmin=axis_min, vmax=axis_max)
        ax[1, i].set_title("% bias of 5th percentile \n {}".format(k))
        fig.colorbar(plot2, ax=ax[1, i])

        plot3 = ax[2, i].imshow(bias_highpc[k], cmap=plt.get_cmap("coolwarm"), vmin=axis_min, vmax=axis_max)
        ax[2, i].set_title("% bias of 95th percentile \n {}".format(k))
        fig.colorbar(plot3, ax=ax[2, i])

        i = i + 1

    return fig
