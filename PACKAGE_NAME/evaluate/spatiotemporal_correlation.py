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

import math
import sklearn.metrics

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


def rmse_spatial_correlation_distribution(variable, obs_data, **cm_data):

    rmsd_array = np.empty((0, 2))

    for k in cm_data.keys():

        rmsd = np.zeros((obs_data.shape[1], obs_data.shape[2]))

        for a in range(obs_data.shape[1]):
            for b in range(obs_data.shape[2]):

                corr_matrix_obs = np.zeros((obs_data.shape[1], obs_data.shape[2]))
                corr_matrix_cm = np.zeros((obs_data.shape[1], obs_data.shape[2]))

                for i in range(obs_data.shape[1]):
                    for j in range(obs_data.shape[2]):

                        corr_matrix_obs[i, j] = np.corrcoef(obs_data[:, a, b], obs_data[:, i, j])[0, 1]
                        corr_matrix_cm[i, j] = np.corrcoef(cm_data[k][:, a, b], cm_data[k][:, i, j])[0, 1]

                rmsd[a, b] = math.sqrt(sklearn.metrics.mean_squared_error(corr_matrix_obs, corr_matrix_cm))

        array = np.transpose(np.array([[k] * len(np.ndarray.flatten(rmsd)), np.transpose(np.ndarray.flatten(rmsd))]))

        rmsd_array = np.append(rmsd_array, array, axis=0)

    rmsd_data = pd.DataFrame(rmsd_array, columns=["Correction Method", "RMSE spatial correlation"])
    rmsd_data["RMSE spatial correlation"] = pd.to_numeric(rmsd_data["RMSE spatial correlation"])

    return rmsd_data


def rmse_spatial_correlation_boxplot(variable, dataset):

    fig = plt.figure(figsize=(8, 6))
    seaborn.boxplot(y="RMSE spatial correlation", x="Correction Method", data=dataset, palette="colorblind")

    fig.suptitle("{} \n RMSE of spatial correlation matrices)".format(variable_dictionary.get(variable).get("name")))

    return fig
