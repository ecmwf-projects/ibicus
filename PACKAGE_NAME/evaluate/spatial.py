# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from itertools import chain
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from PACKAGE_NAME.evaluate import marginal
from PACKAGE_NAME.variables import *


def plot_bias_spatial(variable: str, statistic, metric, obs_data: np.ndarray, **cm_data):
    """
    Calculates and plots the bias at each location with respect to thhe specified metric.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    metric: Union(str, float)
        Specifies the metric analysed. Should be one of the following: 'mean' (str), quantile value (float between 0 and 1),
        or metric key (str) specified in the metrics dictionary
    obs_data: np.ndarray
        Observation data in validation period
    cm_data:
        Keyword arguments specifying debiasers to be analysed. Example: QM = tas_val_debiased_QM or raw = tas_cm_validate

    """

    plot_data = {}

    for key, value in cm_data.items():

        if statistic == "mean":
            plot_data[key] = marginal._mean_marginal_bias(obs_data, value)
            title = "{} ({}) \n Percentage bias of mean".format(
                map_variable_str_to_variable_class(variable).name, map_variable_str_to_variable_class(variable).unit
            )

        elif statistic.isnumeric():

            plot_data[key] = marginal._quantile_marginal_bias(metric, obs_data, value)
            title = "{} ({}) \n Percentage bias of {} % quantile ".format(
                map_variable_str_to_variable_class(variable).name,
                map_variable_str_to_variable_class(variable).unit,
                str(100 * metric),
            )

        elif statistic == "metric":

            plot_data[key] = marginal._metrics_marginal_bias(metric=metric, obs_data=obs_data, cm_data=value)
            title = "{} \n Percentage bias".format(metric.name)

        else:
            raise ValueError(
                "Metric not recognized, choose either a metric from the metrics dictionary, the mean, or a percentile."
            )

    axis_max = max(
        abs(max(np.ndarray.flatten(np.vstack(list(chain(*plot_data.values())))))),
        abs(min(np.ndarray.flatten(np.vstack(list(chain(*plot_data.values())))))),
    )
    axis_min = -axis_max

    fig_width = 6 * len(cm_data.keys())
    fig, ax = plt.subplots(1, len(cm_data.keys()), figsize=(fig_width, 5))
    fig.suptitle(title)

    i = 0
    for k in cm_data.keys():
        plot = ax[i].imshow(plot_data[k], cmap=plt.get_cmap("coolwarm"), vmin=axis_min, vmax=axis_max)
        ax[i].set_title("{}".format(k))
        fig.colorbar(plot, ax=ax[i])
        i = i + 1
