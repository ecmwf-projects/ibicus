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
import scipy.stats
import seaborn
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm

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


def calculate_aic_goodness_of_fit(variable, dataset, distribution_names=["default"]):

    aic = np.empty((0, 4))

    for distribution_name in distribution_names:

        if distribution_name == "default":
            distribution = variable_dictionary.get(variable).get("distribution")
        else:
            distribution = distribution_name

        for i in range(dataset.shape[1]):
            for j in range(dataset.shape[2]):

                fit = distribution.fit(dataset[:, i, j])

                k = len(fit)
                logLik = np.sum(distribution.logpdf(dataset[:, i, j], *fit))
                # logLik = np.sum(math.log(distribution.pdf(dataset[:, i, j], *fit)))
                aic_location = 2 * k - 2 * (logLik)

                aic = np.append(aic, [[i, j, aic_location, distribution_name]], axis=0)

    aic_dataframe = pd.DataFrame(aic, columns=["x", "y", "AIC_value", "Distribution"])
    aic_dataframe["AIC_value"] = pd.to_numeric(aic_dataframe["AIC_value"])

    return aic_dataframe


def plot_aic_goodness_of_fit(variable, aic_data):

    fig = plt.figure(figsize=(10, 6))

    seaborn.boxplot(data=aic_data, x="Distribution", y="AIC_value", palette="colorblind")
    
    return(fig)


def plot_worst_fit_aic(variable, dataset, aic, data_type, distribution_name="default", number_bins=100):

    if distribution_name == "default":
        distribution = variable_dictionary.get(variable).get("distribution")
    else:
        distribution = distribution_name

    x_location = aic.loc[aic["AIC_value"].idxmax()]["x"]
    y_location = aic.loc[aic["AIC_value"].idxmax()]["y"]
    data_slice = dataset[:, int(x_location), int(y_location)]

    fit = distribution.fit(data_slice)

    fig = plt.figure(figsize=(8, 6))

    plt.hist(data_slice, bins=number_bins, density=True, label=data_type, alpha=0.5)
    xmin, xmax = plt.xlim()

    x = np.linspace(xmin, xmax, 100)
    p = distribution.pdf(x, *fit)

    plt.plot(x, p, "k", linewidth=2)
    title = "{} {}, distribution = {} \n Location = ({}, {})".format(
        data_type, variable_dictionary.get(variable).get("name"), distribution_name, x_location, y_location
    )
    plt.title(title)

    return fig


def plot_quantile_residuals(dataset, variable, data_type, distribution_name="default"):

    if distribution_name == "default":
        distribution = variable_dictionary.get(variable).get("distribution")
    else:
        distribution = distribution_name

    fit = distribution.fit(dataset)
    q = distribution.cdf(dataset, *fit)

    q_normal = norm.ppf(q)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    fig.suptitle(
        "{} - {}. Distribution = {}".format(variable_dictionary.get(variable).get("name"), data_type, distribution_name)
    )

    x = range(0, len(q))
    ax[0].plot(x, q)
    ax[0].set_title("Quantile Residuals - Timeseries")

    plot_acf(q, lags=1000, ax=ax[1])
    ax[1].set_title("Quantile Residuals - ACF")

    sm.qqplot(q_normal, line="45", ax=ax[2])
    ax[2].set_title("Normalized Quantile Residuals - QQ Plot")

    return fig
