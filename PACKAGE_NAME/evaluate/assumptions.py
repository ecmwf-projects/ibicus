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
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf

from PACKAGE_NAME.variables import *


def calculate_aic_goodness_of_fit(variable: str, dataset: np.ndarray, distribution_names: np.ndarray) -> pd.DataFrame:

    """
    Calculates Akaike Information Criterion at each location for each of the distributions specified.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    dataset : np.ndarray
        Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
    distribution_names:
        Distribution functions to be tested, elements are scipy.stats.rv_continuous

    """

    aic = np.empty((0, 4))

    for distribution_name in distribution_names:

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


def plot_aic_goodness_of_fit(variable: str, aic_data: pd.DataFrame):

    """
    Boxplot of AIC.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    aic_data: DataFrame
        Pandas dataframe of type output by calculate_aic_goodness_of_fit.

    """

    seaborn.boxplot(data=aic_data, x="Distribution", y="AIC_value", palette="colorblind")


def plot_worst_fit_aic(
    variable: str,
    dataset: np.ndarray,
    aic: np.ndarray,
    data_type: str,
    distribution_name: scipy.stats.rv_continuous,
    number_bins=100,
):

    """
    Plots histogram and fit at location of worst AIC.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    dataset : np.ndarray
        Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
    aic: pd.DataFrame
        Pandas dataframe of type output by calculate_aic_goodness_of_fit.
    data_type : pd.DataFrame
        Data type analysed - can be observational data or raw / debiased climate model data. Used to generate title only.
    distribution_name : scipy.rv_continuous
        Name of the distribution analysed, used for title only.

    """

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
    title = "{} {} ({}), distribution = {} \n Location = ({}, {})".format(
        data_type,
        map_variable_str_to_variable_class(variable).name,
        map_variable_str_to_variable_class(variable).unit,
        distribution_name,
        x_location,
        y_location,
    )
    plt.title(title)

    return fig


def plot_quantile_residuals(
    variable: str, dataset: np.ndarray, data_type: str, distribution_name: scipy.stats.rv_continuous
):

    """
    Plots timeseries and autocorrelation function of quantile residuals, as well as QQ-plot of normalized quantile residuals at one location

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    dataset : np.ndarray
        Input data, either observations or climate projectionsdataset at one location, numeric entries expected.
    data_type: str
        Data type analysed - can be observational data or raw / debiased climate model data. Used to generate title only.
    distribution_name: scipy.stats.rv_continuous
        Name of the distribution analysed, used for title only.

    Example code:

    >>> tas_obs_plot_gof = assumptions.plot_quantile_residuals(variable = 'tas', dataset = tas_obs[:,0,0], data_type = 'observation data', distribution_name = scipy.stats.norm)

    """

    distribution = distribution_name

    fit = distribution.fit(dataset)
    q = distribution.cdf(dataset, *fit)

    q_normal = norm.ppf(q)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    fig.suptitle(
        "{} ({}) - {}. Distribution = {}".format(
            map_variable_str_to_variable_class(variable).name,
            map_variable_str_to_variable_class(variable).unit,
            data_type,
            distribution_name,
        )
    )

    x = range(0, len(q))
    ax[0].plot(x, q)
    ax[0].set_title("Quantile Residuals - Timeseries")

    plot_acf(q, lags=1000, ax=ax[1])
    ax[1].set_title("Quantile Residuals - ACF")

    sm.qqplot(q_normal, line="45", ax=ax[2])
    ax[2].set_title("Normalized Quantile Residuals - QQ Plot")

    return fig
