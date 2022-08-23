# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from PACKAGE_NAME.variables import *


def _calculate_aic(dataset, distribution):

    fit = distribution.fit(dataset)

    k = len(fit)
    logLik = np.sum(distribution.logpdf(dataset, *fit))

    return 2 * k - 2 * (logLik)


def calculate_aic(variable: str, dataset: np.ndarray, *distributions) -> pd.DataFrame:
    """
    Calculates the Akaike Information Criterion (AIC) at each location for each of the distributions specified.

    .. warning:: `*distributions` can currently only be :py:class:`scipy.stats.rv_continuous` and not as usually also :py:class:`StatisticalModel`.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    dataset : np.ndarray
        Input data, either observations or climate projections dataset to be analysed, numeric entries expected.
    *distributions : list[scipy.stats.rv_continuous]
        Distributions to be tested, elements are scipy.stats.rv_continuous

    Returns
    -------
    pd.DataFrame
        DataFrame with all locations, distributions and associated AIC values.
    """

    aic = []
    for distribution in distributions:
        for i, j in np.ndindex(dataset.shape[1:]):
            aic.append([i, j, _calculate_aic(dataset[:, i, j], distribution), distribution.name])

    aic_dataframe = pd.DataFrame(aic, columns=["x", "y", "AIC value", "Distribution"])
    aic_dataframe["AIC_value"] = pd.to_numeric(aic_dataframe["AIC value"])

    return aic_dataframe


def plot_aic(variable: str, aic_values: pd.DataFrame):
    """
    Creates a boxplot of AIC values across all locations.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    aic_values : pd.DataFrame
        Pandas dataframe of type of output by calculate_aic_goodness_of_fit.

    """

    return seaborn.boxplot(data=aic_values, x="Distribution", y="AIC value", palette="colorblind")


def plot_fit_worst_aic(
    variable: str,
    dataset: np.ndarray,
    data_type: str,
    distribution: scipy.stats.rv_continuous,
    nr_bins: Union[int, str] = "auto",
    aic_values: Optional[pd.DataFrame] = None,
):
    """
    Plots a histogram and overlayed fit at the location of worst AIC.

    .. warning:: `distribution` can currently only be :py:class:`scipy.stats.rv_continuous` and not as usually also :py:class:`StatisticalModel`.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    dataset : np.ndarray
        Input data, either observations or climate projections dataset to be analysed, numeric entries expected.
    data_type : pd.DataFrame
        Data type analysed - can be observational data or raw / debiased climate model data. Used to generate title only.
    distribution : scipy.stats.rv_continuous
        Distribution providing fit
    nr_bins : Union[int, str] = "auto"
        Number of bins used for the histogram. Either :py:class:int` or `"auto"` (default).
    aic_values : Optional[pd.DataFrame] = None
        Pandas dataframe of type output by calculate_aic_goodness_of_fit. If `None` then they are recalculated;
    """
    if aic_values is None:
        aic_values = calculate_aic(variable, dataset, distribution)

    x_location = aic_values.loc[aic_values["AIC_value"].idxmax()]["x"]
    y_location = aic_values.loc[aic_values["AIC_value"].idxmax()]["y"]
    data_slice = dataset[:, int(x_location), int(y_location)]

    fit = distribution.fit(data_slice)

    fig = plt.figure(figsize=(8, 6))

    plt.hist(data_slice, bins=nr_bins, density=True, label=data_type, alpha=0.5)
    xmin, xmax = plt.xlim()

    x = np.linspace(xmin, xmax, 100)
    p = distribution.pdf(x, *fit)

    plt.plot(x, p, "k", linewidth=2)
    title = "{} {} ({}), distribution = {} \n Location = ({}, {})".format(
        data_type,
        map_variable_str_to_variable_class(variable).name,
        map_variable_str_to_variable_class(variable).unit,
        distribution.name,
        x_location,
        y_location,
    )
    plt.title(title)

    return fig


def plot_quantile_residuals(
    variable: str, dataset: np.ndarray, distribution: scipy.stats.rv_continuous, data_type: str
):
    """
    Plots timeseries and autocorrelation function of quantile residuals, as well as a QQ-plot of normalized quantile residuals at one location.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    dataset : np.ndarray
        1d numpy array. Input data, either observations or climate projections dataset at one location, numeric entries expected.
    distribution: scipy.stats.rv_continuous
        Name of the distribution analysed, used for title only.
    data_type: str
        Data type analysed - can be observational data or raw / debiased climate model data. Used to generate title only.

    Example
    -------

    >>> tas_obs_plot_gof = assumptions.plot_quantile_residuals(variable = 'tas', dataset = tas_obs[:,0,0], distribution = scipy.stats.norm, data_type = 'observation data')

    """

    fit = distribution.fit(dataset)
    q = distribution.cdf(dataset, *fit)
    q_normal = scipy.stats.norm.ppf(q)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    fig.suptitle(
        "{} ({}) - {}. Distribution = {}".format(
            map_variable_str_to_variable_class(variable).name,
            map_variable_str_to_variable_class(variable).unit,
            data_type,
            distribution.name,
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
