# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import sklearn.metrics
from sklearn.metrics import mutual_info_score

from PACKAGE_NAME.variables import *


def rmse_spatial_correlation_distribution(variable: str, obs_data: np.ndarray, **cm_data) -> pd.DataFrame:

    """
    Calculate Root-Mean-Squared-Error between observed and modelled spatial correlation matrix at each location.

    The computation involves the following steps: At each location, calculate the correlation to each other location in the observed as well as the climate model
    data set. Then calculate the mean squared error between these two matrices.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    dataset : np.ndarray
        Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
    distribution_names:
        Distribution functions to be tested.

    """

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


def rmse_spatial_correlation_boxplot(variable: str, dataset: pd.DataFrame):

    """
    Boxplot of RMSE of spatial correlation across locations.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    dataset : pd.DataFrame
        Ouput format of function rmse_spatial_correlation_distribution

    """

    fig = plt.figure(figsize=(8, 6))
    seaborn.boxplot(y="RMSE spatial correlation", x="Correction Method", data=dataset, palette="colorblind")

    fig.suptitle(
        "{} ({}) \n RMSE of spatial correlation matrices".format(
            map_variable_str_to_variable_class(variable).name, map_variable_str_to_variable_class(variable).unit
        )
    )

    return fig


def calculate_multivariate_correlation_locationwise(variables, **kwargs):

    """
    Calculates correlation between two variables specified in keyword arguments at each location and outputs spatial plot.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    kwargs :
        Keyword arguments specifying a list of two np.ndarrays containing the two variables of interest.

    Example code:
    >>> correlation.calculate_multivariate_correlation_locationwise(variables = ['tas', 'pr'], obs = [tas_obs_validate, pr_obs_validate], raw = [tas_cm_validate, pr_cm_validate], ISIMIP = [tas_val_debiased_ISIMIP, pr_val_debiased_ISIMIP])

    """

    correlation_matrix = {}

    for k in kwargs.keys():

        variable1 = kwargs[k][0]
        variable2 = kwargs[k][1]

        correlation_matrix[k] = np.zeros((variable1.shape[1], variable1.shape[2]))

        for i in range(variable1.shape[1]):
            for j in range(variable1.shape[2]):

                correlation_matrix[k][i, j] = np.corrcoef(variable1[:, i, j].T, variable2[:, i, j].T)[0, 1]

    axis_max = max(
        abs(max(np.ndarray.flatten(np.vstack(list(chain(*correlation_matrix.values())))))),
        abs(min(np.ndarray.flatten(np.vstack(list(chain(*correlation_matrix.values())))))),
    )
    axis_min = -axis_max

    fig_width = 6 * len(kwargs.keys())
    fig, ax = plt.subplots(1, len(kwargs.keys()), figsize=(fig_width, 5))
    fig.suptitle(
        "Multivariate Correlation: {} and {}".format(
            variable_dictionary.get(variables[0]).get("name"), variable_dictionary.get(variables[1]).get("name")
        )
    )

    i = 0
    for k in kwargs.keys():

        plot = ax[i].imshow(correlation_matrix[k], cmap=plt.get_cmap("coolwarm"), vmin=axis_min, vmax=axis_max)
        ax[i].set_title("{}".format(k))
        fig.colorbar(plot, ax=ax[i])
        i = i + 1

    return fig


def define_multivariate_dataframes(variables: list, datasets_obs: list, datasets_bc: list, gridpoint=(0, 0)):

    """
    Convert 2d np.ndarrays into pd.DataFrame in order to use plot_direction_comparison_of_correlation_at_gridpoint, calculate_corr_bs_replicates and calculate_bs_replicates_mi

    Parameters
    ----------
    variable : list
        List of variable names, has to be given in standard form specified in documentation.
    kwargs :
        Keyword arguments specifying a list of two np.ndarrays containing the two variables of interest.

    Example code:
    >>> correlation.calculate_multivariate_correlation_locationwise(variables = ['tas', 'pr'], obs = [tas_obs_validate, pr_obs_validate], raw = [tas_cm_validate, pr_cm_validate], ISIMIP = [tas_val_debiased_ISIMIP, pr_val_debiased_ISIMIP])

    """

    obs_dataframe = pd.DataFrame(columns=[variables[0], variables[1]])
    obs_dataframe[variables[0]] = datasets_obs[0][:, 0, 0]
    obs_dataframe[variables[1]] = datasets_obs[1][:, 0, 0]

    bc_dataframe = pd.DataFrame(columns=[variables[0], variables[1]])
    bc_dataframe[variables[0]] = datasets_bc[0][:, 0, 0]
    bc_dataframe[variables[1]] = datasets_bc[1][:, 0, 0]

    return (obs_dataframe, bc_dataframe)


def plot_direct_comparison_of_correlation_at_gridpoint(
    variables: list, obs_dataframe: pd.DataFrame, bc_dataframe: pd.DataFrame
):

    """

    Uses function seaborn.regplot to plot scatterplot and Pearson correlation estimate of the two specified variables. Offers visual comparison of correlation at single location.

    Parameters
    ----------
    variable : list
        List of variable name, has to be given in standard form specified in documentation.
    kwargs :
        Keyword arguments specifying a list of two np.ndarrays containing the two variables of interest.

    """

    seaborn.set_style("white")

    seaborn.regplot(x=variables[0], y=variables[1], data=obs_dataframe, ci=95, scatter_kws={"s": 0.8, "color": "g"})

    seaborn.regplot(x=variables[0], y=variables[1], data=bc_dataframe, ci=95, scatter_kws={"s": 0.8, "color": "r"})


def calculate_corr_bs_replicates(data: pd.DataFrame, size: int):

    """

    Bootstrapping to obtain estimates of Pearson correlation coefficient in specified dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Input data of type calculated in define_multivariate_dataframes
    size :
        Number of draws in bootstrapping algorithm

    """

    corr_replicates = np.empty(size)

    for i in range(size):

        bs_sample = data.sample(n=data.shape[0], replace=True)

        corr_replicates[i] = bs_sample["pr"].corr(bs_sample["tas"])

    return corr_replicates


def _calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def calculate_bs_replicates_mi(data: pd.DataFrame, bins: int, size: int) -> np.ndarray:

    """

    Bootstrapping to obtain estimates of mutual information in specified dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Input data of type calculated in define_multivariate_dataframes
    bins : int
        Size of each draw in bootstrapping algorithm
    size :
        Number of draws in bootstrapping algorithm

    """

    corr_replicates = np.empty(size)

    for i in range(size):

        bs_sample = data.sample(n=data.shape[0], replace=True).to_numpy()

        corr_replicates[i] = _calc_MI(bs_sample[:, 0], bs_sample[:, 1], bins)

    return corr_replicates
