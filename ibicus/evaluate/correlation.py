# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Correlation module - Calculate and plot the RMSE between spatial correlation matrices at each location.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import sklearn.metrics

from ..variables import map_variable_str_to_variable_class, str_to_variable_class


def rmse_spatial_correlation_distribution(
    variable: str, obs_data: np.ndarray, **cm_data
) -> pd.DataFrame:
    """
    Calculates Root-Mean-Squared-Error between observed and modelled spatial correlation matrix at each location.

    The computation involves the following steps: At each location, calculate the correlation to each other location in the observed as well as the climate model
    data set. Then calculate the mean squared error between these two matrices.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    obs_data : np.ndarray
        Optional argument present in all plot functions: manual_title will be used as title of the plot.
    cm_data :
        Keyword arguments specifying climate model datasets, for example: QM = tas_debiased_QM

    Examples
    --------
    >>> tas_rmsd_spatial = rmse_spatial_correlation_distribution(variable = 'tas', obs_data = tas_obs_validate, raw = tas_cm_future, QDM = tas_val_debiased_QDM)

    """

    rmsd_arrays = []

    for k in cm_data.keys():

        for a, b in np.ndindex(obs_data.shape[1:]):

            # initialize two arrays to be filled with correlation values
            corr_matrix_obs = np.zeros((obs_data.shape[1], obs_data.shape[2]))
            corr_matrix_cm = np.zeros((obs_data.shape[1], obs_data.shape[2]))

            for i, j in np.ndindex(obs_data.shape[1:]):

                corr_matrix_obs[i, j] = np.corrcoef(
                    obs_data[:, a, b], obs_data[:, i, j]
                )[0, 1]
                corr_matrix_cm[i, j] = np.corrcoef(
                    cm_data[k][:, a, b], cm_data[k][:, i, j]
                )[0, 1]

            # calculate rmsd between two correlation matrices
            rmsd = math.sqrt(
                sklearn.metrics.mean_squared_error(corr_matrix_obs, corr_matrix_cm)
            )

            rmsd_arrays.append(
                pd.DataFrame(
                    data={
                        "x": [a],
                        "y": [b],
                        "Correction Method": k,
                        "RMSE spatial correlation": rmsd,
                    }
                )
            )

    rmsd_data = pd.concat(rmsd_arrays)
    rmsd_data["RMSE spatial correlation"] = pd.to_numeric(
        rmsd_data["RMSE spatial correlation"]
    )

    return rmsd_data


def rmse_spatial_correlation_boxplot(
    variable: str, dataset: pd.DataFrame, manual_title: str = " "
):

    """
    Boxplot of RMSE of spatial correlation across locations.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    dataset : pd.DataFrame
        Ouput format of function :py:func:`rmse_spatial_correlation_distribution`
    manual_title : str
        Optional argument present in all plot functions: manual_title will be used as title of the plot.

    """

    # create figure and plot
    fig = plt.figure(figsize=(8, 6))
    seaborn.boxplot(
        y="RMSE spatial correlation",
        x="Correction Method",
        data=dataset,
        palette="colorblind",
    )

    # set plot title
    if manual_title == " ":
        if variable in str_to_variable_class.keys():
            plot_title = "{} ({}) \n RMSE of spatial correlation matrices".format(
                map_variable_str_to_variable_class(variable).name,
                map_variable_str_to_variable_class(variable).unit,
            )
        else:
            plot_title = manual_title
            raise Warning(
                "Variable not recognized, using manual_title to generate plot_title"
            )
    else:
        plot_title = manual_title

    fig.suptitle(plot_title)

    return fig
