# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Multivariate module - calculate and plot conditional threshold exceedances, and analyse and plot the correlation between two variables at each location before and after bias adjustment to check for changes in the multivariate structure.
"""

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from scipy.stats import kstest

from ..utils._utils import _unpack_df_of_numpy_arrays
from ..variables import map_variable_str_to_variable_class, str_to_variable_class


def _calculate_chi(metric1, metric2, dataset1, dataset2, time=None):
    metric1_instances = metric1.calculate_instances_of_threshold_exceedance(
        dataset1, time=time
    )
    metric2_instances = metric2.calculate_instances_of_threshold_exceedance(
        dataset2, time=time
    )

    metric1_instances[metric1_instances == 0] = 2

    cooccurrence = (metric1_instances == metric2_instances).astype(int)

    if np.any(np.einsum("ijk -> jk", metric2_instances) == 0):
        raise ValueError(
            "There is at least one location where threshold exceedance for metric2 does not occur. Calculation cannot be performed"
        )

    chi = np.einsum("ijk -> jk", cooccurrence) / np.einsum(
        "ijk -> jk", metric2_instances
    )

    return chi


def calculate_conditional_joint_threshold_exceedance(metric1, metric2, **climate_data):
    """
    Returns a :py:class:`pd.DataFrame` containing location-wise conditional exceedance probability. Calculates:

    .. math:: p (\\text{Metric1} | \\text{Metric2}) = p (\\text{Metric1} , \\text{Metric2}) / p(\\text{Metric2})

    Output is a pd.DataFrame with 3 columns:

    - Correction Method: Type of climate data - obs, raw, bias_correction_name. Given through keys of climate_data.
    - Compound metric: str reading ``Metric1.name given Metric2.name``.
    - Conditional exceedance probability: 2d numpy array with conditional exceedance probability at each location.

    Parameters
    ----------
    metric1 : ThresholdMetric
        Metric 1 whose exceedance conditional on metric 2 shall be assessed.
    metric2 : ThresholdMetric
        Metric 2 on which metric 1 is contioned upon.
    **climate_data :
        Keyword arguments of type ``key = [variable1_debiased_dataset, variable2_debiased_dataset]``. Here the exceedance of metric 1 is calculated on the first dataset (`variable1_debiased_dataset`) and the exceedance of metric 2 on the second one (`variable2_debiased_dataset`) to calculate the conditional exceedance. Example: ``obs = [pr_obs_validate, tasmin_obs_validate]``, or ``ISIMIP = [pr_val_debiased_ISIMIP, tasmin_val_debiased_ISIMIP]``).

        If one the metrics is time sensitive (defined daily, monthly, seasonally: ``metric.threshold_scope = ['day', 'month', 'year']``) a third list element needs to be passed: a 1D :py:class:`np.ndarray` containing the time information corresponding to the entries in the variable1 and variable2 datasets. Example: ``obs = [pr_obs_validate, tasmin_obs_validate, time_obs_validate]``.

        .. warning:: Datasets for variable1 and variable2 need to be during the same time period and the entries need to happen at the same dates.

    Returns
    -------
    pd.DataFrame
        DataFrame with conditional exceedance probability at all locations for the combination of metrics chosen.

    Examples
    --------
    >>> dry_frost_data = calculate_conditional_exceedance(metric1 = dry_days, metric2 = frost_days, obs = [pr_obs_validate, tasmin_obs_validate], raw = [pr_cm_validate, tasmin_cm_validate], ISIMIP = [pr_val_debiased_ISIMIP, tasmin_val_debiased_ISIMIP])
    """

    conditional_exceedance_dfs = []

    compound_metric_name = "{} given {}".format(metric1.name, metric2.name)

    for climate_data_key, climate_data_value in climate_data.items():
        chi = (
            _calculate_chi(
                metric1,
                metric2,
                climate_data_value[0],
                climate_data_value[1],
                time=climate_data_value[2] if len(climate_data_value) > 2 else None,
            )
            * 100
        )

        conditional_exceedance_dfs.append(
            pd.DataFrame(
                data={
                    "Correction Method": climate_data_key,
                    "Compound metric": compound_metric_name,
                    "Conditional exceedance probability": [chi],
                }
            )
        )

    conditional_exceedance = pd.concat(conditional_exceedance_dfs)

    return conditional_exceedance


def plot_conditional_joint_threshold_exceedance(
    conditional_exceedance_df: pd.DataFrame,
):
    """
    Accepts ouput given by :py:func:`calculate_conditional_joint_threshold_exceedance` and creates an overview boxplot of the conditional exceedance probability across locations in the chosen datasets.

    Parameters
    ----------
    bias_array: np.ndarray
        Output of :py:func:`calculate_conditional_joint_threshold_exceedance`. py:class:`pd.DataFrame` containing location-wise conditional exceedance probability.
    """

    # unpack dataframe
    conditional_exceedance_df_unpacked = _unpack_df_of_numpy_arrays(
        df=conditional_exceedance_df,
        numpy_column_name="Conditional exceedance probability",
    )

    # create figure and plot
    fig = plt.figure(figsize=(10, 6))
    ax = seaborn.boxplot(
        y="Conditional exceedance probability",
        x="Correction Method",
        data=conditional_exceedance_df_unpacked,
        palette="colorblind",
    )
    [ax.axvline(x + 0.5, color="k") for x in ax.get_xticks()]

    # generate and set plot title
    plot_title = "Probability of {}".format(
        conditional_exceedance_df_unpacked.iat[0, 1]
    )
    fig.suptitle(plot_title)

    return fig


def plot_conditional_probability_spatial(
    bias_df: pd.DataFrame,
    remove_outliers: bool = False,
    outlier_threshold: int = 100,
    plot_title: str = " ",
):
    """
    Spatial plot of bias at each location with respect to one specified metric.

    Parameters
    ----------
    bias_df: pd.DataFrame
        :py:class:`pd.DataFrame` containing conditional exceedance probability, expects output of :py:func:`calculate_conditional_joint_threshold_exceedance`.
    remove_outliers: bool
            If set to True, values above the threshold specified through the next argument are removed.
    outlier_threshold: int,
            Threshold above which to remove values from the plot.
    plot_title : str
        No default plot title set within the function, plot_title will be used as title of the plot.

    Examples
    --------
    >>> warm_wet_vis = plot_conditional_probability_spatial(bias_df=warm_wet, plot_title ="Conditional probability of warm days (>20°C) given wet days (>1mm)")

    """

    # find maximum value to set axis bounds
    bias_df_unpacked = _unpack_df_of_numpy_arrays(
        df=bias_df, numpy_column_name="Conditional exceedance probability"
    )
    if remove_outliers:
        bias_df_unpacked = bias_df_unpacked[
            abs(bias_df_unpacked["Conditional exceedance probability"])
            < outlier_threshold
        ]

    axis_max = bias_df_unpacked["Conditional exceedance probability"].max()
    axis_min = 0

    # create figure and plot
    fig_width = 6 * bias_df.shape[0]
    fig, ax = plt.subplots(1, bias_df.shape[0], figsize=(fig_width, 5))
    fig.suptitle(plot_title)

    i = 0
    for _, row_array in bias_df.iterrows():
        plot_title = row_array.values[0]
        plot_data = row_array.values[2]

        plot = ax[i].imshow(
            plot_data, cmap=plt.get_cmap("Reds"), vmin=axis_min, vmax=axis_max
        )
        ax[i].set_title(plot_title)
        fig.colorbar(plot, ax=ax[i])
        i = i + 1

    return fig


def calculate_and_spatialplot_multivariate_correlation(
    variables: list, manual_title: str = " ", **kwargs
):
    """
    Calculates correlation between the two variables specified in keyword arguments (such as tas and pr) at each location and outputs spatial plot.

    Parameters
    ----------
    variable : list
        Variable name, has to be given in standard form specified in documentation.
    manual_title : str
        Optional argument present in all plot functions: manual_title will be used as title of the plot.
    kwargs :
        Keyword arguments specifying a list of two np.ndarrays containing the two variables of interest.

    Examples
    --------
    >>> correlation.calculate_multivariate_correlation_locationwise(variables = ['tas', 'pr'], obs = [tas_obs_validate, pr_obs_validate], raw = [tas_cm_validate, pr_cm_validate], ISIMIP = [tas_val_debiased_ISIMIP, pr_val_debiased_ISIMIP])

    """

    correlation_matrix = {}

    for k in kwargs.keys():
        variable1 = kwargs[k][0]
        variable2 = kwargs[k][1]

        correlation_matrix[k] = np.zeros((variable1.shape[1], variable1.shape[2]))

        for i, j in np.ndindex(variable1.shape[1:]):
            correlation_matrix[k][i, j] = np.corrcoef(
                variable1[:, i, j].T, variable2[:, i, j].T
            )[0, 1]

    # set axis bounds to maximum value attained
    axis_max = max(
        abs(
            max(
                np.ndarray.flatten(np.vstack(list(chain(*correlation_matrix.values()))))
            )
        ),
        abs(
            min(
                np.ndarray.flatten(np.vstack(list(chain(*correlation_matrix.values()))))
            )
        ),
    )
    axis_min = -axis_max

    # create figure and plot
    fig_width = 6 * len(kwargs.keys())
    fig, ax = plt.subplots(1, len(kwargs.keys()), figsize=(fig_width, 5))

    i = 0
    for k in kwargs.keys():
        plot = ax[i].imshow(
            correlation_matrix[k],
            cmap=plt.get_cmap("coolwarm"),
            vmin=axis_min,
            vmax=axis_max,
        )
        ax[i].set_title("{}".format(k))
        fig.colorbar(plot, ax=ax[i])
        i = i + 1

    # set plot title
    if (variables[0] in str_to_variable_class.keys()) and (
        variables[1] in str_to_variable_class.keys()
    ):
        plot_title = "Multivariate Correlation: {} and {}".format(
            map_variable_str_to_variable_class(variables[0]).name,
            map_variable_str_to_variable_class(variables[1]).name,
        )
    else:
        plot_title = manual_title
        raise Warning(
            "Variable not recognized, using manual_title to generate plot_title"
        )
    fig.suptitle(plot_title)

    return fig


def create_multivariate_dataframes(
    variables: list, datasets_obs: list, datasets_bc: list, gridpoint=(0, 0)
) -> pd.DataFrame:
    """
    Helper function creating a joint :py:class:`pd.Dataframe` of two variables specified, for observational dataset as well as one bias corrected dataset at one datapoint.

    Parameters
    ----------
    variables : list
        List of two variable names, has to be given in standard form following CMIP convention
    datasets_obs : list
        List of two observational datasets during same period for the two variables.
    datasets_bc : list
        List of two bias corrected datasets during same period for the two variables.
    gridpoint : tupel
        Tuple that specifies location from which data will be extracted

    Examples
    --------

    >>> tas_pr_obs, tas_pr_isimip = create_multivariate_dataframes(variables = ['tas', 'pr'], datasets_obs = [tas_obs_validate, pr_obs_validate], datasets_bc = [tas_val_debiased_ISIMIP, pr_val_debiased_ISIMIP], gridpoint = (1,1))

    """

    obs_df = pd.DataFrame(columns=[variables[0], variables[1]])
    obs_df[variables[0]] = datasets_obs[0][:, gridpoint[0], gridpoint[1]]
    obs_df[variables[1]] = datasets_obs[1][:, gridpoint[0], gridpoint[1]]

    bc_df = pd.DataFrame(columns=[variables[0], variables[1]])
    bc_df[variables[0]] = datasets_bc[0][:, gridpoint[0], gridpoint[1]]
    bc_df[variables[1]] = datasets_bc[1][:, gridpoint[0], gridpoint[1]]

    return (obs_df, bc_df)


def plot_correlation_single_location(
    variables: list, obs_df: pd.DataFrame, bc_df: pd.DataFrame
):
    """
    Uses seaborn.regplot and output of :py:func:`create_multivariate_dataframes` to plot scatterplot and Pearson correlation estimate of the two specified variables. Offers visual comparison of correlation at single location.

    Parameters
    ----------
    variable : list
        List of variable name, has to be given in standard form following CMIP convetion.
    obs_df : pd.DataFrame
        First argument of output of :py:func:`create_multivariate_dataframes`
    bc_df : pd.DataFrame
        Second argument of output of :py:func:`create_multivariate_dataframes`

    Examples
    --------
    >>> plot_correlation_single_location(variables = ['tas', 'pr'], obs_df = tas_pr_obs, bc_df = tas_pr_isimip)

    """

    seaborn.set_style("white")
    seaborn.regplot(
        x=variables[0],
        y=variables[1],
        data=obs_df,
        ci=95,
        scatter_kws={"s": 0.8, "color": "g"},
    )
    seaborn.regplot(
        x=variables[0],
        y=variables[1],
        data=bc_df,
        ci=95,
        scatter_kws={"s": 0.8, "color": "r"},
    )


def _calculate_bootstrap_correlation_replicates(data: pd.DataFrame, size: int):
    correlation_replicates = np.empty(size)

    for i in range(size):
        bs_sample = data.sample(n=data.shape[0], replace=True)

        correlation_replicates[i] = bs_sample["pr"].corr(bs_sample["tas"])

    return correlation_replicates


def plot_bootstrap_correlation_replicates(
    obs_df: pd.DataFrame, bc_df: pd.DataFrame, bc_name: str, size: int
):
    """
    Plots histograms of correlation between variables in input dataframes estimated via bootstrap using :py:func:`_calculate_bootstrap_correlation_replicates`.

    Parameters
    ----------
    obs_df : pd.DataFrame
        First argument of output of :py:func:`create_multivariate_dataframes`
    bc_df : pd.DataFrame
        Second argument of output of :py:func:`create_multivariate_dataframes`
    bc_name: str
        Name of bias correction method
    size: int
        Number of draws in bootstrapping procedure


    Examples
    --------
    >>> plot_bootstrap_correlation_replicates(obs_df = tas_pr_obs, bc_df = tas_pr_isimip, bc_name = 'ISIMIP', size=500)

    """

    corr_obs = _calculate_bootstrap_correlation_replicates(obs_df, size)
    corr_bc = _calculate_bootstrap_correlation_replicates(bc_df, size)

    corr = np.stack((corr_obs, corr_bc), axis=1)

    fig = plt.figure(figsize=(8, 6))
    colors = ["forestgreen", "peru"]
    plt.hist(corr, 50, alpha=0.7, color=colors, label=["Observations", bc_name])
    plt.legend()

    print(kstest(corr_obs, corr_bc, "auto"))
    return fig
