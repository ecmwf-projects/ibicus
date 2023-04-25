# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Marginal module - Calculate and plot the location-wise bias of the climate model before and after applying different bias adjustment methods. Provides the possiblity to calculate either the absolute or percentage bias at each location, for both statistical properties of the variable and specified threshold metrics, and plot either a boxplot across locations, or a spatial heatmap.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from ..utils._utils import (
    _check_if_list_of_two_and_unpack_else_none,
    _unpack_df_of_numpy_arrays,
    year,
)
from ..variables import map_variable_str_to_variable_class, str_to_variable_class


def _marginal_metrics_absolute_bias(
    metric,
    obs_data: np.ndarray,
    cm_data: np.ndarray,
    time_obs_data: np.ndarray = None,
    time_cm_data: np.ndarray = None,
):
    """
    Calculates location-wise percentage bias of metric specified
    """
    obs_metric = metric.calculate_exceedance_probability(obs_data, time=time_obs_data)
    cm_metric = metric.calculate_exceedance_probability(cm_data, time=time_cm_data)
    bias = 365 * cm_metric - 365 * obs_metric

    return bias


def _marginal_mean_bias(obs_data: np.ndarray, cm_data: np.ndarray, bias_type):
    """
    Calculates location-wise percentage bias of mean
    """
    if bias_type == "percentage":
        mean_bias = (
            100
            * (np.mean(cm_data, axis=0) - np.mean(obs_data, axis=0))
            / np.mean(obs_data, axis=0)
        )
    if bias_type == "absolute":
        mean_bias = np.mean(cm_data, axis=0) - np.mean(obs_data, axis=0)

    return mean_bias


def _marginal_quantile_bias(
    quantile: float, obs_data: np.ndarray, cm_data: np.ndarray, bias_type
):
    """
    Calculates location-wise percentage bias of specified quantile. If any value at chosen quantile is zero, function returns zero bias.
    """

    if quantile < 0 or quantile > 1:
        raise ValueError("quantile needs to be between 0 and 1")

    qn_obs = np.quantile(obs_data, quantile, axis=0)

    if bias_type == "percentage":
        qn_bias = 100 * (np.quantile(cm_data, quantile, axis=0) - qn_obs) / qn_obs
    if bias_type == "absolute":
        qn_bias = np.quantile(cm_data, quantile, axis=0) - qn_obs

    return qn_bias


def _marginal_metrics_bias(
    metric,
    obs_data: np.ndarray,
    cm_data: np.ndarray,
    time_obs_data: np.ndarray = None,
    time_cm_data: np.ndarray = None,
):
    """
    Calculates location-wise percentage bias of metric specified
    """

    obs_metric = metric.calculate_exceedance_probability(obs_data, time=time_obs_data)
    cm_metric = metric.calculate_exceedance_probability(cm_data, time=time_cm_data)
    bias = 100 * (cm_metric - obs_metric) / obs_metric

    return bias


def calculate_marginal_bias(
    obs: np.ndarray,
    statistics: list = ["mean", 0.05, 0.95],
    metrics: list = [],
    percentage_or_absolute: str = "percentage",
    **cm_data
) -> pd.DataFrame:
    """
    Returns a :py:class:`pd.DataFrame` containing location-wise percentage bias of different metrics: mean, 5th and 95th percentile, as well as metrics specific in `metrics`, comparing observations to climate model output during a validation period.

    Output dataframes contains four columns: 'Correction Method' (str) correspond to the cm_data keys, 'Metric', which shows the name of the threshold metric or statistic calculated, 'Type' which specifies whether the absolute or percentage bias is calculated,
    and 'Bias' which contains a np.ndarray which in turn contains the output values at each location.

    The output can be plotted using :py:func:`plot_marginal_bias` or :py:func:`plot_bias_spatial` or manually be analyzed further.

    Parameters
    ----------
    obs : np.ndarray
        observational dataset in validation period.
        If one of the metrics is time sensitive (defined daily, monthly, seasonally) this needs to be a list of form ``[obs_data, time_obs_data]`` where `time_obs_data` is a 1d numpy arrays of times corresponding the the values in `obs_data`.
    statistics : list
        List containing float values as well as "mean" specifying for which distributional aspects the trend bias shall be calculated. Default: ``["mean", 0.05, 0.95]``.
    metrics : list
        List of :py:class:`ThresholdMetric` metrics whose trend bias shall be calculated. Example: ``metrics = [ibicus.metrics.dry_days', 'ibicus.metrics.wet_days']``.
    percentage_or_absolute : str
        Specifies whether for the climate threshold metrics the percentage bias (p(cm)-p(obs))/p(obs) is computed, or the absolute bias. For threshold metrics, the absolute bias is difference in the mean days per year that this metric is exceeded
        and for statistics it is the absolute difference in [physical units] of the two statistics.
    **cm_data :
        Keyword arguments of type ``debiaser_name = debiased_dataset`` in validation period (example: ``QM = tas_val_debiased_QM``), covering all debiasers that are to be compared.
        If one of the metrics is time sensitive (defined daily, monthly, seasonally: ``metric.threshold_scope = ['day', 'month', 'year']``) this needs to be a list of form lists of ``[cm_data, time_cm_data]`` where `time_cm_data` is a 1d numpy arrays of times corresponding the the values in `cm_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame with marginal bias at all locations, for all metrics specified.

    Examples
    --------
    >>> tas_marginal_bias_df = marginal.calculate_marginal_bias(obs = tas_obs_validate, metrics = tas_metrics, raw = tas_cm_validate, ISIMIP = tas_val_debiased_ISIMIP)

    """

    marginal_bias_dfs = []

    obs_data, time_obs_data = _check_if_list_of_two_and_unpack_else_none(obs)

    for cm_data_key, cm_data_value in cm_data.items():
        cm_data_value, time_cm_data_value = _check_if_list_of_two_and_unpack_else_none(
            cm_data_value
        )

        for q in statistics:
            if q == "mean":
                mean_bias = _marginal_mean_bias(
                    obs_data=obs_data,
                    cm_data=cm_data_value,
                    bias_type=percentage_or_absolute,
                )

                if np.any(np.isinf(mean_bias)):
                    warnings.warn(
                        "{}: Division by zero encountered in bias of mean calculation, not showing results for this debiaser.".format(
                            cm_data_key
                        ),
                        stacklevel=2,
                    )
                else:
                    marginal_bias_dfs.append(
                        pd.DataFrame(
                            data={
                                "Correction Method": cm_data_key,
                                "Metric": "Mean",
                                "Type": percentage_or_absolute,
                                "Bias": [mean_bias],
                            }
                        )
                    )
            else:
                if q <= 1 and q >= 0:
                    qn_bias = _marginal_quantile_bias(
                        quantile=q,
                        obs_data=obs_data,
                        cm_data=cm_data_value,
                        bias_type=percentage_or_absolute,
                    )

                    if np.any(np.isinf(qn_bias)):
                        warnings.warn(
                            "{}: Division by zero encountered in bias of low quantile calculation, not showing results for this debiaser.".format(
                                cm_data_key
                            ),
                            stacklevel=2,
                        )
                    else:
                        marginal_bias_dfs.append(
                            pd.DataFrame(
                                data={
                                    "Correction Method": cm_data_key,
                                    "Metric": str(q) + " qn",
                                    "Type": percentage_or_absolute,
                                    "Bias": [qn_bias],
                                }
                            )
                        )
                else:
                    warnings.warn(
                        "Quantile values below 0 or above 1 encountered. No quantiles are calculated."
                    )

        for m in metrics:
            if percentage_or_absolute == "percentage":
                metric_bias = _marginal_metrics_bias(
                    m,
                    obs_data=obs_data,
                    cm_data=cm_data_value,
                    time_obs_data=time_obs_data,
                    time_cm_data=time_cm_data_value,
                )

                if np.any(np.isinf(metric_bias)):
                    warnings.warn(
                        "{}: Division by zero encountered in bias of {} calculation, not showing results for this debiaser.".format(
                            cm_data_key, m
                        ),
                        stacklevel=2,
                    )
                else:
                    marginal_bias_dfs.append(
                        pd.DataFrame(
                            data={
                                "Correction Method": cm_data_key,
                                "Metric": m.name,
                                "Type": percentage_or_absolute,
                                "Bias": [metric_bias],
                            }
                        )
                    )

            elif percentage_or_absolute == "absolute":
                metric_bias = _marginal_metrics_absolute_bias(
                    m,
                    obs_data=obs_data,
                    cm_data=cm_data_value,
                    time_obs_data=time_obs_data,
                    time_cm_data=time_cm_data_value,
                )
                marginal_bias_dfs.append(
                    pd.DataFrame(
                        data={
                            "Correction Method": cm_data_key,
                            "Metric": m.name,
                            "Type": percentage_or_absolute,
                            "Bias": [metric_bias],
                        }
                    )
                )

            else:
                warnings.warn(
                    "percentage_or_absolute values not valid, needs to be either 'percentage' or 'absolute'."
                )

    plot_data = pd.concat(marginal_bias_dfs)

    return plot_data


def plot_marginal_bias(
    variable: str,
    bias_df: pd.DataFrame,
    statistics: list = ["Mean", "0.95 qn", "0.05 qn"],
    manual_title: str = " ",
    remove_outliers: bool = False,
    outlier_threshold_statistics: int = 100,
    outlier_threshold_metrics: int = 100,
    color_palette="tab10",
    metrics_title: str = " ",
    statistics_title: str = " ",
):
    """
    Returns boxplots showing distribution of the percentage bias over locations of different metrics, based on calculation performed in :py:func:`calculate_marginal_bias`.

    Two boxplots are created: one for the descriptive statistics and one for threshold metrics present in the bias_df dataframe.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    bias_df : pd.DataFrame
        :py:class:`pd.DataFrame` containing percentage bias for descriptive statistics and specified metrics. Output of :py:func:`calculate_marginal_bias`.
    statistics : list
        List of strings specifying summary statistics computed on the data. Strings have to be equal to entry in the 'Metric' column of bias_df.
    manual_title : str
        Optional argument present in all plot functions: manual_title will be used as title of the plot.
    remove_outliers : bool
        If set to True, values above the threshold specified through the next argument are removed
    outlier_threshold_statistics : int,
        Threshold above which to remove values from the plot for bias statistics (mean, quantiles)
    outlier_threshold_metrics : int
        Threshold above which to remove values from the plot for bias in metrics (such as dry days, hot days, etc)
    color_palette : str
        Seaborn color palette to use for the boxplot.

    Examples
    --------
    >>> tas_marginal_bias_plot = marginal.plot_marginal_bias(variable = 'tas', bias_df = tas_marginal_bias)

    """

    # unpack dataframe
    bias_df_unpacked = _unpack_df_of_numpy_arrays(df=bias_df, numpy_column_name="Bias")

    # split dataframe for two plots
    plot_data1 = bias_df_unpacked[bias_df_unpacked["Metric"].isin(statistics)]
    plot_data2 = bias_df_unpacked[~bias_df_unpacked["Metric"].isin(statistics)]

    if remove_outliers:
        plot_data1 = plot_data1[abs(plot_data1["Bias"]) < outlier_threshold_statistics]
        plot_data2 = plot_data2[abs(plot_data2["Bias"]) < outlier_threshold_metrics]

    # generate plots
    fig_width = 3 * bias_df_unpacked["Metric"].nunique() + 3
    fig, ax = plt.subplots(1, 2, figsize=(fig_width, 6))

    seaborn.boxplot(
        ax=ax[0],
        y="Bias",
        x="Metric",
        data=plot_data1,
        palette=color_palette,
        hue="Correction Method",
    ).set_title(statistics_title)
    [ax[0].axvline(x + 0.5, color="k") for x in ax[0].get_xticks()]
    [ax[0].axhline(linestyle="--", color="k")]

    seaborn.boxplot(
        ax=ax[1],
        y="Bias",
        x="Metric",
        data=plot_data2,
        palette=color_palette,
        hue="Correction Method",
    ).set_title(metrics_title)
    [ax[1].axvline(x + 0.5, color="k") for x in ax[1].get_xticks()]
    [ax[1].axhline(linestyle="--", color="k")]

    # generate and set plot title
    if manual_title == " ":
        if variable in str_to_variable_class.keys():
            plot_title = "{} ({}) - Bias".format(
                map_variable_str_to_variable_class(variable).name,
                map_variable_str_to_variable_class(variable).unit,
            )
        else:
            plot_title = manual_title
    else:
        plot_title = manual_title

    fig.suptitle(plot_title)

    return fig


def plot_bias_spatial(
    variable: str,
    metric: str,
    bias_df: pd.DataFrame,
    remove_outliers: bool = False,
    outlier_threshold: int = 100,
    manual_title: str = " ",
):
    """
    Returns spatial plots of bias at each location with respect to one specified metric, based on calculation performed in :py:func:`calculate_marginal_bias`.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form following CMIP convention.
    metric: str
        Specifies the metric analysed. Has to exactly match the name of this metric in the bias_df DataFrame.
    bias_df: pd.DataFrame
        :py:class:`pd.DataFrame` containing percentage bias for descriptive statistics and specified metrics. Output of :py:func:`calculate_marginal_bias`.
    remove_outliers: bool
            If set to True, values above the threshold specified through the next argument are removed.
    outlier_threshold: int,
            Threshold above which to remove values from the plot.
    manual_title : str
        Optional argument present in all plot functions: manual_title will be used as title of the plot.

    Examples
    --------
    >>> tas_marginal_bias_plot_mean = marginal.plot_bias_spatial(variable = 'tas', metric = 'Mean', bias_df = tas_marginal_bias)

    """

    # check if value passed to metric is present in bias_df
    if metric not in bias_df["Metric"].unique():
        raise ValueError(
            "Chosen metric not calculated in dataframe given as input. Either change the metric argument, or re-calculate the dataframe and include the metric of your choice."
        )

    # filter bias_df
    bias_df_filtered = bias_df[bias_df["Metric"] == metric]

    # generate plot title
    if manual_title == " ":
        if variable in str_to_variable_class.keys():
            plot_title = "{} ({}) \n {} bias of mean".format(
                map_variable_str_to_variable_class(variable).name,
                map_variable_str_to_variable_class(variable).unit,
                bias_df_filtered["Type"].iloc[0],
            )
        else:
            plot_title = manual_title
            warnings.warn(
                "Variable not recognized, using manual_title to generate plot_title",
                stacklevel=2,
            )
    else:
        plot_title = manual_title

    # find maximum value to set axis bounds
    bias_df_unpacked = _unpack_df_of_numpy_arrays(
        df=bias_df_filtered, numpy_column_name="Bias"
    )
    if remove_outliers:
        bias_df_unpacked = bias_df_unpacked[
            abs(bias_df_unpacked["Bias"]) < outlier_threshold
        ]

    axis_max = bias_df_unpacked["Bias"].max()
    axis_min = -axis_max

    # create figure and plot
    fig_width = 6 * bias_df_filtered.shape[0]
    fig, ax = plt.subplots(1, bias_df_filtered.shape[0], figsize=(fig_width, 5))
    fig.suptitle(plot_title)

    i = 0
    for _, row_array in bias_df_filtered.iterrows():
        plot_title = row_array.values[0]
        plot_data = row_array.values[3]

        plot = ax[i].imshow(
            plot_data, cmap=plt.get_cmap("coolwarm"), vmin=axis_min, vmax=axis_max
        )
        ax[i].set_title(plot_title)
        fig.colorbar(plot, ax=ax[i])
        i = i + 1


def _yearly_exceedances(metric, dataset: np.ndarray, time: np.ndarray):
    threshold_matrix = metric.calculate_instances_of_threshold_exceedance(
        dataset, time=time
    )

    year_array = year(time)
    _, counts = np.unique(year_array, return_counts=True)
    index_array = [counts[0]]
    for i in range(len(counts) - 2):
        index_array = np.append(index_array, sum(counts[0 : i + 2]))

    list_years = np.split(threshold_matrix, index_array, axis=0)

    year_count = list()

    for i in range(len(list_years)):
        year_count.append(np.sum(list_years[i], axis=0))

    yearly_threshold_exceedances = np.stack(year_count, axis=0)

    return yearly_threshold_exceedances


def _mean_yearly_exceedances(metric, dataset: np.ndarray, time: np.ndarray):
    yearly_threshold_exceedances = _yearly_exceedances(metric, dataset, time)

    mean_yearly_threshold_exceedances = np.mean(yearly_threshold_exceedances, axis=0)

    return mean_yearly_threshold_exceedances


def calculate_bias_days_metrics(
    obs_data: np.ndarray, metrics: list = [], **cm_data
) -> pd.DataFrame:
    """
    Returns a :py:class:`pd.DataFrame` containing location-wise mean number of yearly threshold exceedances.

    The output dataframes contains five columns: 'Correction Method' (str) correspond to the cm_data keys,
    'Metric', which is in [metrics_names], 'CM' which contains the mean number of days of threshold exceedance in the climate
    models, 'Obs' which which contains the mean number of days of threshold exceedance in the observations,
    and 'Bias' which contains the difference (CM-Obs) between the mean number of threshold exceedance days in the climate
    model and the observations.

    Parameters
    ----------
    obs_data : np.ndarray
        List of observational dataset in validation period and corresponding time information: ``[obs_data, time_obs_data]``. Here `time_obs_data` is a 1d numpy arrays of times corresponding to the values in `obs_data`.
    metrics : list
        Array of strings containing the names of the metrics that are to be assessed.
    **cm_data :
        Keyword arguments of type ``debiaser_name = [cm_data, time_cm_data]`` covering all debiasers to be compared. Here `time_cm_data` is a 1d numpy arrays of times corresponding the the values in `cm_data` and `cm_data` refers to a debiased dataset in a validation period. Example: ``QM = [tas_val_debiased_QM, time_val]``.

    Returns
    -------
    pd.DataFrame
        DataFrame with marginal bias at all locations, for all metrics specified.

    Examples
    --------
    >>> tas_marginal_bias_df = marginal.calculate_marginal_bias(obs_data = tas_obs_validate, metrics = tas_metrics, raw = tas_cm_validate, ISIMIP = tas_val_debiased_ISIMIP)

    """

    marginal_bias_dfs = []

    if not isinstance(obs_data, (list, tuple)):
        raise ValueError(
            "obs_data needs to be a list of two of the form [obs, time_obs]."
        )
    if not len(obs_data) == 2:
        raise ValueError(
            "obs_data needs to be a list of two of the form [obs, time_obs]."
        )

    for cm_data_key, cm_data_value in cm_data.items():
        if not isinstance(cm_data_value, (list, tuple)):
            raise ValueError(
                "Each cm_data keyword needs to be a list of two of the form [cm_data, time_cm_data]."
            )
        if not len(cm_data_value) == 2:
            raise ValueError(
                "Each cm_data keyword needs to be a list of two of the form [cm_data, time_cm_data]."
            )

        for m in metrics:
            # calculate days per year that this metric is exceeded

            cm_mean = _mean_yearly_exceedances(
                metric=m, dataset=cm_data_value[0], time=cm_data_value[1]
            )

            obs_mean = _mean_yearly_exceedances(
                metric=m, dataset=obs_data[0], time=obs_data[1]
            )

            metric_bias = cm_mean - obs_mean

            marginal_bias_dfs.append(
                pd.DataFrame(
                    data={
                        "Correction Method": cm_data_key,
                        "Metric": m.name,
                        "CM": [cm_mean],
                        "Obs": [obs_mean],
                        "Bias": [metric_bias],
                    }
                )
            )

    return pd.concat(marginal_bias_dfs)


def plot_spatiotemporal(
    data=[],
    column_names=[
        "Spell length (days)",
        "Spatiotemporal cluster size",
        "Spatial extent (% of area)",
    ],
    xlims=[30, 30, 1],
):

    """
    Plots empirical CDFs of spatiotemporal clustersizes over entire area.

    Parameters
    ----------
    data : list = []
        List of dataframes, output of the type produced by metrics.calculate_spell_length, metrics.calculate_spatial_extent, metrics.calculate_spatiotemporal_clusters expected as elements of the list
    column_names : list = ["Spell length (days)", "Spatiotemporal cluster size", "Spatial extent (% of area)",]
        Names of the columns containing spatiotemporal cluster sizes corresponding to the dataframes given to the argument data.
    xlims : list = [30, 30, 1]
        xlim for each of the plots, corresponding to the dataframes given to the argument data.

    Examples
    --------
    >>> spatiotemporal_figure = marginal.plot_spatiotemporal(data = [spelllength_dry, spatiotemporal_dry, spatial_dry])

    """
    figure_number = len(data)

    fig, axes = plt.subplots(
        1, figure_number, figsize=(figure_number * 5, 5), sharey=False
    )

    for i in range(len(data)):
        seaborn.ecdfplot(
            ax=axes[i], data=data[i], hue="Correction Method", x=column_names[i]
        )  # , stat='count')
        axes[i].set_xlim([0, xlims[i]])
        axes[i].set_title(column_names[i])

    plt.tight_layout()
    return fig


def plot_histogram(
    variable: str,
    data_obs: np.ndarray,
    bin_number: int = 100,
    manual_title: str = " ",
    **cm_data
):
    """
    Plots histogram over entire area or at single location. Expects a one-dimensional array as input.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    data_obs : np.ndarray
        1d-array - either observational data specified at one location, or flattened array of all observed values over the area. Numeric values expected.
    bin_number : int
        Number of bins plotted in histogram, set to 100 by default
    manual_title : str
        Optional argument present in all plot functions: manual_title will be used as title of the plot.

    Examples
    --------
    >>> histogram = plot_histogram(variable='tas', data_obs=tas_obs_validate[:, 0,0], raw = tas_cm_validate[:, 0,0],  ISIMIP = tas_val_debiased_ISIMIP[:, 0,0], CDFt = tas_val_debiased_CDFT[:, 0,0])

    """

    # set plot features
    number_biascorrections = len(cm_data.keys())
    figure_length = number_biascorrections * 5
    plot_number = number_biascorrections
    fig, ax = plt.subplots(1, plot_number, figsize=(figure_length, 5), squeeze=True)

    # generate plot title
    if manual_title == " ":
        if variable in str_to_variable_class.keys():
            plot_title = "Distribution {} ({}) over entire area".format(
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

    # generate plots
    i = 0
    for k, cm_data in cm_data.items():
        ax[i].hist(data_obs, bins=bin_number, alpha=0.5, label="Observed")
        ax[i].hist(cm_data, bins=bin_number, alpha=0.5, label="Climate model")
        ax[i].set_title(k)
        ax[i].legend()
        i = i + 1

    return fig
