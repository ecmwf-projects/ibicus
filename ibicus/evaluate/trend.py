# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Trend module - Calculate and plot changes to the climate model trend through application of different bias adjustment methods. Changes in the trend in both statistical properties (mean, quantiles), as well as threshold metrics (:py:class:`ThresholdMetric`) can be calculated. Trends are defined here between a validation and future period.
"""

import warnings
from logging import warning

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from ..utils._utils import _unpack_df_of_numpy_arrays
from ..variables import map_variable_str_to_variable_class, str_to_variable_class


def _calculate_mean_trend_bias(
    trend_type: str,
    raw_validate: np.ndarray,
    raw_future: np.ndarray,
    bc_validate: np.ndarray,
    bc_future: np.ndarray,
) -> np.ndarray:
    if trend_type == "additive":
        bc_trend = np.mean(bc_future, axis=0) - np.mean(bc_validate, axis=0)
        raw_trend = np.mean(raw_future, axis=0) - np.mean(raw_validate, axis=0)
    elif trend_type == "multiplicative":
        bc_trend = np.mean(bc_future, axis=0) / np.mean(bc_validate, axis=0)
        raw_trend = np.mean(raw_future, axis=0) / np.mean(raw_validate, axis=0)
    else:
        raise ValueError(
            f"trend_type needs to be one of ['additive', 'multiplicative']. Was: {trend_type}."
        )

    bias = 100 * (bc_trend - raw_trend) / raw_trend
    return bias


def _calculate_mean_trend(
    trend_type: str,
    bc_validate: np.ndarray,
    bc_future: np.ndarray,
) -> np.ndarray:
    if trend_type == "additive":
        bc_trend = np.mean(bc_future, axis=0) - np.mean(bc_validate, axis=0)
    elif trend_type == "multiplicative":
        bc_trend = np.mean(bc_future, axis=0) / np.mean(bc_validate, axis=0)
    else:
        raise ValueError(
            f"trend_type needs to be one of ['additive', 'multiplicative']. Was: {trend_type}."
        )

    return bc_trend


def _calculate_quantile_trend_bias(
    trend_type: str,
    quantile: float,
    raw_validate: np.ndarray,
    raw_future: np.ndarray,
    bc_validate: np.ndarray,
    bc_future: np.ndarray,
) -> np.ndarray:
    if trend_type == "additive":
        bc_trend = np.quantile(bc_future, quantile, axis=0) - np.quantile(
            bc_validate, quantile, axis=0
        )
        raw_trend = np.quantile(raw_future, quantile, axis=0) - np.quantile(
            raw_validate, quantile, axis=0
        )

    elif trend_type == "multiplicative":
        if (q_bc_validate := np.quantile(bc_validate, quantile, axis=0)) != 0 and (
            q_raw_validate := np.quantile(raw_validate, quantile, axis=0)
        ) != 0:
            bc_trend = np.quantile(bc_future, quantile, axis=0) / q_bc_validate
            raw_trend = np.quantile(raw_future, quantile, axis=0) / q_raw_validate

        else:
            raise ZeroDivisionError(
                f"Selected quantile is zero either for the bias corrected or raw model in the validation period. Cannot analyse multiplicative trend in quantile: {str(quantile)}"
            )
    else:
        raise ValueError(
            f"trend_type needs to be one of ['additive', 'multiplicative']. Was: {trend_type}."
        )

    bias = 100 * (bc_trend - raw_trend) / raw_trend
    return bias


def _calculate_quantile_trend(
    trend_type: str,
    quantile: float,
    bc_validate: np.ndarray,
    bc_future: np.ndarray,
) -> np.ndarray:
    if trend_type == "additive":
        bc_trend = np.quantile(bc_future, quantile, axis=0) - np.quantile(
            bc_validate, quantile, axis=0
        )

    elif trend_type == "multiplicative":
        if q_bc_validate := np.quantile(bc_validate, quantile, axis=0) != 0:
            bc_trend = np.quantile(bc_future, quantile, axis=0) / q_bc_validate
        else:
            raise ZeroDivisionError(
                f"Selected quantile is zero either for the bias corrected or raw model in the validation period. Cannot analyse multiplicative trend in quantile: {str(quantile)}"
            )
    else:
        raise ValueError(
            f"trend_type needs to be one of ['additive', 'multiplicative']. Was: {trend_type}."
        )
    return bc_trend


def _calculate_metrics_trend_bias(
    trend_type: str,
    metric,
    raw_validate: np.ndarray,
    raw_future: np.ndarray,
    bc_validate: np.ndarray,
    bc_future: np.ndarray,
    time_validate: np.ndarray = None,
    time_future: np.ndarray = None,
) -> np.ndarray:
    if trend_type == "additive":
        bc_trend = metric.calculate_exceedance_probability(
            bc_future, time=time_future
        ) - metric.calculate_exceedance_probability(bc_validate, time=time_validate)
        raw_trend = metric.calculate_exceedance_probability(
            raw_future, time=time_future
        ) - metric.calculate_exceedance_probability(raw_validate, time=time_validate)

    elif trend_type == "multiplicative":
        if (
            m_bc_validate := metric.calculate_exceedance_probability(
                bc_validate, time=time_validate
            )
        ) != 0 and (
            m_raw_validate := metric.calculate_exceedance_probability(
                bc_validate, time=time_validate
            )
        ) != 0:
            bc_trend = (
                metric.calculate_exceedance_probability(bc_future, time=time_future)
                / m_bc_validate
            )
            raw_trend = (
                metric.calculate_exceedance_probability(raw_future, time=time_future)
                / m_raw_validate
            )

        else:
            raise ZeroDivisionError(
                "Occurrence probability of selected metric is zero either for the bias corrected or raw model in the validation period."
            )
    else:
        raise ValueError(
            f"trend_type needs to be one of ['additive', 'multiplicative']. Was: {trend_type}."
        )

    trend_bias = 100 * (bc_trend - raw_trend) / raw_trend
    return trend_bias


def _calculate_metrics_trend(
    trend_type: str,
    metric,
    bc_validate: np.ndarray,
    bc_future: np.ndarray,
    time_validate: np.ndarray = None,
    time_future: np.ndarray = None,
) -> np.ndarray:
    if trend_type == "additive":
        bc_trend = metric.calculate_exceedance_probability(
            bc_future, time=time_future
        ) - metric.calculate_exceedance_probability(bc_validate, time=time_validate)

    elif trend_type == "multiplicative":
        if (
            m_bc_validate := metric.calculate_exceedance_probability(
                bc_validate, time=time_validate
            )
        ) != 0:
            bc_trend = (
                metric.calculate_exceedance_probability(bc_future, time=time_future)
                / m_bc_validate
            )

        else:
            raise ZeroDivisionError(
                "Occurrence probability of selected metric is zero either for the bias corrected or raw model in the validation period."
            )
    else:
        raise ValueError(
            f"trend_type needs to be one of ['additive', 'multiplicative']. Was: {trend_type}."
        )
    return bc_trend


def calculate_future_trend_bias(
    raw_validate: np.ndarray,
    raw_future: np.ndarray,
    statistics: list = ["mean", 0.05, 0.95],
    trend_type: str = "additive",
    metrics: list = [],
    time_validate: np.ndarray = None,
    time_future: np.ndarray = None,
    **debiased_cms,
) -> pd.DataFrame:
    """
    For each location, calculates the bias in the trend of the bias corrected model compared to the raw climate model for the following metrics:
    mean, 5% and 95% quantile (default) as well as metrics of class :py:class:`ThresholdMetric` (from ``ibicus.evaluate.metrics``) passed as arguments to the function.

    The trend can be specified as either additive or multiplicative by setting ``trend_type`` (default: additive).

    The function returns numpy array with three columns:
    [Correction method: str, Metric: str, Bias: List containing one 2d np.ndarray containing trend bias at each location]

    Parameters
    ----------
    raw_validate : np.ndarray
        Raw climate data set in validation period.
    raw_future: np.ndarray
        Raw climate data set in future period.
    statistics : list
        List containing float values as well as "mean" specifying for which distributional aspects the trend bias shall be calculated.
    trend_type : str
        Determines the type of trend that is analysed. Has to be one of ['additive', 'multiplicative'].
    metrics : list
        List of :py:class:`ThresholdMetric` metrics whose trend bias shall be calculated. Example: ``metrics = [ibicus.metrics.dry_days', 'ibicus.metrics.wet_days']``.
    time_validate : np.ndarray
        If one of the metrics is time sensitive (defined daily, monthly, seasonally: ``metric.threshold_scope = ['day', 'month', 'year']``) time information needs to be passed to calculate it. This is a 1d numpy arrays of times to which the values in `raw_validate` and the first entry of each `debiased_cms` keyword arguments correspond.
    time_future : np.ndarray
        If one of the metrics is time sensitive (defined daily, monthly, seasonally: ``metric.threshold_scope = ['day', 'month', 'year']``) time information needs to be passed to calculate it. This is a 1d numpy arrays of times to which the values in `raw_future` and the second entry of each `debiased_cms` keyword arguments correspond.
    debiased_cms : np.ndarray
        Keyword arguments given in format ``debiaser_name = [debiased_dataset_validation_period, debiased_dataset_future_period]`` specifying the climate models to be analysed for trends in biases.
        Example: ``QM = [tas_val_debiased_QM, tas_future_debiased_QM]``.

    Examples
    --------
    >>> tas_trend_bias_data = trend.calculate_future_trend_bias(variable = 'tas', raw_validate = tas_cm_validate, raw_future = tas_cm_future, metrics = [ibicus.metrics.warm_days, ibicus.metrics.cold_days], trend_type = "additive", QDM = [tas_val_debiased_QDM, tas_fut_debiased_QDM], CDFT = [tas_val_debiased_CDFT, tas_fut_debiased_CDFT])
    """

    trend_bias_dfs = []

    for debiased_cms_key, debiased_cms_value in debiased_cms.items():
        if len(debiased_cms_value) != 2:
            raise ValueError(
                "Debiased climate datasets in ``*debiased_cms`` should have following form: ``debiaser_name = [debiased_dataset_validation_period, debiased_dataset_future_period]``. Input does not have the required length of two."
            )

        # calculate trend bias in descriptive statistics

        for q in statistics:
            if q == "mean":
                mean_bias = _calculate_mean_trend_bias(
                    trend_type, raw_validate, raw_future, *debiased_cms_value
                )

                if np.any(np.isinf(mean_bias)):
                    warning(
                        "{}: Division by zero encountered in trend bias of mean calculation, not showing results for this debiaser.".format(
                            debiased_cms_key
                        )
                    )
                else:
                    trend_bias_dfs.append(
                        pd.DataFrame(
                            data={
                                "Correction Method": debiased_cms_key,
                                "Metric": "Mean",
                                "Bias": [mean_bias],
                            }
                        )
                    )
            else:
                if q <= 1 and q >= 0:
                    qn_bias = _calculate_quantile_trend_bias(
                        trend_type, q, raw_validate, raw_future, *debiased_cms_value
                    )

                    if np.any(np.isinf(qn_bias)):
                        warning(
                            "{}: Division by zero encountered in trend bias of low quantile calculation, not showing results for this debiaser.".format(
                                debiased_cms_key
                            )
                        )
                    else:
                        trend_bias_dfs.append(
                            pd.DataFrame(
                                data={
                                    "Correction Method": debiased_cms_key,
                                    "Metric": str(q) + " qn",
                                    "Bias": [qn_bias],
                                }
                            )
                        )
                else:
                    warnings.warn(
                        "Quantile values below 0 or above 1 encountered. No quantiles are calculated."
                    )

        # calculate trend bias in metrics

        for m in metrics:
            metric_bias = _calculate_metrics_trend_bias(
                trend_type,
                m,
                raw_validate,
                raw_future,
                *debiased_cms_value,
                time_validate,
                time_future,
            )

            if np.any(np.isinf(metric_bias)):
                warning(
                    "{}: Division by zero encountered in bias of {} calculation, not showing results for this debiaser.".format(
                        debiased_cms_key, m
                    )
                )
            else:
                trend_bias_dfs.append(
                    pd.DataFrame(
                        data={
                            "Correction Method": debiased_cms_key,
                            "Metric": m.name,
                            "Bias": [metric_bias],
                        }
                    )
                )

    plot_data = pd.concat(trend_bias_dfs)

    return plot_data


def calculate_future_trend(
    statistics: list = ["mean", 0.05, 0.95],
    trend_type: str = "additive",
    metrics: list = [],
    time_validate: np.ndarray = None,
    time_future: np.ndarray = None,
    **debiased_cms,
) -> pd.DataFrame:
    """
    For each location, calculates the trend of the bias corrected model compared to the raw climate model for the following metrics:
    mean, 5% and 95% quantile (default) as well as metrics of class :py:class:`ThresholdMetric` (from ``ibicus.evaluate.metrics``) passed as arguments to the function.

    The trend can be specified as either additive or multiplicative by setting ``trend_type`` (default: additive).

    The function returns numpy array with three columns:
    [Correction method: str, Metric: str, Bias: List containing one 2d np.ndarray containing trend bias at each location]

    Parameters
    ----------
    statistics : list
        List containing float values as well as "mean" specifying for which distributional aspects the trend bias shall be calculated.
    trend_type : str
        Determines the type of trend that is analysed. Has to be one of ['additive', 'multiplicative'].
    metrics : list
        List of :py:class:`ThresholdMetric` metrics whose trend bias shall be calculated. Example: ``metrics = [ibicus.metrics.dry_days', 'ibicus.metrics.wet_days']``.
    time_validate : np.ndarray
        If one of the metrics is time sensitive (defined daily, monthly, seasonally: ``metric.threshold_scope = ['day', 'month', 'year']``) time information needs to be passed to calculate it. This is a 1d numpy arrays of times to which the first entry of each `debiased_cms` keyword arguments correspond.
    time_future : np.ndarray
        If one of the metrics is time sensitive (defined daily, monthly, seasonally: ``metric.threshold_scope = ['day', 'month', 'year']``) time information needs to be passed to calculate it. This is a 1d numpy arrays of times to which the second entry of each `debiased_cms` keyword arguments correspond.
    debiased_cms : np.ndarray
        Keyword arguments given in format ``debiaser_name = [debiased_dataset_validation_period, debiased_dataset_future_period]`` specifying the climate models to be analysed for trends in biases.
        Example: ``QM = [tas_val_debiased_QM, tas_future_debiased_QM]``.

    Examples
    --------
    >>> tas_trend_bias_data = trend.calculate_future_trend(variable = 'tas', metrics = [ibicus.metrics.warm_days, ibicus.metrics.cold_days], trend_type = "additive", QDM = [tas_val_debiased_QDM, tas_fut_debiased_QDM], CDFT = [tas_val_debiased_CDFT, tas_fut_debiased_CDFT])
    """

    trend_bias_dfs = []

    for debiased_cms_key, debiased_cms_value in debiased_cms.items():
        if len(debiased_cms_value) != 2:
            raise ValueError(
                "Debiased climate datasets in ``*debiased_cms`` should have following form: ``debiaser_name = [debiased_dataset_validation_period, debiased_dataset_future_period]``. Input does not have the required length of two."
            )

        # calculate trend bias in descriptive statistics

        for q in statistics:
            if q == "mean":
                mean_bias = _calculate_mean_trend(trend_type, *debiased_cms_value)

                if np.any(np.isinf(mean_bias)):
                    warning(
                        "{}: Division by zero encountered in trend bias of mean calculation, not showing results for this debiaser.".format(
                            debiased_cms_key
                        )
                    )
                else:
                    trend_bias_dfs.append(
                        pd.DataFrame(
                            data={
                                "Correction Method": debiased_cms_key,
                                "Metric": "Mean",
                                "Bias": [mean_bias],
                            }
                        )
                    )
            else:
                if q <= 1 and q >= 0:
                    qn_bias = _calculate_quantile_trend(
                        trend_type, q, *debiased_cms_value
                    )

                    if np.any(np.isinf(qn_bias)):
                        warning(
                            "{}: Division by zero encountered in trend bias of low quantile calculation, not showing results for this debiaser.".format(
                                debiased_cms_key
                            )
                        )
                    else:
                        trend_bias_dfs.append(
                            pd.DataFrame(
                                data={
                                    "Correction Method": debiased_cms_key,
                                    "Metric": str(q) + " qn",
                                    "Bias": [qn_bias],
                                }
                            )
                        )
                else:
                    warnings.warn(
                        "Quantile values below 0 or above 1 encountered. No quantiles are calculated."
                    )

        # calculate trend bias in metrics

        for m in metrics:
            metric_bias = _calculate_metrics_trend(
                trend_type,
                m,
                *debiased_cms_value,
                time_validate,
                time_future,
            )

            if np.any(np.isinf(metric_bias)):
                warning(
                    "{}: Division by zero encountered in bias of {} calculation, not showing results for this debiaser.".format(
                        debiased_cms_key, m
                    )
                )
            else:
                trend_bias_dfs.append(
                    pd.DataFrame(
                        data={
                            "Correction Method": debiased_cms_key,
                            "Metric": m.name,
                            "Bias": [metric_bias],
                        }
                    )
                )

    plot_data = pd.concat(trend_bias_dfs)

    return plot_data


def plot_future_trend_bias_boxplot(
    variable: str,
    bias_df: pd.DataFrame,
    manual_title: str = " ",
    remove_outliers: bool = False,
    outlier_threshold: int = 100,
    color_palette="tab10",
):
    """
    Accepts ouput given by :py:func:`calculate_future_trend_bias` and creates an overview boxplot of the bias in the trend of different metrics.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    bias_df : pd.DataFrame
        Numpy array with three columns: [Bias correction method, Metric, Bias value at certain location]. Output from :py:func:`calculate_future_trend_bias`.
    manual_title : str
        Manual title to replace the automatically generated one.
    remove_outliers : bool
        If set to ``True``, values above the threshold specified through the next argument are removed
    outlier_threshold: int,
        Threshold above which to remove values from the plot
    color_palette : str
        Seaborn color palette to use for the boxplot.
    """

    # unpack numpy arrays in column 'Bias'
    bias_df_unpacked = _unpack_df_of_numpy_arrays(df=bias_df, numpy_column_name="Bias")

    if remove_outliers:
        bias_df_unpacked = bias_df_unpacked[
            abs(bias_df_unpacked["Bias"]) < outlier_threshold
        ]

    # create figure and plot
    fig = plt.figure(figsize=(10, 6))
    ax = seaborn.boxplot(
        y="Bias",
        x="Metric",
        data=bias_df_unpacked,
        palette=color_palette,
        hue="Correction Method",
    )
    [ax.axvline(x + 0.5, color="k") for x in ax.get_xticks()]
    [ax.axhline(linestyle="--", color="k")]

    # generate and set plot title
    if manual_title == " ":
        plot_title = "Bias in climate model trend between validation and future period \n {} ({})".format(
            map_variable_str_to_variable_class(variable).name,
            map_variable_str_to_variable_class(variable).unit,
        )
    else:
        plot_title = manual_title
    fig.suptitle(plot_title)

    return fig


def plot_future_trend_bias_spatial(
    variable: str,
    metric: str,
    bias_df: pd.DataFrame,
    manual_title: str = " ",
    remove_outliers: bool = False,
    outlier_threshold: int = 100,
):
    """
    Accepts ouput given by :py:func:`calculate_future_trend_bias` and creates an spatial plot of trend bias for one chosen metric.

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    metric : str
        Metric in `bias_df` to plot.
    bias_df: pd.DataFrame
        Dataframe with three columns: [Bias correction method, Metric, Bias value at certain location].  Output from :py:func:`calculate_future_trend_bias`.
    manual_title : str
        Optional argument present in all plot functions: manual_title will be used as title of the plot.
    """

    # check if value passed to metric is present in bias_df
    if metric not in bias_df["Metric"].unique():
        raise ValueError(
            "Chosen metric not calculated in dataframe given as input. Either change the metric argument, or re-calculate the dataframe and include the metric of your choice."
        )

    # filter bias_df
    bias_df_filtered = bias_df[bias_df["Metric"] == metric]

    # generate plot title
    if variable in str_to_variable_class.keys():
        plot_title = "Bias in climate model trend between validation and future period \n {} ({})".format(
            map_variable_str_to_variable_class(variable).name,
            map_variable_str_to_variable_class(variable).unit,
        )
    else:
        plot_title = manual_title
        raise Warning(
            "Variable not recognized, using manual_title to generate plot_title"
        )

    # find maximum value to set axis bounds
    bias_df_unpacked = _unpack_df_of_numpy_arrays(
        df=bias_df_filtered, numpy_column_name="Percentage bias"
    )
    if remove_outliers:
        bias_df_unpacked = bias_df_unpacked[
            abs(bias_df_unpacked["Percentage bias"]) < outlier_threshold
        ]

    axis_max = bias_df_unpacked["Percentage bias"].max()
    axis_min = -axis_max

    # create figure and plot
    fig_width = 6 * bias_df_filtered.shape[0]
    fig, ax = plt.subplots(1, bias_df_filtered.shape[0], figsize=(fig_width, 5))
    fig.suptitle(plot_title)

    i = 0
    for index, row_array in bias_df_filtered.iterrows():
        plot_title = row_array.values[0]
        plot_data = row_array.values[2]

        plot = ax[i].imshow(
            plot_data, cmap=plt.get_cmap("coolwarm"), vmin=axis_min, vmax=axis_max
        )
        ax[i].set_title(plot_title)
        fig.colorbar(plot, ax=ax[i])
        i = i + 1
