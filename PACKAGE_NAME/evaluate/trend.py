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
import seaborn

from PACKAGE_NAME.variables import *


def _calculate_mean_trend_bias(
    variable: str,
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
        raise ValueError(f"trend_type needs to be one of ['additive', 'multiplicative']. Was: {trend_type}.")

    bias = 100 * (bc_trend - raw_trend) / raw_trend
    return bias


def _calculate_quantile_trend_bias(
    variable: str,
    trend_type: str,
    quantile: float,
    raw_validate: np.ndarray,
    raw_future: np.ndarray,
    bc_validate: np.ndarray,
    bc_future: np.ndarray,
) -> np.ndarray:

    if trend_type == "additive":
        bc_trend = np.quantile(bc_future, quantile, axis=0) - np.quantile(bc_validate, quantile, axis=0)
        raw_trend = np.quantile(raw_future, quantile, axis=0) - np.quantile(raw_validate, quantile, axis=0)

    elif trend_type == "multiplicative":
        if (q_bc_validate := np.quantile(bc_validate, quantile, axis=0)) != 0 and (
            q_raw_validate := np.quantile(raw_validate, quantile, axis=0) != 0
        ):
            bc_trend = np.quantile(bc_future, quantile, axis=0) / q_bc_validate
            raw_trend = np.quantile(raw_future, quantile, axis=0) / q_raw_validate

        else:
            raise ZeroDivisionError(
                f"Selected quantile is zero either for the bias corrected or raw model in the validation period. Cannot analyse multiplicative trend in quantile: {str(quantile)}"
            )
    else:
        raise ValueError(f"trend_type needs to be one of ['additive', 'multiplicative']. Was: {trend_type}.")

    bias = 100 * (bc_trend - raw_trend) / raw_trend
    return bias


def _calculate_metrics_trend_bias(
    variable: str,
    metric,
    raw_validate: np.ndarray,
    raw_future: np.ndarray,
    bc_validate: np.ndarray,
    bc_future: np.ndarray,
) -> np.ndarray:

    trend_raw = metric.calculate_exceedance_probability(raw_future) - metric.calculate_exceedance_probability(
        raw_validate
    )
    trend_bc = metric.calculate_exceedance_probability(bc_future) - metric.calculate_exceedance_probability(bc_validate)
    trend_bias = 100 * (trend_bc - trend_raw) / trend_raw
    return trend_bias


def calculate_future_trend_bias(
    variable: str, raw_validate: np.ndarray, raw_future: np.ndarray, metrics: list = [], **debiased_cms
) -> np.ndarray:
    """
    For each location, calculates the bias in the trend of the bias corrected model compared to the raw climate model for the following metrics: mean, 5% and 95% quantile (default) as well as metrics passed as arguments to the function.

    Trend can be specified as either additive or multiplicative. Function returns numpy array with three columns:
    [Bias correction method, Metric, Bias value at certain location]

    Parameters
    ----------
    variable : str
        Variable name, has to be given in standard form specified in documentation.
    metrics : np.ndarray
        1d numpy array of strings containing the keys of the metrics to be analysed. Example: `metrics = ['dry', 'wet']`
    raw_validate : np.ndarray
        Raw climate data set in validation period
    raw_future: np.ndarray
        Raw climate data set in future period
    debiased_cms : np.ndarray
        Keyword arguments given in format `debiaser_name = [debiased_dataset_validation_period, debiased_dataset_future_period]`
        Example: `QM = [tas_val_debiased_QM, tas_future_debiased_QM]`.

    """

    trend_bias_dfs = []

    number_locations = np.prod(raw_validate.shape[1:])

    for debiased_cms_key, debiased_cms_value in debiased_cms.items():

        if len(debiased_cms_value) != 2:
            raise ValueError(
                "Debiased climate datasets in ``*debiased_cms`` should have following form: ``debiaser_name = [debiased_dataset_validation_period, debiased_dataset_future_period]``. It does not have the corred length of two."
            )

        mean_bias = _calculate_mean_trend_bias(variable, "additive", raw_validate, raw_future, *debiased_cms_value)
        trend_bias_dfs.append(
            pd.DataFrame(
                data={
                    "Correction Method": [debiased_cms_key] * number_locations,
                    "Metric": ["Mean"] * number_locations,
                    "Relative change bias (%)": mean_bias.flatten(),
                }
            )
        )

        lowqn_bias = _calculate_quantile_trend_bias(
            variable, "additive", 0.05, raw_validate, raw_future, *debiased_cms_value
        )
        trend_bias_dfs.append(
            pd.DataFrame(
                data={
                    "Correction Method": [debiased_cms_key] * number_locations,
                    "Metric": ["5% qn"] * number_locations,
                    "Relative change bias (%)": lowqn_bias.flatten(),
                }
            )
        )

        highqn_bias = _calculate_quantile_trend_bias(
            variable, "additive", 0.95, raw_validate, raw_future, *debiased_cms_value
        )
        trend_bias_dfs.append(
            pd.DataFrame(
                data={
                    "Correction Method": [debiased_cms_key] * number_locations,
                    "Metric": ["95% qn"] * number_locations,
                    "Relative change bias (%)": highqn_bias.flatten(),
                }
            )
        )

        for m in metrics:

            metric_bias = _calculate_metrics_trend_bias(variable, m, raw_validate, raw_future, *debiased_cms_value)

            trend_bias_dfs.append(
                pd.DataFrame(
                    data={
                        "Correction Method": [debiased_cms_key] * number_locations,
                        "Metric": [m.name] * number_locations,
                        "Relative change bias (%)": metric_bias.flatten(),
                    }
                )
            )

    plot_data = pd.concat(trend_bias_dfs)
    plot_data["Relative change bias (%)"] = pd.to_numeric(plot_data["Relative change bias (%)"])

    return plot_data


def plot_future_trend_bias(variable: str, bias_array: np.ndarray):
    """
    Accepts ouput given by :py:func:`calculate_future_trend_bias` and creates an overview boxplot of the bias in the trend of metrics.

    Parameters
    ----------
    variable: str
        Variable name, has to be given in standard form specified in documentation.
    bias_array: np.ndarray
        Numpy array with three columns: [Bias correction method, Metric, Bias value at certain location]
    """

    fig = plt.figure(figsize=(10, 6))
    ax = seaborn.boxplot(
        y="Relative change bias (%)", x="Metric", data=bias_array, palette="colorblind", hue="Correction Method"
    )
    [ax.axvline(x + 0.5, color="k") for x in ax.get_xticks()]
    fig.suptitle(
        "Bias in climate model trend between historical and future period \n {} ({})".format(
            map_variable_str_to_variable_class(variable).name, map_variable_str_to_variable_class(variable).unit
        )
    )
