# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Metrics module - Standard metric definitions"""

from typing import Optional, Union

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from scipy.ndimage import measurements

from ibicus import utils


@attrs.define(eq=False)
class ThresholdMetric:

    """
    Generic climate metric defined by exceedance or underceedance of threshold; or values between an upper and lower threshold.

    Organises the definition and functionalities of such metrics. Enables to implement a subsection of the `Climdex climate extreme indices<https://www.climdex.org/learn/indices/>`.

    Attributes
    ----------
    threshold_value : Union[float, list[float], tuple[float]]
        Threshold value(s) for the variable (in the correct unit).
        If `threshold_type = "higher"` or `threshold_type = "lower"` this is just a single `float` value and the metric is defined as exceedance or underceedance of that value.
        If `threshold_type = "between"` or `threshold_type = "outside"` then this needs to be a list in the form: `[lower_bound, upper_bound]` and the metric is defined as falling in between, or falling outside these values.
    threshold_type : str
        One of `["higher", "lower", "between", "outside"]`. Indicates whether we are either interested in values above the threshold value (`"higher"`, strict `>`), values below the threshold value (`"lower"`, strict `<`), values between the threshold values (`"between"`, not strict including the bounds) or outside the threshold values (`"outside"`, strict not including the bounds).
    name : str = "unknown"
        Metric name. Will be used in dataframes, plots etc. Recommended to include threshold value and units. Example : 'Frost days \n  (tasmin < 0°C)'. Default: `"unknown"`.
    variable : str = "unknown"
        Unique variable that this threshold metric refers to. Example for frost days: tasmin. Default: `"unknown"`.

    Examples
    --------

    >>> warm_days = ThresholdMetric(name = 'Mean warm days (K)', variable = 'tas', threshold_value = [295], threshold_type = 'higher')

    """

    threshold_value: Union[float, list] = attrs.field()
    threshold_type: str = attrs.field(validator=attrs.validators.in_(["higher", "lower", "between", "outside"]))
    name: str = attrs.field(default="unknown", validator=attrs.validators.instance_of(str))
    variable: str = attrs.field(default="unknown", validator=attrs.validators.instance_of(str))

    def __attrs_post_init__(self):
        if self.threshold_type in ["between", "outside"]:
            if not isinstance(self.threshold_value, (list, tuple)):
                raise ValueError(
                    "threshold_value should be a list with a lower and upper bound for threshold_type = 'between'."
                )
            if len(self.threshold_value) != 2:
                raise ValueError(
                    "threshold_value should have only a lower and upper bound for threshold_type = 'between'."
                )
            if not all(isinstance(elem, (int, float)) for elem in self.threshold_value):
                raise ValueError("threshold_value needs to be a list of floats")
            if not self.threshold_value[0] < self.threshold_value[1]:
                raise ValueError(
                    "lower bounds needs to be smaller than upper bound in threshold_value for threshold_type = 'between'."
                )
        elif self.threshold_type in ["higher", "lower"]:
            if not isinstance(self.threshold_value, (int, float)):
                raise ValueError(
                    "threshold_value needs to be either int or float for threshold_type = 'higher' or threshold_type = 'lower'."
                )
        else:
            raise ValueError(
                "Invalid threshold_type. Needs to be one of ['higher', 'lower', 'between']. Modify the class attribute."
            )

    def _get_mask_threshold_condition(self, x):
        if self.threshold_type == "higher":
            return x > self.threshold_value
        elif self.threshold_type == "lower":
            return x < self.threshold_value
        elif self.threshold_type == "between":
            return np.logical_and(x > self.threshold_value[0], x < self.threshold_value[1])
        elif self.threshold_type == "outside":
            return np.logical_and(x < self.threshold_value[0], x > self.threshold_value[1])
        else:
            raise ValueError(
                "Invalid self.threshold_type. Needs to be one of ['higher', 'lower', 'between']. Modify the class attribute."
            )

    def calculate_instances_of_threshold_exceedance(self, dataset: np.ndarray) -> np.ndarray:
        """
        Returns an array of the same size as `dataset` containing 1 when the threshold condition is met and 0 when not.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected.
        """
        return self._get_mask_threshold_condition(dataset).astype(int)

    def filter_threshold_exceedances(self, dataset: np.ndarray) -> np.ndarray:
        """
        Returns an array containing the values of dataset where the threshold condition is met and zero where not.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected.
        """

        mask_threshold_condition = self._get_mask_threshold_condition(dataset)
        dataset[mask_threshold_condition] = 0
        return dataset

    def calculate_exceedance_probability(self, dataset: np.ndarray) -> np.ndarray:
        """
        Returns the probability of metrics occurrence (threshold exceedance/underceedance or inside/outside range), at each location (across the entire time period).

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected

        Returns
        -------
        np.ndarray
            Probability of metric occurrence at each location.
        """

        threshold_data = self.calculate_instances_of_threshold_exceedance(dataset)
        threshold_probability = np.einsum("ijk -> jk", threshold_data) / threshold_data.shape[0]
        return threshold_probability

    def calculate_number_annual_days_beyond_threshold(
        self, dataset: np.ndarray, dates_array: np.ndarray, time_func=utils.year
    ) -> np.ndarray:

        """
        Calculates number of days beyond threshold for each year in the dataset.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected.
        dates_array : np.ndarray
            Array of dates matching time dimension of dataset. Has to be of form time_dictionary[time_specification] - for example: tas_dates_validate['time_obs']
        time_func : functions
            Points to utils function to either extract days or months.

        Returns
        -------
        np.ndarray
            3d array - [years, lat, long]
        """

        eot_matrix = self.calculate_instances_of_threshold_exceedance(dataset)

        time_array = time_func(dates_array)

        years = np.unique(time_array)

        threshold_exceedances = np.zeros((years.shape[0], dataset.shape[1], dataset.shape[2]))

        for j in range(eot_matrix.shape[1]):
            for k in range(eot_matrix.shape[2]):

                threshold_exceedances[:, j, k] = [(eot_matrix[time_array == i, j, k].sum()) for i in years]

        return threshold_exceedances

    @staticmethod
    def _calculate_spell_lengths_one_location(mask_threshold_condition_one_location):
        return np.diff(
            np.where(
                np.concatenate(
                    (
                        [mask_threshold_condition_one_location[0]],
                        mask_threshold_condition_one_location[:-1] != mask_threshold_condition_one_location[1:],
                        [True],
                    )
                )
            )[0]
        )[::2]

    def calculate_spell_length(self, minimum_length: int, **climate_data) -> pd.DataFrame:
        """
        Returns a `py:class:`pd.DataFrame` of individual spell lengths of metrics occurrences (threshold exceedance/underceedance or inside/outside range), counted across locations, for each climate dataset specified in `**climate_data`.

        A spell length is defined as the number of days that a threshold is continuesly exceeded, underceeded or where values are continuously between or outside the threshold (depending on `self.threshold_type`).
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser as specified in `**climate_data`, 'Metric' - name of the threshold metric, 'Spell length - individual spell length counts'.

        Parameters
        ----------
        minimum length : int
            Minimum spell length (in days) investigated.
        climate_data :
            Keyword arguments, providing the input data to investigate.

        Returns
        -------
        pd.DataFrame
            Dataframe of spell lengths of metrics occurrences.

        Examples
        --------
        >>> dry_days.calculate_spell_length(minimum_length = 4, obs = tas_obs_validate, raw = tas_cm_validate, ISIMIP = tas_val_debiased_ISIMIP)
        """

        spell_length_dfs = []
        for climate_data_key, climate_data_value in climate_data.items():
            mask_threshold_condition = self._get_mask_threshold_condition(climate_data_value)

            spell_length = []
            for i, j in np.ndindex(climate_data_value.shape[1:]):
                spell_length.append(
                    ThresholdMetric._calculate_spell_lengths_one_location(mask_threshold_condition[:, i, j])
                )
            spell_length = np.concatenate(spell_length)
            spell_length = spell_length[spell_length > minimum_length]

            spell_length_dfs.append(
                pd.DataFrame(
                    data={
                        "Correction Method": [climate_data_key] * spell_length.size,
                        "Metric": [self.name] * spell_length.size,
                        "Spell length (days)": spell_length,
                    }
                )
            )
        plot_data = pd.concat(spell_length_dfs)
        plot_data["Spell length (days)"] = pd.to_numeric(plot_data["Spell length (days)"])

        return plot_data

    def calculate_spatial_extent(self, **climate_data):
        """
        Returns a `py:class:`pd.DataFrame` of spatial extends of metrics occurrences (threshold exceedance/underceedance or inside/outside range), for each climate dataset specified in `**climate_data`.

        The spatial extent is defined as the percentage of the area where the threshold is exceeded/underceeded or values are between or outside the bounds (depending on `self.threshold_type`), given that it is exceeded at one location.
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser, 'Metric' - name of the threshold metric, 'Spatial extent (% of area)'

        Parameters
        ----------
        **climate_data :
            Keyword arguments, providing the input data to investigate.

        Returns
        -------
        pd.DataFrame
            Dataframe of spatial extends of metrics occurrences.

        Examples
        --------
        >>> dry_days.calculate_spatial_extent(obs = tas_obs_validate, raw = tas_cm_validate, ISIMIP = tas_val_debiased_ISIMIP)
        """

        spatial_clusters_dfs = []
        for climate_data_key, climate_data_value in climate_data.items():

            threshold_data = self.calculate_instances_of_threshold_exceedance(dataset=climate_data_value)
            spatial_clusters = np.einsum("ijk -> i", threshold_data) / np.prod(threshold_data.shape[1:])
            spatial_clusters = spatial_clusters[spatial_clusters != 0]

            spatial_clusters_dfs.append(
                pd.DataFrame(
                    data={
                        "Correction Method": [climate_data_key] * spatial_clusters.size,
                        "Metric": [self.name] * spatial_clusters.size,
                        "Spatial extent (% of area)": spatial_clusters,
                    }
                )
            )

        plot_data = pd.concat(spatial_clusters_dfs)
        plot_data["Spatial extent (% of area)"] = pd.to_numeric(plot_data["Spatial extent (% of area)"])

        return plot_data

    def calculate_spatiotemporal_clusters(self, **climate_data):
        """
        Returns a `py:class:`pd.DataFrame` of sizes of individual spatiotemporal clusters of metrics occurrences (threshold exceedance/underceedance or inside/outside range), for each climate dataset specified in `**climate_data`.

        A spatiotemporal cluster is defined as a connected set (in time and/or space) where the threshold is exceeded/underceeded or values are between or outside the bounds (depending on `self.threshold_type`).
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser, 'Metric' - name of the threshold metric, 'Spatiotemporal cluster size'

        Parameters
        ----------
        climate_data :
            Keyword arguments, providing the input data to investigate.

        Returns
        -------
        pd.DataFrame
            Dataframe of sizes of individual spatiotemporal clusters of metrics occurrences.

        Examples
        --------
        >>> dry_days.calculate_spatiotemporal_clusters(obs = tas_obs_validate, raw = tas_cm_validate, ISIMIP = tas_val_debiased_ISIMIP)
        """

        spatiotemporal_clusters_dfs = []
        for climate_data_key, climate_data_value in climate_data.items():

            threshold_data = self.calculate_instances_of_threshold_exceedance(dataset=climate_data_value)
            threshold_data_lw, _ = measurements.label(threshold_data)
            area = measurements.sum(threshold_data, threshold_data_lw, index=np.arange(threshold_data_lw.max() + 1))

            spatiotemporal_clusters_dfs.append(
                pd.DataFrame(
                    data={
                        "Correction Method": [climate_data_key] * area.size,
                        "Metric": [self.name] * area.size,
                        "Spatiotemporal cluster size": area,
                    }
                )
            )

        plot_data = pd.concat(spatiotemporal_clusters_dfs)
        plot_data["Spatiotemporal cluster size"] = pd.to_numeric(plot_data["Spatiotemporal cluster size"])

        return plot_data

    def violinplots_clusters(self, minimum_length, **climate_data):
        """
        Returns three violinplots with distributions of temporal, spatial and spatiotemporal extends of metric occurrences, comparing all climate dataset specified in `**climate_data`.

        Parameters
        ----------
        minimum length : int
            Minimum spell length (in days) investigated for temporal extends.
        """

        temporal_data = self.calculate_spell_length(minimum_length, **climate_data)
        spatial_data = self.calculate_spatial_extent(**climate_data)
        spatiotemporal_data = self.calculate_spatiotemporal_clusters(**climate_data)

        fig, ax = plt.subplots(1, 3, figsize=(16, 6))

        seaborn.violinplot(
            ax=ax[0],
            data=temporal_data,
            x="Metric",
            y="Spell length (days)",
            palette="colorblind",
            hue="Correction Method",
        )
        ax[0].set_title("Spell length (days)")

        seaborn.violinplot(
            ax=ax[1],
            data=spatial_data,
            x="Metric",
            y="Spatial extent (% of area)",
            palette="colorblind",
            hue="Correction Method",
        )
        ax[1].set_title("Spatial extent (% of area)")

        seaborn.violinplot(
            ax=ax[2],
            data=spatiotemporal_data,
            x="Metric",
            y="Spatiotemporal cluster size",
            palette="colorblind",
            hue="Correction Method",
        )
        ax[2].set_title("Spatiotemporal cluster size")


@attrs.define
class AccumulativeThresholdMetric(ThresholdMetric):

    """
    Class for climate metrics that are defined by thresholds (child class of :py:class:`ThresholdMetric`), but are accumulative. This mainly concerns precipitation metrics.

    An example of such a metric is total precipitation by very wet days (days > 10mm precipitation).
    """

    def calculate_percent_of_total_amount_beyond_threshold(self, dataset: np.ndarray) -> np.ndarray:

        """
        Calculates percentage of total amount beyond threshold for each location over all timesteps.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.

        Returns
        -------
        np.ndarray
            2d array with percentage of total amount above threshold at each location.
        """

        eot_matrix = self.filter_threshold_exceedances(dataset)

        exceedance_percentage = 100 * np.einsum("ijk -> jk", eot_matrix) / np.einsum("ijk -> jk", dataset)

        return exceedance_percentage

    def calculate_annual_value_beyond_threshold(
        self, dataset: np.ndarray, dates_array: np.ndarray, time_func=utils.year
    ) -> np.ndarray:

        """
        Calculates amount beyond threshold for each year in the dataset.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected.
        dates_array : np.ndarray
            Array of dates matching time dimension of dataset. Has to be of form time_dictionary[time_specification] - for example: tas_dates_validate['time_obs']
        time_func : functions
            Points to utils function to either extract days or months.

        Returns
        -------
        np.ndarray
            3d array - [years, lat, long]
        """

        eot_matrix = self.filter_threshold_exceedances(dataset)

        time_array = time_func(dates_array)

        years = np.unique(time_array)

        threshold_exceedances = np.zeros((years.shape[0], dataset.shape[1], dataset.shape[2]))

        for j in range(eot_matrix.shape[1]):
            for k in range(eot_matrix.shape[2]):

                threshold_exceedances[:, j, k] = [(eot_matrix[time_array == i, j, k].sum()) for i in years]

        return threshold_exceedances

    def calculate_intensity_index(self, dataset):
        """
        Calculates the amount beyond a threshold divided by the number of instance the threshold is exceeded.

        Designed to calculate the simple precipitation intensity index but can be used for other variables.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
        """

        eot_value_matrix = self.filter_threshold_exceedances(dataset)
        eot_threshold_matrix = self.calculate_instances_of_threshold_exceedance(dataset)
        intensity_index = np.einsum("ijk -> jk", eot_value_matrix) / np.einsum("ijk -> jk", eot_threshold_matrix)
        return intensity_index


# ----- pr metrics ----- #
dry_days = AccumulativeThresholdMetric(
    name="Dry days \n (< 1 mm/day)", variable="pr", threshold_value=1 / 86400, threshold_type="lower"
)
"""
Dry days (< 1 mm/day) for `pr`.
"""

wet_days = AccumulativeThresholdMetric(
    name="Wet days \n (> 1 mm/day)", variable="pr", threshold_value=1 / 86400, threshold_type="higher"
)
"""
Wet days (> 1 mm/day) for `pr`.
"""

R10mm = AccumulativeThresholdMetric(
    name="Very wet days \n (> 10 mm/day)", variable="pr", threshold_value=10 / 86400, threshold_type="higher"
)
"""
Very wet days (> 10 mm/day) for `pr`.
"""

R20mm = AccumulativeThresholdMetric(
    name="Extremely wet days \n (> 20 mm/day)", variable="pr", threshold_value=20 / 86400, threshold_type="higher"
)
"""
Extremely wet days (> 20 mm/day) for `pr`.
"""

# ----- tas metrics ----- #
warm_days = ThresholdMetric(name="Mean warm days (K)", variable="tas", threshold_value=295, threshold_type="higher")
"""
Warm days (>295K) for `tas`.
"""

cold_days = ThresholdMetric(name="Mean cold days (K)", variable="tas", threshold_value=275, threshold_type="lower")
"""
Cold days (<275) for `tas`.
"""

# ----- tasmin metrics ----- #
frost_days = ThresholdMetric(
    name="Frost days \n  (tasmin<0°C)", variable="tasmin", threshold_value=273.13, threshold_type="lower"
)
"""
Frost days (<0°C) for `tasmin`.
"""

tropical_nights = ThresholdMetric(
    name="Tropical Nights \n (tasmin>20°C)", variable="tasmin", threshold_value=293.13, threshold_type="higher"
)
"""
Tropical Nights (>20°C) for `tasmin`.
"""

# ----- tasmax metrics ----- #
summer_days = ThresholdMetric(
    name="Summer days \n  (tasmax>25°C)", variable="tasmax", threshold_value=298.15, threshold_type="higher"
)
"""
Summer days (>25°C) for `tasmax`.
"""

icing_days = ThresholdMetric(
    name="Icing days \n (tasmax<0°C)", variable="tasmax", threshold_value=273.13, threshold_type="lower"
)
"""
Icing days (<0°C) for `tasmax`.
"""
