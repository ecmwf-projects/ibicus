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
from pylab import arange
from scipy.ndimage import measurements

from PACKAGE_NAME import utils


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

    Example
    -------

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
            return np.logical_and(x >= self.threshold_value[0], x <= self.threshold_value[1])
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
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
        """

        mask_threshold_condition = self._get_mask_threshold_condition(dataset)
        dataset[mask_threshold_condition] = 0
        return dataset

    def calculate_exceedance_probability(self, dataset: np.ndarray) -> np.ndarray:
        """
        Returns the probability of exceeding a specified threshold at each location (across the entire time period).

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected
        """

        threshold_data = self.calculate_instances_of_threshold_exceedance(dataset)
        threshold_probability = np.einsum("ijk -> jk", threshold_data) / threshold_data.shape[0]
        return threshold_probability

    def calculate_spell_length(self, minimum_length: int, **climate_data) -> pd.DataFrame:
        """
        Returns a `py:class:`pd.DataFrame` of individual spell lengths of metrics
        Converts input climate data into a pandas Dataframe of individual spell lengths counted across locations in the dataframe.

        A spell length is defined as the number of days that a threshold is continuesly exceeded.
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser, 'Metric' - name of the threshold metric, 'Spell length (days) - individual spell length counts'

        Parameters
        ----------
        minimum length : int
            Minimum spell length (in days) investigated.
        climate_data :
            Keyword arguments, providing the input data to investigate.

        Example of how this function is used:

        >>> dry_days.calculate_spell_length(minimum_length = 4, obs = tas_obs_validate, raw = tas_cm_validate, ISIMIP = tas_val_debiased_ISIMIP)
        """

        spell_length_array = np.empty((0, 3))

        for k in climate_data.keys():

            threshold_data = self.calculate_instances_of_threshold_exceedance(climate_data[k])
            spell_length = np.array([])

            for i in range(threshold_data.shape[1]):
                for j in range(threshold_data.shape[2]):
                    N = 0
                    for t in range(threshold_data.shape[0]):
                        if threshold_data[t, i, j] == 1:
                            N = N + 1
                        elif (threshold_data[t, i, j] == 0) and (N != 0):
                            spell_length = np.append(spell_length, N)
                            N = 0

            spell_length = spell_length[spell_length > minimum_length]

            spell_length_array = np.append(
                spell_length_array,
                np.transpose(
                    np.array(
                        [
                            [k] * len(spell_length),
                            [self.name] * len(spell_length),
                            np.transpose(spell_length),
                        ]
                    )
                ),
                axis=0,
            )

        plot_data = pd.DataFrame(spell_length_array, columns=["Correction Method", "Metric", "Spell length (days)"])
        plot_data["Spell length (days)"] = pd.to_numeric(plot_data["Spell length (days)"])

        return plot_data

    def calculate_spatial_clusters(self, **climate_data):
        """
        Converts input climate data into a pandas Dataframe of spatial extents of threshold exceedances.

        The spatial extent is defined as the percentage of the area where the threshold is exceeded, given that it is exceeded at one location.
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser, 'Metric' - name of the threshold metric, 'Spatial extent (% of area)'

        Parameters
        ----------
        climate_data :
            Keyword arguments, providing the input data to investigate.

        Example of how this function is used:

        >>> dry_days.calculate_spatial_extent(obs = tas_obs_validate, raw = tas_cm_validate, ISIMIP = tas_val_debiased_ISIMIP)
        """

        clusters_array = np.empty((0, 3))

        for k in climate_data.keys():

            spatial_count = np.array([])

            number_gridpoints = climate_data[k].shape[1] * climate_data[k].shape[2]

            threshold_data = self.calculate_instances_of_threshold_exceedance(dataset=climate_data[k])

            for i in range(threshold_data.shape[0]):

                count = np.sum(threshold_data[i, :, :]) / number_gridpoints
                spatial_count = np.append(spatial_count, count)

            spatial_count = spatial_count[spatial_count != 0]

            clusters_array = np.append(
                clusters_array,
                np.transpose(
                    np.array(
                        [
                            [k] * len(spatial_count),
                            [self.name] * len(spatial_count),
                            np.transpose(spatial_count),
                        ]
                    )
                ),
                axis=0,
            )

        spatial_clusters = pd.DataFrame(
            clusters_array, columns=["Correction Method", "Metric", "Spatial extent (% of area)"]
        )
        spatial_clusters["Spatial extent (% of area)"] = pd.to_numeric(spatial_clusters["Spatial extent (% of area)"])

        return spatial_clusters

    def calculate_spatiotemporal_clusters(self, **climate_data):
        """
        Converts input climate data into a pandas Dataframe detailing the size of individual spatiotemporal clusters of threshold exceedances.

        A spatiotemporal cluster is defined as a connected set (in time and/or space) of threshold exceedances.
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser, 'Metric' - name of the threshold metric, 'Spatiotemporal cluster size'

        Parameters
        ----------
        climate_data :
            Keyword arguments, providing the input data to investigate.

        Example of how this function is used:

        >>> dry_days.calculate_spatiotemporal_clusters(obs = tas_obs_validate, raw = tas_cm_validate, ISIMIP = tas_val_debiased_ISIMIP)
        """

        clusters_array = np.empty((0, 3))

        for k in climate_data.keys():

            threshold_data = self.calculate_instances_of_threshold_exceedance(dataset=climate_data[k])
            threshold_data_lw, threshold_data_num = measurements.label(threshold_data)
            area = measurements.sum(threshold_data, threshold_data_lw, index=arange(threshold_data_lw.max() + 1))

            clusters_array = np.append(
                clusters_array,
                np.transpose(np.array([[k] * len(area), [self.name] * len(area), np.transpose(area)])),
                axis=0,
            )

        spatiotemporal_clusters = pd.DataFrame(
            clusters_array, columns=["Correction Method", "Metric", "Spatiotemporal cluster size"]
        )
        spatiotemporal_clusters["Spatiotemporal cluster size"] = pd.to_numeric(
            spatiotemporal_clusters["Spatiotemporal cluster size"]
        )

        return spatiotemporal_clusters

    def plot_clusters_violinplots(self, minimum_length, **climate_data):
        """
        Takes pandas dataframes of temporal, spatial and spatiotemporal extent as input and outputs three violinplot comparing the obs/raw/debiasers specified in the dataframes.

        Parameters
        ----------
        temporal_data: pd.DataFrame
            pandas dataframe of type output by function _calculate_spell_length
        spatial_data: pd.DataFrame
            pandas dataframe of type output by function _calculate_spatial_clusters
        spatiotemporal_data: pd.DataFrame
            pandas dataframe of type output by function _calculate_spatiotemporal_clusters

        """

        temporal_data = self.calculate_spell_length(minimum_length, **climate_data)
        spatial_data = self.calculate_spatial_clusters(**climate_data)
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
    Child-class of ThresholdMetric class. Adds functionalities for metrics that are accumulative such as precipitation.

    """

    def calculate_mean_value_beyond_threshold(self, dataset: np.ndarray, percentage: bool = True) -> np.ndarray:

        """
        Calculates mean total or percentage value beyond threshold, returns 2d array with one value per location.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
        percentage : bool
            Function calculates percentage value if True, absolut annual value if False. Default: True
        """

        eot_matrix = self.filter_threshold_exceedances(dataset)

        years_in_dataset = dataset.shape[0] / 365

        if percentage is True:

            exceedance_amount = 100 * np.einsum("ijk -> jk", eot_matrix) / np.einsum("ijk -> jk", dataset)

        else:

            exceedance_amount = np.einsum("ijk -> jk", eot_matrix) / years_in_dataset

        return exceedance_amount

    def calculate_annual_value_beyond_threshold(
        self, dataset: np.ndarray, time_dictionary, time_specification: str, time_func=utils.year, percentage=False
    ) -> np.ndarray:

        """
        Calculates annual total or percentage value beyond threshold for each year in the dataset.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
        time_dictionary : dict
            Dictionary of dates, specified from original netCDF4 input files.
        time_specification : str
            Dictionary keyword pointing to the dates needed for this dataset.
        time_func : functions
            Points to utils function to either extract days or months.
        percentage : bool
            Function calculates percentage value if True, absolut annual value if False. Default: False
        """

        eot_matrix = self.filter_threshold_exceedances(dataset)

        time_array = time_func(time_dictionary[time_specification])

        years = np.unique(time_array)

        threshold_exceedances = np.zeros((years.shape[0], dataset.shape[1], dataset.shape[2]))

        for j in range(eot_matrix.shape[1]):
            for k in range(eot_matrix.shape[2]):

                if percentage is True:

                    threshold_exceedances[:, j, k] = np.asarray(
                        [(eot_matrix[time_array == i, j, k].sum()) for i in years]
                    ) / np.asarray([(dataset[time_array == i, j, k].sum()) for i in years])

                else:

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


# pr metrics
dry_days = AccumulativeThresholdMetric(
    name="Dry days \n (< 1 mm/day)", variable="pr", threshold_value=1 / 86400, threshold_type="lower"
)
wet_days = AccumulativeThresholdMetric(
    name="Wet days \n (> 1 mm/day)", variable="pr", threshold_value=1 / 86400, threshold_type="higher"
)
R10mm = AccumulativeThresholdMetric(
    name="Very wet days \n (> 10 mm/day)", variable="pr", threshold_value=10 / 86400, threshold_type="higher"
)
R20mm = AccumulativeThresholdMetric(
    name="Extremely wet days \n (> 20 mm/day)", variable="pr", threshold_value=20 / 86400, threshold_type="higher"
)

# tas metrics
warm_days = ThresholdMetric(name="Mean warm days (K)", variable="tas", threshold_value=295, threshold_type="higher")
cold_days = ThresholdMetric(name="Mean cold days (K)", variable="tas", threshold_value=275, threshold_type="lower")

# tasmin metrics
frost_days = ThresholdMetric(
    name="Frost days \n  (tasmin<0°C)", variable="tasmin", threshold_value=273.13, threshold_type="lower"
)
tropical_nights = ThresholdMetric(
    name="Tropical Nights \n (tasmin>20°C)", variable="tasmin", threshold_value=293.13, threshold_type="higher"
)

# tasmax metrics
summer_days = ThresholdMetric(
    name="Summer days \n  (tasmax>25°C)", variable="tasmax", threshold_value=298.15, threshold_type="higher"
)
icing_days = ThresholdMetric(
    name="Icing days \n (tasmax<0°C)", variable="tasmax", threshold_value=273.13, threshold_type="lower"
)
