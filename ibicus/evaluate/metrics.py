# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Metrics module - Provides the possiblity to define threshold sensitive climate metrics that are analysed here and used further in :py:mod:`ibicus.marginal`, :py:mod:`ibicus.multivariate` and :py:mod:`ibicus.trend`.
"""

import warnings
from typing import Union

import attrs
import numpy as np
import pandas as pd
from scipy.ndimage import measurements

from ibicus import utils


@attrs.define(eq=False)
class ThresholdMetric:
    """
    Generic climate metric defined by exceedance or underceedance of threshold; or values between an upper and lower threshold. This is determined by ``threshold_type``. These metrics can be defined either overall or daily, monthly, seasonally (``threshold_scope``) and either globally or location-wise (``threshold_locality``).

    Organises the definition and functionalities of such metrics. This enables among others to implement the `ETCCDI <http://etccdi.pacificclimate.org/index.shtml>`_ / `Climdex climate extreme indices <https://www.climdex.org/learn/indices/>`_.

    Attributes
    ----------
    threshold_value : Union[np.array, float, list, dict]
        Threshold value(s) for the variable (in the correct unit).

        - If ``threshold_type = "higher"`` or ``threshold_type = "lower"``, this is just a single float value and the metric is defined as exceedance or underceedance of that value (if `threshold_scope = 'overall'` and `threshold_locality = 'global'`).
        - If ``threshold_type = "between"`` or ``threshold_type = "outside"``, then this needs to be a list in the form: `[lower_bound, upper_bound]` and the metric is defined as falling in between, or falling outside these values (if `threshold_scope = 'overall'` and `threshold_locality = 'global'`).

        - If ``threshold_locality = "local"`` then instead of a single element (within a list, depending on `threshold_type`) a :py:class:`np.ndarray` is stored here for locally defined threshold.
        - If ``threshold_scope`` is one of ``["day", "month", "season"]`` then instead of a (list of) single element(s) or a :py:class:`np.ndarray` a dict is stored whose keys are the times (for example the seasons) and values contain the thresholds (either locally or globally).
    threshold_type : str
        One of ``["higher", "lower", "between", "outside"]``. Indicates whether we are either interested in values above the threshold value (`"higher"`, strict `>`), values below the threshold value (`"lower"`, strict `<`), values between the threshold values (`"between"`, not strict including the bounds) or outside the threshold values (`"outside"`, strict not including the bounds).
    threshold_scope : str = "overall"
        One of ``["day", "month", "season", "overall"]``. Indicates wether thresholds are irrespective of time or defined on a daily, monthly or seasonal basis.
    threshold_locality : str = "global"
        One of ``["global", "local"]``. Indicates wether thresholds are defined globally or locationwise.
    name : str = "unknown"
        Metric name. Will be used in dataframes, plots etc. Recommended to include threshold value and units. Example : `Frost days \\n  (tasmin < 0°C)`. Default: `"unknown"`.
    variable : str = "unknown"
        Unique variable that this threshold metric refers to. Example for frost days: tasmin. Default: `"unknown"`.

    Examples
    --------

    >>> warm_days = ThresholdMetric(threshold_value = 295, threshold_type = "higher", name = "Mean warm days (K)", variable = "tas")
    >>> warm_days_by_season = ThresholdMetric(threshold_value = {"Winter": 290, "Spring": 292, "Summer": 295, "Autumn": 292}, threshold_type = "higher", threshold_scope = "season", name = "Mean warm days (K)", variable = "tas")
    >>> q_90 = ThresholdMetricfrom_quantile(obs, 0.9, threshold_type = "higher", name = "90th observational quantile", variable = "tas")
    >>> q_10_season = ThresholdMetric.from_quantile(obs, 0.1, threshold_type = "lower", threshold_scope = "season", time = time_obs, name = "10th quantile by season", variable = "tas")
    >>> outside_10_90_month_local = ThresholdMetric.from_quantile(obs, [0.1, 0.9], threshold_type = "outside", threshold_scope = "month", threshold_locality = "local", time = time_obs, name = "Outside 10th, 9th quantile by month", variable = "tas")

    """

    threshold_value: Union[np.array, float, list, dict] = attrs.field()
    threshold_type: str = attrs.field(
        validator=attrs.validators.in_(["higher", "lower", "between", "outside"])
    )
    threshold_scope: str = attrs.field(
        default="overall",
        validator=attrs.validators.in_(["day", "month", "season", "overall"]),
    )
    threshold_locality: str = attrs.field(
        default="global",
        validator=attrs.validators.in_(["global", "local"]),
    )
    name: str = attrs.field(
        default="unknown", validator=attrs.validators.instance_of(str)
    )
    variable: str = attrs.field(
        default="unknown", validator=attrs.validators.instance_of(str)
    )

    # Helpers attrs_post_init
    @staticmethod
    def _check_types_locality(threshold_value, threshold_locality):
        if threshold_locality == "global":
            if not isinstance(threshold_value, (float, int)):
                raise ValueError(
                    "If threshold_locality is global then threshold_value should have int/floats as entries (in a dict if threshold_scope != 'overall' and in a list if threshold_type is 'between' or 'outside')."
                )
        elif threshold_locality == "local":
            if not isinstance(threshold_value, (np.ndarray, list)):
                raise ValueError(
                    "If threshold_locality is local then threshold_value should have np.ndarrays (a threshold for each location) as entries (in a dict if threshold_scope != 'overall' and in a list if threshold_type is 'between' or 'outside')."
                )
        else:
            raise ValueError("threshold_scope needs to be 'local' or 'global'.")

    @staticmethod
    def _check_types_scope_and_locality(
        threshold_value, threshold_scope, threshold_locality
    ):
        if threshold_scope in ["day", "month", "season"]:
            if not isinstance(threshold_value, dict):
                raise ValueError(
                    "threshold_value should be a dict with days, months, seasons as keys and thresholds as values for threshold_scope in ['day', 'month', 'season']"
                )
            ThresholdMetric._check_completeness_of_time_categories_and_warn(
                threshold_value, threshold_scope
            )
            for key, value in threshold_value.items():
                ThresholdMetric._check_types_locality(value, threshold_locality)
        elif threshold_scope == "overall":
            ThresholdMetric._check_types_locality(threshold_value, threshold_locality)
        else:
            raise ValueError(
                "threshold_scope needs to be one of ['day', 'month', 'season', 'overall']"
            )

    def __attrs_post_init__(self):
        if self.threshold_type in ["higher", "lower"]:
            ThresholdMetric._check_types_scope_and_locality(
                self.threshold_value, self.threshold_scope, self.threshold_locality
            )
        elif self.threshold_type in ["between", "outside"]:
            if not isinstance(self.threshold_value, (list, tuple)):
                raise ValueError(
                    "threshold_value should be a list with a lower and upper bound for threshold_type in ['between', 'outside']."
                )
            if len(self.threshold_value) != 2:
                raise ValueError(
                    "threshold_value should have (only) a lower and upper bound for threshold_type in ['between', 'outside']."
                )
            ThresholdMetric._check_types_scope_and_locality(
                self.threshold_value[0], self.threshold_scope, self.threshold_locality
            )
            ThresholdMetric._check_types_scope_and_locality(
                self.threshold_value[1], self.threshold_scope, self.threshold_locality
            )

    @staticmethod
    def _get_time_group_by_scope(time, threshold_scope):
        if threshold_scope in ["day", "month", "season"]:
            if time is None:
                raise ValueError(
                    "time argument cannot be None if the threshold is time-sensitive (threshold_scope is one of['day', 'month', 'season'])."
                )
            if threshold_scope == "day":
                return utils.day_of_year(time)
            elif threshold_scope == "month":
                return utils.month(time)
            elif threshold_scope == "season":
                return utils.season(time)
        elif threshold_scope == "overall":
            return None
        else:
            raise ValueError(
                "threshold_scope needs to be one of ['day', 'month', 'season', 'overall']"
            )

    @staticmethod
    def _get_quantile_by_locality(x, q, time, threshold_scope, threshold_locality):
        if threshold_scope == "overall":
            if threshold_locality == "global":
                quantiles = np.quantile(x, q)
            elif threshold_locality == "local":
                quantiles = np.quantile(x, q, axis=0)
            else:
                raise ValueError(
                    "threshold_locality needs to be either 'global' or 'local'"
                )
        else:
            if threshold_locality == "global":
                quantiles = {
                    t: np.quantile(x[np.where(time == t)], q) for t in np.unique(time)
                }
            elif threshold_locality == "local":
                quantiles = {
                    t: [np.quantile(x[np.where(time == t)], q, axis=0)]
                    for t in np.unique(time)
                }
            else:
                raise ValueError(
                    "threshold_locality needs to be either 'global' or 'local'"
                )

        return quantiles

    @staticmethod
    def _check_completeness_of_time_categories_and_warn(thresholds, threshold_scope):
        if threshold_scope == "day":
            days_of_year = np.arange(1, 367)
            if not all(
                mask_days_of_year_in_quantiles := np.in1d(
                    days_of_year, list(thresholds.keys())
                )
            ):
                warnings.warn(
                    "Not all days of year present inside the dataset used for initialisation. Not threshold defined for days %s"
                    % days_of_year[np.logical_not(mask_days_of_year_in_quantiles)]
                )
        elif threshold_scope == "month":
            months = np.arange(1, 13)
            if not all(
                mask_months_in_quantiles := np.in1d(months, list(thresholds.keys()))
            ):
                warnings.warn(
                    "Not all months present inside the dataset used for initialisation. Not threshold defined for months %s"
                    % months[np.logical_not(mask_months_in_quantiles)]
                )
        elif threshold_scope == "season":
            seasons = np.array(["Spring", "Summer", "Autumn", "Winter"])
            if not all(
                mask_seasons_in_quantiles := np.in1d(seasons, list(thresholds.keys()))
            ):
                warnings.warn(
                    "Not all seasons present inside the dataset used for initialisation. Not threshold defined for seasons %s"
                    % seasons[np.logical_not(mask_seasons_in_quantiles)]
                )
        elif threshold_scope == "overall":
            pass
        else:
            raise ValueError(
                "scope needs to be one of ['day', 'month', 'season', 'overall']"
            )

    @staticmethod
    def _get_threshold_from_quantile(
        x, q, time=None, threshold_scope="overall", threshold_locality="global"
    ):
        time = ThresholdMetric._get_time_group_by_scope(time, threshold_scope)

        thresholds = ThresholdMetric._get_quantile_by_locality(
            x,
            q,
            time,
            threshold_scope,
            threshold_locality,
        )

        ThresholdMetric._check_completeness_of_time_categories_and_warn(
            thresholds, threshold_scope
        )

        return thresholds

    @classmethod
    def from_quantile(
        cls,
        x,
        q,
        threshold_type,
        threshold_scope="overall",
        threshold_locality="global",
        time=None,
        name="unknown",
        variable="unknown",
    ):
        """
        Creates a threshold metrics from a quantile respective to an array x.

        Parameters
        ----------
        x : np.ndarray
            Array respective to which the quantile is calculated.
        q : Union[int, float, list]
            Quantile (or list of lower and upper quantile if ``threshold_type`` in ``["higher", "lower"]``) as which the threshold is instantiated.
        threshold_type : str
            One of ``["higher", "lower", "between", "outside"]``. Indicates whether we are either interested in values above the threshold value (`"higher"`, strict `>`), values below the threshold value (`"lower"`, strict `<`), values between the threshold values (`"between"`, strict, not including the bounds) or outside the threshold values (`"outside"`, strict not including the bounds).
        threshold_scope : str = "overall"
            One of ``["day", "month", "season", "overall"]``. Indicates wether thresholds (and the quantiles calculated) are irrespective of time or defined on a daily, monthly or seasonal basis.
        threshold_locality : str = "global"
            One of ``["global", "local"]``. Indicates wether thresholds (and the quantiles calculated) are defined globally or locationwise.
        time: Optional[np.ndarray] = None
            If  the threshold is time-sensitive (``threshold_scope`` in ["day", "month", "season"]) then time information corresponding to `x` is required. Should be a numpy 1d array of times.
        name : str = "unknown"
            Metric name. Will be used in dataframes, plots etc. Recommended to include threshold value and units. Example : 'Frost days \n  (tasmin < 0°C)'. Default: `"unknown"`.
        variable : str = "unknown"
            Unique variable that this threshold metric refers to. Example for frost days: tasmin. Default: `"unknown"`.

        Examples
        --------
        >>> m1 = ThresholdMetric.from_quantile(obs, 0.8, threshold_type = "higher", name = "m1")
        >>> m2 = ThresholdMetric.from_quantile(obs, 0.2, threshold_type = "lower", threshold_scope = "season", threshold_locality = "local", time = time_obs, name = "m2")
        >>> m3 = ThresholdMetric.from_quantile(obs, [0.2, 0.8], threshold_type = "outside", threshold_scope="month", threshold_locality = "local", time=time_obs, name = "m3")


        """
        if threshold_type == "inside" or threshold_type == "outside":
            if not isinstance(q, (list, tuple, np.ndarray)):
                raise ValueError(
                    "If threshold_type is one of ['inside', 'outside'] then q needs to be a list of lower and upper quantile."
                )

            if len(q) != 2:
                raise ValueError(
                    "If threshold_type is one of ['inside', 'outside'] then q needs to be a list of lower and upper quantile."
                )
            if not q[0] < q[1]:
                raise ValueError(
                    "The lower quantile needs to be smaller than the upper quantile."
                )

            threshold_value = [
                ThresholdMetric._get_threshold_from_quantile(
                    x,
                    q[0],
                    time=time,
                    threshold_scope=threshold_scope,
                    threshold_locality=threshold_locality,
                ),
                ThresholdMetric._get_threshold_from_quantile(
                    x,
                    q[1],
                    time=time,
                    threshold_scope=threshold_scope,
                    threshold_locality=threshold_locality,
                ),
            ]
        else:
            threshold_value = ThresholdMetric._get_threshold_from_quantile(
                x,
                q,
                time=time,
                threshold_scope=threshold_scope,
                threshold_locality=threshold_locality,
            )

        return cls(
            threshold_value=threshold_value,
            threshold_scope=threshold_scope,
            threshold_type=threshold_type,
            threshold_locality=threshold_locality,
            name=name,
            variable=variable,
        )

    def _get_mask_higher_or_lower(self, x, threshold_value, higher_or_lower, time=None):
        if self.threshold_scope == "overall":
            thresholds = threshold_value

            # Extend upon spatial dimension
            if self.threshold_locality == "local":
                thresholds = thresholds[None, :, :]
        elif self.threshold_scope in ["day", "month", "season"]:
            if time is None:
                raise ValueError(
                    "For ThresholdMetrics with scope ['day', 'month', 'season'] time information is required to calculate the score"
                )

            time = ThresholdMetric._get_time_group_by_scope(time, self.threshold_scope)

            if not np.all(np.in1d(time, list(threshold_value.keys()))):
                raise ValueError(
                    "time contains values for %ss for which no thresholds exist in self.threshold_value"
                    % self.threshold_scope
                )

            thresholds = (
                pd.DataFrame({"time": time})
                .merge(
                    pd.DataFrame(
                        threshold_value.items(), columns=["time", "threshold"]
                    ),
                    how="left",
                    on="time",
                )
                .threshold.values
            )

            if self.threshold_locality == "local":
                thresholds = np.concatenate(thresholds)
            elif self.threshold_locality == "global":
                thresholds = np.array(thresholds)
                # Extend upon spatial dimension
                thresholds = thresholds[:, None, None]
            else:
                ValueError("self.threshold_locality needs to be global or local")
        else:
            raise ValueError(
                "self.threshold_scope needs to be one of ['day', 'month', 'season', 'overall']"
            )

        if higher_or_lower == "higher":
            return x > thresholds
        elif higher_or_lower == "lower":
            return x < thresholds
        else:
            raise ValueError("higher_or_lower needs to be one of ['higher', 'lower'].")

    def _get_mask_threshold_condition(self, x, time=None):
        if self.threshold_type == "higher":
            return self._get_mask_higher_or_lower(
                x, self.threshold_value, "higher", time
            )
        elif self.threshold_type == "lower":
            return self._get_mask_higher_or_lower(
                x, self.threshold_value, "lower", time
            )
        elif self.threshold_type == "between":
            return np.logical_and(
                self._get_mask_higher_or_lower(
                    x, self.threshold_value[0], "higher", time
                ),
                self._get_mask_higher_or_lower(
                    x, self.threshold_value[1], "lower", time
                ),
            )
        elif self.threshold_type == "outside":
            return np.logical_or(
                self._get_mask_higher_or_lower(
                    x, self.threshold_value[0], "lower", time
                ),
                self._get_mask_higher_or_lower(
                    x, self.threshold_value[1], "higher", time
                ),
            )
        else:
            raise ValueError(
                "Invalid self.threshold_type. Needs to be one of ['higher', 'lower', 'between']. Modify the class attribute."
            )

    def calculate_instances_of_threshold_exceedance(
        self, dataset: np.ndarray, time: np.ndarray = None
    ) -> np.ndarray:
        """
        Returns an array of the same size as `dataset` containing 1 when the threshold condition is met and 0 when not.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected.
        time : np.ndarray = None
            Time corresponding to each observation in dataset, required only for time sensitive thresholds (threshold_scope = ['day', 'month', 'year']).
        """
        return self._get_mask_threshold_condition(dataset, time=time).astype(int)

    def filter_threshold_exceedances(
        self, dataset: np.ndarray, time: np.ndarray = None
    ) -> np.ndarray:
        """
        Returns an array containing the values of dataset where the threshold condition is met and zero where not.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected.
        time : np.ndarray = None
            Time corresponding to each observation in dataset, required only for time sensitive thresholds (threshold_scope = ['day', 'month', 'year']).
        """

        mask_threshold_condition = self._get_mask_threshold_condition(
            dataset, time=time
        )
        dataset[mask_threshold_condition] = 0
        return dataset

    def calculate_exceedance_probability(
        self, dataset: np.ndarray, time: np.ndarray = None
    ) -> np.ndarray:
        """
        Returns the probability of metrics occurrence (threshold exceedance/underceedance or inside/outside range), at each location (across the entire time period).

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected
        time : np.ndarray = None
            Time corresponding to each observation in dataset, required only for time sensitive thresholds (threshold_scope = ['day', 'month', 'year']).

        Returns
        -------
        np.ndarray
            Probability of metric occurrence at each location.
        """

        threshold_data = self.calculate_instances_of_threshold_exceedance(
            dataset, time=time
        )
        threshold_probability = (
            np.einsum("ijk -> jk", threshold_data) / threshold_data.shape[0]
        )
        return threshold_probability

    def calculate_number_annual_days_beyond_threshold(
        self, dataset: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        """
        Calculates number of days beyond threshold for each year in the dataset.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected.
        time : np.ndarray
            Time corresponding to each observation in dataset, required to calculate annual threshold occurrences.

        Returns
        -------
        np.ndarray
            3d array - [years, lat, long]
        """

        eot_matrix = self.calculate_instances_of_threshold_exceedance(
            dataset, time=time
        )

        time_array = utils.year(time)

        years = np.unique(time_array)

        threshold_exceedances = np.zeros(
            (years.shape[0], dataset.shape[1], dataset.shape[2])
        )

        for j in range(eot_matrix.shape[1]):
            for k in range(eot_matrix.shape[2]):
                threshold_exceedances[:, j, k] = [
                    (eot_matrix[time_array == i, j, k].sum()) for i in years
                ]

        return threshold_exceedances

    @staticmethod
    def _calculate_spell_lengths_one_location(mask_threshold_condition_one_location):
        return np.diff(
            np.where(
                np.concatenate(
                    (
                        [mask_threshold_condition_one_location[0]],
                        mask_threshold_condition_one_location[:-1]
                        != mask_threshold_condition_one_location[1:],
                        [True],
                    )
                )
            )[0]
        )[::2]

    def calculate_spell_length(
        self, minimum_length: int, **climate_data
    ) -> pd.DataFrame:
        """
        Returns a :py:class:`pd.DataFrame` of individual spell lengths of metrics occurrences (threshold exceedance/underceedance or inside/outside range), counted across locations, for each climate dataset specified in `**climate_data`.

        A spell length is defined as the number of days that a threshold is continuesly exceeded, underceeded or where values are continuously between or outside the threshold (depending on ``self.threshold_type``).
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser as specified in `**climate_data`, 'Metric' - name of the threshold metric, 'Spell length - individual spell length counts'.

        Parameters
        ----------
        minimum length : int
            Minimum spell length (in days) investigated.
        climate_data :
            Keyword arguments, providing the input data to investigate. Should be :py:class:`np.ndarrays` of observations or if the threshold is time sensitive (``threshold_scope = ['day', 'month', 'year']``) lists of `[cm_data, time_cm_data]` where `time_cm_data` are 1d numpy arrays of times corresponding the the values in `cm_data`.

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
            if isinstance(climate_data_value, (list, tuple)):
                mask_threshold_condition = self._get_mask_threshold_condition(
                    climate_data_value[0], time=climate_data_value[1]
                )
            else:
                if self.threshold_scope in ["day", "month", "season"]:
                    raise ValueError(
                        "time information is required if threshold scope is one of ['day', 'month', 'season']. Please pass lists of structure [climate_data, time_information] as key words arguments."
                    )
                mask_threshold_condition = self._get_mask_threshold_condition(
                    climate_data_value, time=None
                )

            spell_length = []
            for i, j in np.ndindex(mask_threshold_condition.shape[1:]):
                spell_length.append(
                    ThresholdMetric._calculate_spell_lengths_one_location(
                        mask_threshold_condition[:, i, j]
                    )
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
        plot_data["Spell length (days)"] = pd.to_numeric(
            plot_data["Spell length (days)"]
        )

        return plot_data

    def calculate_spatial_extent(self, **climate_data):
        """
        Returns a :py:class:`pd.DataFrame` of spatial extends of metrics occurrences (threshold exceedance/underceedance or inside/outside range), for each climate dataset specified in `**climate_data`.

        The spatial extent is defined as the percentage of the area where the threshold is exceeded/underceeded or values are between or outside the bounds (depending on ``self.threshold_type``), given that it is exceeded at one location.
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser, 'Metric' - name of the threshold metric, 'Spatial extent (% of area)'

        Parameters
        ----------
        **climate_data :
            Keyword arguments, providing the input data to investigate. Should be `np.ndarrays` of observations or if the threshold is time sensitive (threshold_scope = ['day', 'month', 'year']) lists of `[cm_data, time_cm_data]` where `time_cm_data` are 1d numpy arrays of times corresponding the the values in `cm_data`.

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
            if isinstance(climate_data_value, (list, tuple)):
                threshold_data = self.calculate_instances_of_threshold_exceedance(
                    climate_data_value[0], time=climate_data_value[1]
                )
            else:
                if self.threshold_scope in ["day", "month", "season"]:
                    raise ValueError(
                        "time information is required if threshold scope is one of ['day', 'month', 'season']. Please pass lists of structure [climate_data, time_information] as key words arguments."
                    )
                threshold_data = self.calculate_instances_of_threshold_exceedance(
                    climate_data_value, time=None
                )

            spatial_clusters = np.einsum("ijk -> i", threshold_data) / np.prod(
                threshold_data.shape[1:]
            )
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
        plot_data["Spatial extent (% of area)"] = pd.to_numeric(
            plot_data["Spatial extent (% of area)"]
        )

        return plot_data

    def calculate_spatiotemporal_clusters(self, **climate_data):
        """
        Returns a py:class:`pd.DataFrame` of sizes of individual spatiotemporal clusters of metrics occurrences (threshold exceedance/underceedance or inside/outside range), for each climate dataset specified in `**climate_data`.

        A spatiotemporal cluster is defined as a connected set (in time and/or space) where the threshold is exceeded/underceeded or values are between or outside the bounds (depending on ``self.threshold_type``).
        The output dataframe has three columns: 'Correction Method' - obs/raw or name of debiaser, 'Metric' - name of the threshold metric, 'Spatiotemporal cluster size'

        Parameters
        ----------
        climate_data :
            Keyword arguments, providing the input data to investigate. Should be `np.ndarrays` of observations or if the threshold is time sensitive (threshold_scope = ['day', 'month', 'year']) lists of `[cm_data, time_cm_data]` where `time_cm_data` are 1d numpy arrays of times corresponding the the values in `cm_data`.

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
            if isinstance(climate_data_value, (list, tuple)):
                threshold_data = self.calculate_instances_of_threshold_exceedance(
                    climate_data_value[0], time=climate_data_value[1]
                )
            else:
                if self.threshold_scope in ["day", "month", "season"]:
                    raise ValueError(
                        "time information is required if threshold scope is one of ['day', 'month', 'season']. Please pass lists of structure [climate_data, time_information] as key words arguments."
                    )
                threshold_data = self.calculate_instances_of_threshold_exceedance(
                    climate_data_value, time=None
                )

            threshold_data_lw, _ = measurements.label(threshold_data)
            area = measurements.sum(
                threshold_data,
                threshold_data_lw,
                index=np.arange(threshold_data_lw.max() + 1),
            )

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
        plot_data["Spatiotemporal cluster size"] = pd.to_numeric(
            plot_data["Spatiotemporal cluster size"]
        )

        return plot_data


@attrs.define
class AccumulativeThresholdMetric(ThresholdMetric):
    """
    Climate for metrics that are defined by thresholds (child class of :py:class:`ThresholdMetric`), but are accumulative. This mainly concerns precipitation metrics.

    An example of such a metric is "total precipitation by very wet days (days > 10mm precipitation)".

    Examples
    --------
    >>> R10mm = AccumulativeThresholdMetric(name="Very wet days (> 10 mm/day)", variable="pr", threshold_value=10 / 86400,threshold_type="higher")
    """

    def calculate_percent_of_total_amount_beyond_threshold(
        self, dataset: np.ndarray, time=None
    ) -> np.ndarray:
        """
        Calculates percentage of total amount beyond threshold for each location over all timesteps.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
        time : np.ndarray = None
            Time corresponding to each observation in dataset, required only for time sensitive thresholds (threshold_scope = ['day', 'month', 'year']).

        Returns
        -------
        np.ndarray
            2d array with percentage of total amount above threshold at each location.
        """

        eot_matrix = self.filter_threshold_exceedances(dataset, time)

        exceedance_percentage = (
            100 * np.einsum("ijk -> jk", eot_matrix) / np.einsum("ijk -> jk", dataset)
        )

        return exceedance_percentage

    def calculate_annual_value_beyond_threshold(
        self, dataset: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        """
        Calculates amount beyond threshold for each year in the dataset.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projections to be analysed, numeric entries expected.
        time : np.ndarray
            Time corresponding to each observation in dataset, required to calculate annual threshold occurrences.

        Returns
        -------
        np.ndarray
            3d array - [years, lat, long]
        """

        eot_matrix = self.filter_threshold_exceedances(dataset, time=time)

        time_array = utils.year(time)

        years = np.unique(time_array)

        threshold_exceedances = np.zeros(
            (years.shape[0], dataset.shape[1], dataset.shape[2])
        )

        for j in range(eot_matrix.shape[1]):
            for k in range(eot_matrix.shape[2]):
                threshold_exceedances[:, j, k] = [
                    (eot_matrix[time_array == i, j, k].sum()) for i in years
                ]

        return threshold_exceedances

    def calculate_intensity_index(self, dataset, time=None):
        """
        Calculates the amount beyond a threshold divided by the number of instance the threshold is exceeded.

        Designed to calculate the simple precipitation intensity index but can be used for other variables.

        Parameters
        ----------
        dataset : np.ndarray
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected.
        time : np.ndarray = None
            Time corresponding to each observation in dataset, required only for time sensitive thresholds (threshold_scope = ['day', 'month', 'year']).
        """

        eot_value_matrix = self.filter_threshold_exceedances(dataset, time)
        eot_threshold_matrix = self.calculate_instances_of_threshold_exceedance(dataset)
        intensity_index = np.einsum("ijk -> jk", eot_value_matrix) / np.einsum(
            "ijk -> jk", eot_threshold_matrix
        )
        return intensity_index


# ----- pr metrics ----- #
dry_days = AccumulativeThresholdMetric(
    name="Dry days \n (< 1 mm/day)",
    variable="pr",
    threshold_value=1 / 86400,
    threshold_type="lower",
)
"""
Dry days (< 1 mm/day) for `pr`.
"""

wet_days = AccumulativeThresholdMetric(
    name="Wet days \n (> 1 mm/day)",
    variable="pr",
    threshold_value=1 / 86400,
    threshold_type="higher",
)
"""
Wet days (> 1 mm/day) for `pr`.
"""

R10mm = AccumulativeThresholdMetric(
    name="Very wet days \n (> 10 mm/day)",
    variable="pr",
    threshold_value=10 / 86400,
    threshold_type="higher",
)
"""
Very wet days (> 10 mm/day) for `pr`.
"""

R20mm = AccumulativeThresholdMetric(
    name="Extremely wet days \n (> 20 mm/day)",
    variable="pr",
    threshold_value=20 / 86400,
    threshold_type="higher",
)
"""
Extremely wet days (> 20 mm/day) for `pr`.
"""

# ----- tas metrics ----- #
warm_days = ThresholdMetric(
    name="Mean warm days (K)",
    variable="tas",
    threshold_value=295,
    threshold_type="higher",
)
"""
Warm days (>295K) for `tas`.
"""

cold_days = ThresholdMetric(
    name="Mean cold days (K)",
    variable="tas",
    threshold_value=275,
    threshold_type="lower",
)
"""
Cold days (<275) for `tas`.
"""

# ----- tasmin metrics ----- #
frost_days = ThresholdMetric(
    name="Frost days \n  (tasmin<0°C)",
    variable="tasmin",
    threshold_value=273.13,
    threshold_type="lower",
)
"""
Frost days (<0°C) for `tasmin`.
"""

tropical_nights = ThresholdMetric(
    name="Tropical Nights \n (tasmin>20°C)",
    variable="tasmin",
    threshold_value=293.13,
    threshold_type="higher",
)
"""
Tropical Nights (>20°C) for `tasmin`.
"""

# ----- tasmax metrics ----- #
summer_days = ThresholdMetric(
    name="Summer days \n  (tasmax>25°C)",
    variable="tasmax",
    threshold_value=298.15,
    threshold_type="higher",
)
"""
Summer days (>25°C) for `tasmax`.
"""

icing_days = ThresholdMetric(
    name="Icing days \n (tasmax<0°C)",
    variable="tasmax",
    threshold_value=273.13,
    threshold_type="lower",
)
"""
Icing days (<0°C) for `tasmax`.
"""
