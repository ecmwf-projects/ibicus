# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

import attrs
import numpy as np


@attrs.define
class RunningWindowOverYears:
    """
    Implements a running window mode iterating over years.

    Usual usage:
    >>> years = np.arange(2000, 2050)
    >>> rolling_window = RunningWindowOverYears(window_length_in_years = 17, window_step_length_in_years = 9)
    >>> for years_to_debias, years_in_window in rolling_window.use(years):
    ...     # do some calculations with both
    ...     print(years_to_debias)
    ...     print(years_in_window)

    Warning: currently only uneven sizes are allowed for window_length_in_years and window_step_length_in_years. This allows symmetrical windows of the form [window_center - self.window_step_length_in_years//2, window_center + self.window_step_length_in_years//2] for the years to adjust and similar for the years in window.

    Depending on returns (one of ["years", "indices", "mask"]) the behavior and return value of use is different.
        If self.returns = "years" then the selected years inside the window and inside of years to adjust are returned.
        If self.returns = "indices" then indices inside of the array years are returned for the selected years inside the window and inside of years to adjust.
        If self.returns = "mask" then a mask for years is returned for the selected years inside the window and inside of years to adjust.

    Attributes
    ----------
    window_length_in_years: int
        Length of the running window in years: how many values are used to in the calculations later.
    window_step_length_in_years: int
        Step length of the running window in years: how many values are adjusted inside the running window.
    returns: str
        One of ["years", "indices", "mask"].
        If "years" then the selected years inside the window and inside of years to adjust are returned in use.
        If "indices" then indices inside of the array years are returned.
        If "mask" then a mask for years is returned.

    """

    window_length_in_years: int = attrs.field(
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
        converter=round,
    )
    window_step_length_in_years: int = attrs.field(
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
        converter=round,
    )
    returns: str = attrs.field(
        default="years", validator=attrs.validators.in_(["years", "indices", "mask"])
    )

    def __attrs_post_init__(self):
        if self.window_length_in_years % 2 == 0:
            warnings.warn(
                "Currently only uneven window lengths are allowed for window_length_in_years. Automatically increased by 1.",
                stacklevel=2,
            )
            self.window_length_in_years = self.window_length_in_years + 1
        if self.window_step_length_in_years % 2 == 0:
            warnings.warn(
                "Currently only uneven step lengths are allowed for window_step_length_in_years. Automatically increased by 1.",
                stacklevel=2,
            )
            self.window_step_length_in_years = self.window_step_length_in_years + 1

    # ----- Helpers: get window centers and given a window center the years to adjust and the years in the window used for calculations -----
    def _get_years_forming_window_centers(self, unique_years: np.ndarray) -> np.ndarray:
        """
        Given an array of years present in the data this returns an array of window-centers: years that form the center of a running window of size self.window_length_in_years moved in steps of self.window_step_length_in_years.

        Parameters
        ----------
        unique_years : np.ndarray
            Unique years present in the data.
        """
        number_of_years = unique_years.size

        if number_of_years <= self.window_step_length_in_years:
            return np.array([np.round(np.median(unique_years))])

        if (
            years_left_after_last_step := number_of_years
            % self.window_step_length_in_years
        ) == 0:
            first_window_center = (
                unique_years.min() + self.window_step_length_in_years // 2
            )
        else:
            first_window_center = (
                unique_years.min()
                + self.window_step_length_in_years // 2
                - (self.window_step_length_in_years - years_left_after_last_step) // 2
            )

        window_centers = np.arange(
            first_window_center,
            unique_years.max() + 1,
            self.window_step_length_in_years,
        )

        return window_centers

    def _get_years_in_window(self, window_center: int) -> np.ndarray:
        """
        Given a window center (a year forming the center of a window) this returns an array of all other years inside this window of size self.window_length_in_years.

        Parameters
        ----------
        window_center: int
            Window center around which in each year a window of length self.window_length is taken and the indices returned
        """
        years_in_window = np.arange(
            window_center - self.window_length_in_years // 2,
            window_center + self.window_length_in_years // 2 + 1,
        )
        return years_in_window

    def _get_years_in_window_that_are_adjusted(self, window_center: int) -> np.ndarray:
        """
        Given a window center (a year forming the center of a window) this returns an array of the years inside that window that are bias corrected.
        In a window of size self.window_length_in_years those are [window_center - self.window_step_length_in_years//2, window_center + self.window_step_length_in_years//2].

        Parameters
        ----------
        window_center: int
            Window center around which in each year a window of length self.window_length is taken and the indices returned
        """
        years_to_adjust = np.arange(
            window_center - self.window_step_length_in_years // 2,
            window_center + self.window_step_length_in_years // 2 + 1,
        )
        return years_to_adjust

    # ----- Main methods ----- #
    @staticmethod
    def get_if_in_chosen_years(years, chosen_years):
        """
        Given an array of years this returns an array of bools indicating whether a year in years is inside of chosen_years.

        >>> chosen_years = np.array([2022, 2023])
        >>> years = np.arange(2020, 2030)
        >>> RunningWindowOverYears.get_if_in_chosen_years(years, chosen_years)
        array([False, False,  True,  True, False, False, False, False, False, False])

        Parameters
        ----------
        years : np.ndarray
            Array of years.
        chosen_years : np.ndarray
            Array of chosen years.
        """
        return np.in1d(years, chosen_years)

    def use(self, years):
        """
        This applies the running window onto an array of years.
        It returns an iterator of (years_to_adjust, years_in_window) giving:
            years_to_adjust: the years to adjust/modify inside the current window. Given a window center those are [window_center - self.window_step_length_in_years//2, window_center + self.window_step_length_in_years//2 + 1].
            years_in_window: the years inside the current window used for calculations. Given a window center those are [window_center - self.window_length_in_years//2, window_center + self.window_length_in_years//2], so all the years inside the running window of length window_length_in_years.

        Usual usage:
        >>> years = np.arange(2000, 2050)
        >>> rolling_window = RunningWindowOverYears(window_length_in_years = 17, window_step_length_in_years = 9)
        >>> for years_to_debias, years_in_window in rolling_window.use(years):
        ...     # do some calculations with both
        ...     print(years_to_debias)
        ...     print(years_in_window)

        Depending on self.returns (one of ["years", "indices", "mask"]) the behavior and return value is different.
            If self.returns = "years" then the selected years inside the window and inside of years to adjust are returned.
            If self.returns = "indices" then indices inside of the array years are returned for the selected years inside the window and inside of years to adjust.
            If self.returns = "mask" then a mask for years is returned for the selected years inside the window and inside of years to adjust.

        Parameters
        ----------
        years : np.ndarray
            Array of consecutive years on which the running window is calculated/applied on.
        """
        window_centers = self._get_years_forming_window_centers(np.unique(years))
        for window_center in window_centers:
            years_to_adjust = self._get_years_in_window_that_are_adjusted(window_center)
            years_in_window = self._get_years_in_window(window_center)
            if self.returns == "years":
                yield years_to_adjust, years_in_window
            elif self.returns == "mask":
                yield RunningWindowOverYears.get_if_in_chosen_years(
                    years_to_adjust, years
                ), RunningWindowOverYears.get_if_in_chosen_years(years_in_window, years)
            elif self.returns == "indices":
                yield np.where(
                    RunningWindowOverYears.get_if_in_chosen_years(
                        years_to_adjust, years
                    )
                )[0], np.where(
                    RunningWindowOverYears.get_if_in_chosen_years(
                        years_in_window, years
                    )
                )[
                    0
                ]
            else:
                raise ValueError(
                    'self.returns needs to be one of ["years", "indices", "mask"]'
                )


@attrs.define
class RunningWindowOverDaysOfYear:
    """
    Implements a running window mode iterating over the days of a year.

    Usual usage:
    >>> from ibicus.utils import create_array_of_consecutive_dates, day_of_year, year
    >>> dates = create_array_of_consecutive_dates(1000)
    >>> days_of_year, years = day_of_year(dates), year(dates)
    >>> rolling_window = RunningWindowOverDaysOfYear(window_length_in_days = 31, window_step_length_in_days = 3)
    >>> for window_center, indices_vals_to_debias in rolling_window.use(days_of_year, years):
    ...     # do some calculations with both
    ...     print(indices_vals_to_debias)
    ...     indices_vals_in_window = rolling_window.get_indices_vals_in_window(days_of_year, window_center)
    ...     print(indices_vals_in_window)

    Warning: currently only uneven sizes are allowed for window_length_in_days and window_step_length_in_days. This allows symmetrical windows of the form [window_center - self.window_step_length_in_days//2, window_center + self.window_step_length_in_days//2] for the days of year to adjust and similar for the days of year in window.

    In contrast to RunningWindowOverYears this currently only returns only indices of values. Adjusting for leap years is difficult otherwise.

    Attributes
    ----------
    window_length_in_days: int
        Length of the running window in days: how many values are used to in the calculations later.
    window_step_length_in_days: int
        Step length of the running window in days: how many values are adjusted inside the running window.
    """

    window_length_in_days: int = attrs.field(
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
        converter=round,
    )
    window_step_length_in_days: int = attrs.field(
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
        converter=round,
    )

    def __attrs_post_init__(self):
        if self.window_length_in_days % 2 == 0:
            warnings.warn(
                "Currently only uneven window lengths are allowed for window_length_in_days. Automatically increased by 1.",
                stacklevel=2,
            )
            self.window_length_in_days = self.window_length_in_days + 1
        if self.window_step_length_in_days % 2 == 0:
            warnings.warn(
                "Currently only uneven step lengths are allowed for window_step_length_in_years. Automatically increased by 1.",
                stacklevel=2,
            )
            self.window_step_length_in_days = self.window_step_length_in_days + 1

    # ----- Helpers: get window centers and their indices ----- #
    def _get_window_centers(self, days_of_year: np.ndarray) -> np.ndarray:
        """
        Returns an array of window-centers: integers representing a day of year (between 1 and 366) around which a window can be taken.

        Parameters
        ----------
        days_of_year: np.ndarray
            Array of consecutive days of year on which the running window is calculated/applied on.
        """

        min_day_of_year = np.min(days_of_year)
        max_day_of_year = np.max(days_of_year)

        days_left_after_last_step = (
            max_day_of_year - min_day_of_year + 1
        ) % self.window_step_length_in_days

        # Running window with step length fits perfectly into year
        if days_left_after_last_step == 0:
            first_window_center = 1 + self.window_step_length_in_days // 2
        # Adjust start-window-center so the last window is not too much sorter
        else:
            first_window_center = (
                min_day_of_year
                + self.window_step_length_in_days // 2
                - (self.window_step_length_in_days - days_left_after_last_step) // 2
            )

        window_centers = np.arange(
            first_window_center,
            max_day_of_year + 1,
            self.window_step_length_in_days,
        )

        return window_centers

    # ----- Main methods ----- #
    def get_indices_vals_in_window(
        self, days_of_year: np.ndarray, window_center: int
    ) -> np.ndarray:
        """
        Gets the indices inside of days_of_year of vals in a window of length window_length around a window center. Given a window center those are [window_center - self.window_length_in_years//2, window_center + self.window_length_in_years//2].

        Parameters
        ----------
        days_of_year : np.ndarray
            Array of days of years for which indices of window around window_center are returned.
        window_center: int
            Window center around which in each year a window of length self.window_length is taken and the indices returned
        """

        if window_center == 366:
            indices_center = np.where(days_of_year == 365)[0] + 1
        else:
            indices_center = np.where(days_of_year == window_center)[0]

        if indices_center.size == 0:
            return indices_center

        indices = np.sort(
            np.mod(
                np.concatenate(
                    [
                        np.arange(
                            i - self.window_length_in_days // 2,
                            i + self.window_length_in_days // 2 + 1,
                        )
                        for i in indices_center
                    ]
                ),
                days_of_year.size,
            )
        )
        return indices

    def get_indices_vals_to_adjust(
        self, days_of_year: np.ndarray, years: np.ndarray, window_center: int
    ) -> np.ndarray:
        """
        Gets the indices of which values inside a running window are to be adjusted and makes sure that the indices of the values to adjust values do not extend into a neighbouring year.
        For example if the window_center is 364 and the step size 5 then the day of year values to store the bc values would be [362, 363, 364,  and 365, 1 or 1, 2]. However since this would extend into the following year -- values already covered by another running window center/index -- days 1 and 2 are filtered out using years.

        Parameters
        ----------
        days_of_year : np.ndarray
            Array of days of years to find indices of bc values.
        years: np.ndarray:
            Array of years to make sure indices of bc values do not extend into neighbouring years.
        window_center: int
            Window center around which a window of length self.window_step_length_in_days is taken which stores the bc values.

        """
        if window_center == 366:
            indices_center = np.where(days_of_year == 365)[0] + 1
            years_indices_center = years[indices_center - 1]
        else:
            indices_center = np.where(days_of_year == window_center)[0]
            years_indices_center = years[indices_center]

        if indices_center.size == 0:
            return indices_center

        if np.unique(years).size == 1:
            # If timeseries spans only one year then we only circle through this year anyway and do not need to match running windows together
            indices = np.concatenate(
                [
                    np.arange(
                        i - self.window_step_length_in_days // 2,
                        i + self.window_step_length_in_days // 2 + 1,
                    )
                    for i in indices_center
                ]
            )
            indices = indices[(indices <= days_of_year.size) & (indices >= 0)]
        else:
            # If timeseries spans multiple year make sure that indices in window-center that get bc-values are only always in one year.
            years_indices_center = np.unique(years_indices_center)
            indices = np.array(
                [
                    index
                    for index_nr, index_window_center, in enumerate(indices_center)
                    # Compute indices in window-center to store bc values inside
                    for index in np.mod(
                        np.arange(
                            index_window_center - self.window_step_length_in_days // 2,
                            index_window_center
                            + self.window_step_length_in_days // 2
                            + 1,
                        ),
                        days_of_year.size,
                    )
                    # Make sure these indices are in the respective year
                    if years[index] == years_indices_center[index_nr]
                ]
            )

        return indices

    def use(self, days_of_year: np.ndarray, years: np.ndarray) -> np.ndarray:
        """
        This applies the running window onto an array of days of year, whilst respecting year bounds.
        It returns an iterator of (indices_vals_to_adjust, window_center) giving:
            indices_vals_to_adjust: the indices of values inside of days_of_year to adjust/modify inside the current window. Given a window center those are [window_center - self.window_step_length_in_years//2, window_center + self.window_step_length_in_years//2 + 1].
            window_center: the window center as day of year (between 1 and 366) to calculate the indices of the values inside the window for different sets of observations and climate model outputs using get_indices_vals_in_window(days_of_year, window_center).
        This return structure is chosen because usually only one array is adjusted (the future climate model run) but multiple sets of observations and climate model runs come in to calculate the adjusted values: historical observations and climate model run. So for each of those the indices in window given a window center are required and can be calculated given the window center.


        days_of_year and years are suppposed to be arrays of values corresponding to time ordered observations or climate model runs and giving for each the day of year and year respectively. So eg. for observations spanning the period 1980-2000 days_of_year will have values starting from 0 to 364 and repeating that for the number of years (adding 365 in case of a leap year). years will be similar.
        The array of years is required to repect year bounds inside of the values to adjust. For example if the window_center is 365 and the step size 5 then the day of year values to store the bc values would be [363, 364, 365, and 366, 1 or 1, 2]. However since this would extend into the following year -- values already covered by another running window center/index -- days 1 and 2 are filtered out using years.


        Usual usage:
        >>> from ibicus.utils import create_array_of_consecutive_dates, day_of_year, year
        >>> dates = create_array_of_consecutive_dates(1000)
        >>> days_of_year, years = day_of_year(dates), year(dates)
        >>> rolling_window = RunningWindowOverDaysOfYear(window_length_in_days = 31, window_step_length_in_days = 3)
        >>> for window_center, indices_vals_to_debias in rolling_window.use(days_of_year, years):
        ...     # do some calculations with both
        ...     print(indices_vals_to_debias)
        ...     indices_vals_in_window = rolling_window.get_indices_vals_in_window(days_of_year, window_center)
        ...     print(indices_vals_in_window)


        Parameters
        ----------
        days_of_year: np.ndarray
            Array of consecutive days of year on which the running window is calculated/applied on.
        years : np.ndarray
            Array of consecutive years on which the running window is calculated/applied on.
        """
        window_centers = self._get_window_centers(days_of_year)

        for window_center in window_centers:
            yield window_center, self.get_indices_vals_to_adjust(
                days_of_year, years, window_center
            )
