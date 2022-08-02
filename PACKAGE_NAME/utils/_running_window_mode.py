# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from logging import warning

import attrs
import numpy as np


@attrs.define
class RunningWindowModeOverYears:
    """
    Implements a running window mode iterating over years.

    Usual usage:
    >>> years = np.arange(2000, 2050)
    >>> rolling_window = RunningWindowModeOverYears(window_length_in_years = 17, window_step_length_in_years = 9)
    >>> for years_to_debias, years_in_window in rolling_window.use(years):
    ...     # do some calculations with both
    ...     print(years_to_debias)
    ...     print(years_in_window)

    Warning: currently only uneven sizes are allowed for window_length_in_years and window_step_length_in_years. This allows symmetrical windows of the form [window_center - self.window_step_length_in_years//2, window_center + self.window_step_length_in_years//2] for the years to adjust and similar for the years in window.

    Attributes
    ----------
    window_length_in_years: int
        Length of the running window in years: how many values are used to in the calculations later.
    window_step_length_in_years: int
        Step length of the running window in years: how many values are adjusted inside the running window.
    """

    window_length_in_years: int = attrs.field(
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)], converter=round
    )
    window_step_length_in_years: int = attrs.field(
        validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)], converter=round
    )

    def __attrs_post_init__(self):
        if self.window_length_in_years % 2 == 0:
            warning(
                "Currently only uneven window lengths are allowed for window_length_in_years. Automatically increased by 1."
            )
            self.window_length_in_years = self.window_length_in_years + 1
        if self.window_step_length_in_years % 2 == 0:
            warning(
                "Currently only uneven step lengths are allowed for window_step_length_in_years. Automatically increased by 1."
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

        if (years_left_after_last_step := number_of_years % self.window_step_length_in_years) == 0:
            first_window_center = unique_years.min() + self.window_step_length_in_years // 2
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
        indices = np.arange(
            window_center - self.window_step_length_in_years // 2,
            window_center + self.window_step_length_in_years // 2 + 1,
        )
        return indices

    # ----- Main methods ----- #
    @staticmethod
    def get_if_in_chosen_years(years, chosen_years):
        """
        Given an array of years this returns an array of bools indicating whether a year in years is inside of chosen_years.

        >>> chosen_years = np.array([2022, 2023])
        >>> years = np.arange(2020, 2030)
        >>> RunningWindowModeOverYears.get_if_in_chosen_years(years, chosen_years)
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
        >>> rolling_window = RunningWindowModeOverYears(window_length_in_years = 17, window_step_length_in_years = 9)
        >>> for years_to_debias, years_in_window in rolling_window.use(years):
        ...     # do some calculations with both
        ...     print(years_to_debias)
        ...     print(years_in_window)

        Parameters
        ----------
        years : np.ndarray
            Array of consecutive years on which the running window is calculated/applied on.
        """
        window_centers = self._get_years_forming_window_centers(np.unique(years))
        for window_center in window_centers:
            years_to_adjust = self._get_years_in_window_that_are_adjusted(window_center)
            years_in_window = self._get_years_in_window(window_center)
            yield years_to_adjust, years_in_window
