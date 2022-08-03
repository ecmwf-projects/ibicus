# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional

import numpy as np


def get_chunked_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Gets chunked means of an array. Splits an array in chunks of length n, calculates the mean in each chunk and returns an array of these values"""
    ids = np.arange(x.size) // n
    return np.bincount(ids, x) / np.bincount(ids)


def get_yearly_means(x: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Gets an array of yearly means of a timeseries x where each value has a corresponding year in years."""
    return np.array([np.mean(x[years == i_year]) for i_year in np.unique(years)])


def get_years_and_yearly_means(x: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Gets an array of unique years and one of yearly means of a timeseries x where each value has a corresponding year in years."""
    return np.unique(years), get_yearly_means(x, years)


def threshold_cdf_vals(cdf_vals: np.ndarray, cdf_threshold: float = 1e-10) -> np.ndarray:
    """Thresholds an array of cdf-values away from 0 and 1. Rounds down to `1-cdf_threshold` and up to `cdf_threshold` if above or below."""
    return np.maximum(np.minimum(cdf_vals, 1 - cdf_threshold), cdf_threshold)


def interp_sorted_cdf_vals_on_given_length(cdf_vals: np.ndarray, interpolation_length: int) -> np.ndarray:
    """Interpolates an array sorted cdf values onto a given length by linear interpolation."""
    return np.interp(
        np.linspace(1, cdf_vals.size, interpolation_length),
        np.linspace(1, cdf_vals.size, cdf_vals.size),
        cdf_vals,
    )


def sort_array_like_another_one(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gets an array x into the order of another one y, meaning that the biggest value of x is at the position of the biggest value of y, etc."""
    inverse_sort_y = np.argsort(np.argsort(y))
    return np.sort(x)[inverse_sort_y]


# ----- Datetime functionality -----


def _day(x):
    try:
        return x.day
    except:
        raise ValueError(
            "Your datetime object needs to implement a .day attribute. In doubt please use standard python datetime or cftime"
        )


_day = np.vectorize(_day)


def day(x):
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.astype(object)
    return _day(x)


def _month(x):
    try:
        return x.month
    except:
        raise ValueError(
            "Your datetime object needs to implement a .month attribute. In doubt please use standard python datetime or cftime"
        )


_month = np.vectorize(_month)


def month(x):
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.astype(object)
    return _month(x)


def _year(x):
    try:
        return x.year
    except:
        raise ValueError(
            "Your datetime object needs to implement a .year attribute. In doubt please use standard python datetime or cftime"
        )


_year = np.vectorize(_year)


def year(x):
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.astype(object)
    return _year(x)


def _day_of_year(x):
    try:
        if hasattr(x, "timetuple"):
            return x.timetuple().tm_yday
        else:
            first_date_in_year = type(x)(year(x), 1, 1)
            diff = x - first_date_in_year
            return diff.days + 1
    except:
        raise ValueError(
            "Your datetime object needs to implement either the .timetuple-method or a timedelta and datetime constructor using the type-name. In doubt please use standard python datetime or cftime"
        )


_day_of_year = np.vectorize(_day_of_year)


def day_of_year(x):
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.astype(object)
    return _day_of_year(x)


def create_array_of_consecutive_dates(array_length, start_date=np.datetime64("1950-01-01")):
    if not isinstance(start_date, np.datetime64):
        start_date = np.datetime64(start_date)
    return np.arange(start_date, start_date + np.timedelta64(array_length, "D")).astype(object)


def infer_and_create_time_arrays_if_not_given(
    obs: np.ndarray,
    cm_hist: np.ndarray,
    cm_future: np.ndarray,
    time_obs: Optional[np.ndarray] = None,
    time_cm_hist: Optional[np.ndarray] = None,
    time_cm_future: Optional[np.ndarray] = None,
):
    if time_obs is None:
        time_obs = create_array_of_consecutive_dates(obs.size)
    if time_cm_hist is None:
        time_cm_hist = create_array_of_consecutive_dates(cm_hist.size)
    if time_cm_future is None:
        time_cm_future = create_array_of_consecutive_dates(cm_future.size)

    return time_obs, time_cm_hist, time_cm_future
