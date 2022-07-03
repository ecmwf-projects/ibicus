# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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


def day(x):
    try:
        return x.day
    except:
        raise ValueError(
            "Your datetime object needs to implement a .day attribute. In doubt please use standard python datetime or cftime"
        )


day = np.vectorize(day)


def month(x):
    try:
        return x.month
    except:
        raise ValueError(
            "Your datetime object needs to implement a .month attribute. In doubt please use standard python datetime or cftime"
        )


month = np.vectorize(month)


def year(x):
    try:
        return x.year
    except:
        raise ValueError(
            "Your datetime object needs to implement a .year attribute. In doubt please use standard python datetime or cftime"
        )


year = np.vectorize(year)

import datetime


def day_of_year(x):
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


day_of_year = np.vectorize(day_of_year)
