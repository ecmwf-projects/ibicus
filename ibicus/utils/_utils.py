# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Optional

import numpy as np
import pandas as pd


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


def threshold_cdf_vals(
    cdf_vals: np.ndarray, cdf_threshold: float = 1e-10
) -> np.ndarray:
    """Thresholds an array of cdf-values away from 0 and 1. Rounds down to `1-cdf_threshold` and up to `cdf_threshold` if above or below."""
    return np.maximum(np.minimum(cdf_vals, 1 - cdf_threshold), cdf_threshold)


def interp_sorted_cdf_vals_on_given_length(
    cdf_vals: np.ndarray, interpolation_length: int
) -> np.ndarray:
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
    except Exception:
        raise ValueError(
            "Your datetime object needs to implement a .day attribute. In doubt please use standard python datetime or cftime."
        )


_day = np.vectorize(_day)


def day(x):
    x = np.array(x)
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.astype("datetime64[D]").astype(object)
    return _day(x)


def _month(x):
    try:
        return x.month
    except Exception:
        raise ValueError(
            "Your datetime object needs to implement a .month attribute. In doubt please use standard python datetime or cftime."
        )


_month = np.vectorize(_month)


def month(x):
    x = np.array(x)
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.astype("datetime64[D]").astype(object)
    return _month(x)


def season(x):
    x = month(x)

    def month_to_season(x):
        if x in [3, 4, 5]:
            return "Spring"
        elif x in [6, 7, 8]:
            return "Summer"
        elif x in [9, 10, 11]:
            return "Autumn"
        elif x in [12, 1, 2]:
            return "Winter"
        else:
            return None

    month_to_season = np.vectorize(month_to_season)
    return month_to_season(x)


def _year(x):
    try:
        return x.year
    except Exception:
        raise ValueError(
            "Your datetime object needs to implement a .year attribute. In doubt please use standard python datetime or cftime."
        )


_year = np.vectorize(_year)


def year(x):
    x = np.array(x)
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.astype("datetime64[D]").astype(object)
    return _year(x)


def _day_of_year(x):
    try:
        if hasattr(x, "timetuple"):
            return x.timetuple().tm_yday
        else:
            first_date_in_year = type(x)(year(x), 1, 1)
            diff = x - first_date_in_year
            return diff.days + 1
    except Exception:
        raise ValueError(
            "Your datetime object needs to implement either the .timetuple-method or a timedelta and datetime constructor using the type-name. In doubt please use standard python datetime or cftime"
        )


_day_of_year = np.vectorize(_day_of_year)


def day_of_year(x):
    x = np.array(x)
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.astype("datetime64[D]").astype(object)
    return _day_of_year(x)


def create_array_of_consecutive_dates(
    array_length, start_date=np.datetime64("1950-01-01")
):
    if not isinstance(start_date, np.datetime64):
        start_date = np.datetime64(start_date)
    return np.arange(start_date, start_date + np.timedelta64(array_length, "D")).astype(
        object
    )


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


# ----- Variables ----- #


# ----- tas, tasmin, tasmax, tasrange and tasskew ----- #
def _get_tasmax_from_tasmin_and_range(tasrange, tasmin):
    return tasrange + tasmin


# Tasrange and skew
def get_tasrange(tasmin: np.ndarray, tasmax: np.ndarray) -> np.ndarray:
    """
    Calculates numpy array of :py:data:`tasrange` from arrays of :py:data:`tasmin` and :py:data:`tasmax`.

    All input arrays need to have values at the same timesteps and locations.

    Formulas:

    .. math:: \\text{tasrange} = \\text{tasmax} - \\text{tasmin}
    .. math:: \\text{tasskew} = \\frac{\\text{tas} - \\text{tasmin}}{\\text{tasrange}}

    Parameters
    ----------
    tasmin : np.ndarray
        Numpy array of :py:data:`tasmin`-values.
    tasmax : np.ndarray
        Numpy array of :py:data:`tasmax`-values.

    Returns
    -------
    tasrange : np.ndarray
        Numpy array of :py:data:`tasrange` values
    """
    return tasmax - tasmin


def get_tasskew(tas: np.ndarray, tasmin: np.ndarray, tasmax: np.ndarray) -> np.ndarray:
    """
    Calculates numpy array of :py:data:`tasskew` from arrays of :py:data:`tas`, :py:data:`tasmin` and :py:data:`tasmax`.

    All input arrays need to have values at the same timesteps and locations.

    Formulas:

    .. math:: \\text{tasrange} = \\text{tasmax} - \\text{tasmin}
    .. math:: \\text{tasskew} = \\frac{\\text{tas} - \\text{tasmin}}{\\text{tasrange}}

    Parameters
    ----------
    tas : np.ndarray
        Numpy array of :py:data:`tas`-values.
    tasmin : np.ndarray
        Numpy array of :py:data:`tasmin`-values.
    tasmax : np.ndarray
        Numpy array of :py:data:`tasmax`-values.

    Returns
    -------
    tasskew : np.ndarray
        Numpy array of :py:data:`tasskew` values
    """
    return (tas - tasmin) / (tasmax - tasmin)


# Tasmin and max
def get_tasmin(
    tas: np.ndarray, tasrange: np.ndarray, tasskew: np.ndarray
) -> np.ndarray:
    """
    Calculates numpy array of :py:data:`tasmin` from arrays of :py:data:`tas`, :py:data:`tasrange` and :py:data:`tasskew`.

    All input arrays need to have values at the same timesteps and locations.

    Formulas:

    .. math:: \\text{tasrange} = \\text{tasmax} - \\text{tasmin}
    .. math:: \\text{tasskew} = \\frac{\\text{tas} - \\text{tasmin}}{\\text{tasrange}}

    Parameters
    ----------
    tas : np.ndarray
        Numpy array of :py:data:`tas`-values.
    tasrange : np.ndarray
        Numpy array of :py:data:`tasrange`-values.
    tasskew : np.ndarray
        Numpy array of :py:data:`tasskew`-values.

    Returns
    -------
    tasmin : np.ndarray
        Numpy array of :py:data:`tasmin` values
    """
    return tas - tasskew * tasrange


def get_tasmax(
    tas: np.ndarray, tasrange: np.ndarray, tasskew: np.ndarray
) -> np.ndarray:
    """
    Calculates numpy array of :py:data:`tasmax` from arrays of :py:data:`tas`, :py:data:`tasrange` and :py:data:`tasskew`.

    All input arrays need to have values at the same timesteps and locations.

    Formulas:

    .. math:: \\text{tasrange} = \\text{tasmax} - \\text{tasmin}
    .. math:: \\text{tasskew} = \\frac{\\text{tas} - \\text{tasmin}}{\\text{tasrange}}

    Parameters
    ----------
    tas : np.ndarray
        Numpy array of :py:data:`tas`-values.
    tasrange : np.ndarray
        Numpy array of :py:data:`tasrange`-values.
    tasskew : np.ndarray
        Numpy array of :py:data:`tasskew`-values.

    Returns
    -------
    tasmax : np.ndarray
        Numpy array of :py:data:`tasmax` values
    """
    return _get_tasmax_from_tasmin_and_range(
        get_tasmin(tas, tasrange, tasskew), tasrange
    )


# Both
def get_tasmin_tasmax(
    tas: np.ndarray, tasrange: np.ndarray, tasskew: np.ndarray
) -> np.ndarray:
    """
    Calculates numpy arrays of both :py:data:`tasmin` and :py:data:`tasmax` from arrays of :py:data:`tas`, :py:data:`tasrange` and :py:data:`tasskew`.

    All input arrays need to have values at the same timesteps and locations.

    Formulas:

    .. math:: \\text{tasrange} = \\text{tasmax} - \\text{tasmin}
    .. math:: \\text{tasskew} = \\frac{\\text{tas} - \\text{tasmin}}{\\text{tasrange}}

    Parameters
    ----------
    tas : np.ndarray
        Numpy array of :py:data:`tas`-values.
    tasrange : np.ndarray
        Numpy array of :py:data:`tasrange`-values.
    tasskew : np.ndarray
        Numpy array of :py:data:`tasskew`-values.

    Returns
    -------
    tasmin : np.ndarray
        Numpy array of :py:data:`tasmin` values
    tasmax : np.ndarray
        Numpy array of :py:data:`tasmax` values
    """
    tasmin = get_tasmin(tas, tasrange, tasskew)
    tasmax = _get_tasmax_from_tasmin_and_range(tasmin, tasrange)
    return tasmin, tasmax


def get_tasrange_tasskew(
    tas: np.ndarray, tasmin: np.ndarray, tasmax: np.ndarray
) -> np.ndarray:
    """
    Calculates numpy arrays of both :py:data:`tasrange` and :py:data:`tasskew` from arrays of :py:data:`tas`, :py:data:`tasmin` and :py:data:`tasmax`.

    All input arrays need to have values at the same timesteps and locations.

    Formulas:

    .. math:: \\text{tasrange} = \\text{tasmax} - \\text{tasmin}
    .. math:: \\text{tasskew} = \\frac{\\text{tas} - \\text{tasmin}}{\\text{tasrange}}

    Parameters
    ----------
    tas : np.ndarray
        Numpy array of :py:data:`tas`-values.
    tasmin : np.ndarray
        Numpy array of :py:data:`tasmin`-values.
    tasmax : np.ndarray
        Numpy array of :py:data:`tasmax`-values.

    Returns
    -------
    tasrange : np.ndarray
        Numpy array of :py:data:`tasrange` values
    tasskew : np.ndarray
        Numpy array of :py:data:`tasskew` values
    """
    tasrange = get_tasrange(tasmin, tasmax)
    tasskew = get_tasskew(tas, tasmin, tasmax)
    return tasrange, tasskew


# ----- pr, prsn and prsnratio ----- #


def get_prsnratio(pr: np.ndarray, prsn: np.ndarray) -> np.ndarray:
    """
    Calculates numpy array of :py:data:`prsnratio` from arrays of :py:data:`pr` and :py:data:`prsn`.

    All input arrays need to have values at the same timesteps and locations.

    Formula:

    .. math:: \\text{prsnratio} = \\frac{\\text{prsn}}{\\text{pr}}

    Parameters
    ----------
    pr : np.ndarray
        Numpy array of :py:data:`pr`-values.
    prsn : np.ndarray
        Numpy array of :py:data:`prsn`-values.

    Returns
    -------
    prsnration : np.ndarray
        Numpy array of :py:data:`prsnratio` values
    """
    return prsn / pr


def get_pr(prsn: np.ndarray, prsnratio: np.ndarray) -> np.ndarray:
    """
    Calculates numpy array of :py:data:`pr` from arrays of :py:data:`prsn` and :py:data:`prsnratio`.

    All input arrays need to have values at the same timesteps and locations.

    Formula:

    .. math:: \\text{prsnratio} = \\frac{\\text{prsn}}{\\text{pr}}

    Parameters
    ----------
    pr : np.ndarray
        Numpy array of :py:data:`prsn`-values.
    prsn : np.ndarray
        Numpy array of :py:data:`prsnratio`-values.

    Returns
    -------
    pr : np.ndarray
        Numpy array of :py:data:`pr` values
    """
    return prsn / prsnratio


def get_prsn(pr: np.ndarray, prsnratio: np.ndarray) -> np.ndarray:
    """
    Calculates numpy array of :py:data:`prsn` from arrays of :py:data:`pr` and :py:data:`prsnratio`.

    All input arrays need to have values at the same timesteps and locations.

    Formula:

    .. math:: \\text{prsnratio} = \\frac{\\text{prsn}}{\\text{pr}}

    Parameters
    ----------
    pr : np.ndarray
        Numpy array of :py:data:`pr`-values.
    prsn : np.ndarray
        Numpy array of :py:data:`prsnratio`-values.

    Returns
    -------
    prsn : np.ndarray
        Numpy array of :py:data:`prsn` values
    """
    return prsnratio * pr


def _unpack_df_of_numpy_arrays(df, numpy_column_name):
    new_expanded_rows = []
    for _, row in df.iterrows():
        expanded_row = {}
        for index, value in row.iteritems():
            if index == numpy_column_name:
                expanded_row[index] = value[0].flatten()
            else:
                expanded_row[index] = value
        new_expanded_rows.append(pd.DataFrame(data=expanded_row))

    expanded_df = pd.concat(new_expanded_rows)
    expanded_df[numpy_column_name] = pd.to_numeric(expanded_df[numpy_column_name])

    return expanded_df


def get_mask_for_unique_subarray(x):
    _, indices = np.unique(x, return_index=True)
    mask = np.zeros_like(x).astype(bool)
    mask[indices] = True
    return mask


def _check_if_list_of_two_and_unpack_else_none(x):
    if isinstance(x, (list, tuple)):
        if len(x) > 2:
            raise ValueError("Error in input. Needs to be a list of two.")
        return x[0], x[1]
    else:
        return x, None


# ----- Logging functionality -----


def _get_library_name():
    return __name__.split(".")[0]


def get_library_logger():
    """
    Returns the library logger used by the ibicus package.
    """
    return logging.getLogger(_get_library_name())


def get_verbosity_library_logger():
    """
    Returns the verbosity/level for the library logger as ``int``.
    """
    return get_library_logger().getEffectiveLevel()


def set_verbosity_library_logger(verbosity):
    """
    Sets the verbosity/level for the library logger.

    Parameters
    ----------
    verbosity :
        Logging level: ``["logging.INFO", logging.WARNING, "logging.ERROR", ...]``.
    """
    get_library_logger().setLevel(verbosity)
