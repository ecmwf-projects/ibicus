import numpy as np


def get_chunked_mean(x, n):
    ids = np.arange(x.size) // n
    return np.bincount(ids, x) / np.bincount(ids)


def get_yearly_mean(x, years):
    return np.array([np.mean(x[years == i_year]) for i_year in np.unique(years)])


def threshold_cdf_vals(cdf_vals: np.ndarray, cdf_threshold: int = 0.0001):
    return np.maximum(np.minimum(cdf_vals, 1 - cdf_threshold), cdf_threshold)


def interp_sorted_cdf_vals_on_given_length(cdf_vals: np.ndarray, interpolation_length: int):
    return np.interp(
        np.linspace(1, cdf_vals.size, interpolation_length),
        np.linspace(1, cdf_vals.size, cdf_vals.size),
        cdf_vals,
    )


def sort_array_like_another_one(x, y):
    """Sorts an array x like another one y, meaning that the biggest value of x is at the position of the biggest value of y, etc."""
    inverse_sort_y = np.argsort(np.argsort(y))
    return np.sort(x)[inverse_sort_y]


# ----- Datetime functionality -----


def day(x):
    return x.day


day = np.vectorize(day)


def month(x):
    return x.month


month = np.vectorize(month)


def year(x):
    return x.year


year = np.vectorize(year)
