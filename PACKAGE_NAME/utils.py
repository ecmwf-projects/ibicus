import numpy as np


def get_chunked_mean(x, n):
    ids = np.arange(x.size) // n
    return np.bincount(ids, x) / np.bincount(ids)


def threshold_cdf_vals(cdf_vals: np.ndarray, cdf_threshold: int = 0.0001):
    return np.maximum(np.minimum(cdf_vals, 1 - cdf_threshold), cdf_threshold)


def interp_sorted_cdf_vals_on_given_length(
    cdf_vals: np.ndarray, interpolation_length: int
):
    return np.interp(
        np.linspace(1, cdf_vals.size, interpolation_length),
        np.linspace(1, cdf_vals.size, cdf_vals.size),
        cdf_vals,
    )
