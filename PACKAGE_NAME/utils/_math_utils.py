# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""math_helpers module - helpers used by different debiasers"""

from typing import Union

import numpy as np
import scipy.optimize
import scipy.stats
import statsmodels.distributions.empirical_distribution
from scipy.stats import gamma

"""----- Precipitation helpers -----"""
# TODO: in gamma.fit shall we specify gamma.fit(floc = 0) so keep loc fixed at zero?


def fit_precipitation_hurdle_model(
    data: np.ndarray, distribution: scipy.stats.rv_continuous = scipy.stats.gamma
) -> tuple:
    """
    Fits a precipitation hurdle model and returns parameter estimates.

    A hurdle-model is a two-step process: binomially it is determined if it rains (with probability p0 of no rain) and
        then we assume that theamounts follow a given distribution (often gamma) described by a cdf F_A. Mathematically:

    P(X = 0) = p0,
    P(0 < X <= x) = p0 + (1-p0) F_A(x)


    Parameters
    ----------
    data : np.ndarray
        Array containing precipitation values.
    distribution : scipy.stats.rv_continuous
        Distribution assumed for the precipitation amounts.

    Returns
    -------
    tuple
        Tuple containing parameter estimates: (p0, tuple of parameter estimates for the amounts-distribution).
    """
    rainy_days = data[data != 0]

    p0 = 1 - rainy_days.shape[0] / data.shape[0]
    fit_rainy_days = distribution.fit(rainy_days)

    return (p0, fit_rainy_days)


def cdf_precipitation_hurdle_model(
    x: np.ndarray, fit: tuple, distribution: scipy.stats.rv_continuous = scipy.stats.gamma, randomization: bool = False
) -> np.ndarray:
    """
    Returns cdf-values of a vector x for the cdf of a precipitation hurdle-model. If randomization = True then cdf-values for x == 0
        (no rain) are randomized between (0, p0).

    Parameters
    ----------
    x : np.ndarray
        Values for which the cdf shall be evaluated.
    fit : tuple
        Parameter controling the hurdle model: (p0, tuple of parameter estimates for the amounts-distribution).
        Return value of fit_precipitation_hurdle_model.
    distribution : scipy.stats.rv_continuous
        Distribution assumed for the precipitation amounts.
    randomization : bool
        Whether cdf-values for x == 0 (no rain) shall be randomized uniformly within (0, p0).
        Helps for quantile mapping and controlling the zero-inflation.


    Returns
    -------
    np.ndarray
        Array containing cdf-values for x.
    """

    p0 = fit[0]
    fit_rainy_days = fit[1]

    if not randomization:
        return np.where(x == 0, p0, p0 + (1 - p0) * distribution.cdf(x, *fit_rainy_days))
    else:
        return np.where(
            x == 0,
            np.random.uniform(0, p0),
            p0 + (1 - p0) * distribution.cdf(x, *fit_rainy_days),
        )


def ppf_precipitation_hurdle_model(
    q: np.ndarray, fit: tuple, distribution: scipy.stats.rv_continuous = scipy.stats.gamma
):
    """
    Returns ppf (quantile / inverse cdf)-values of a vector x for the cdf of a precipitation hurdle-model.

    Parameters
    ----------
    q : np.ndarray
        Values for which the ppf shall be evaluated.
    fit : tuple
        Parameter controling the hurdle model: (p0, tuple of parameter estimates for the amounts-distribution).
        Return value of fit_precipitation_hurdle_model.
    distribution : scipy.stats.rv_continuous
        Distribution assumed for the precipitation amounts.

    Returns
    -------
    np.ndarray
        Array containing cdf-values for x.
    """
    p0 = fit[0]
    fit_rainy_days = fit[1]

    return np.where(q > p0, distribution.ppf((q - p0) / (1 - p0), *fit_rainy_days), 0)


def quantile_mapping_precipitation_hurdle_model(
    x,
    fit_right,
    fit_left,
    distribution_right=scipy.stats.gamma,
    distribution_left=scipy.stats.gamma,
    randomization=False,
):
    """
    Applies quantile mapping between two precipitation hurdle-models (see fit_precipitation_hurdle_model).

    The values x are first mapped onto quantiles q in [0, 1] using the cdf of fit_right and then
        to new values using the ppf/inverse cdf of fit_left. If randomization = True then on the right application
        of the cdf the cdf-values for zero-precipitation values are randomized between (0, p0). See `cdf_precipitation_hurdle_model`

    Parameters
    ----------
    x : np.ndarray
        Values for which the cdf shall be evaluated.
    fit_right : tuple
        Parameters controling the hurdle model on the right side of the quantile mapping:
            (p0, tuple of parameter estimates for the amounts-distribution).
        Return value of fit_precipitation_hurdle_model.
    fit_left : tuple
        Parameters controling the hurdle model on the left side of the quantile mapping:
            (p0, tuple of parameter estimates for the amounts-distribution).
        Return value of fit_precipitation_hurdle_model.
    distribution_right : scipy.stats.rv_continuous
        Distribution assumed for the precipitation amounts on the right side of the quantile mapping.
    distribution_left : scipy.stats.rv_continuous
        Distribution assumed for the precipitation amounts on the left side of the quantile mapping.
    randomization : bool
        Wether cdf-values for x == 0 (no rain) shall be randomized uniformly within (0, p0).
        Helps for controlling the zero-inflation during quantile mapping.


    Returns
    -------
    np.ndarray
        Array containing the quantile mapped values for x.
    """
    q = cdf_precipitation_hurdle_model(x, fit=fit_right, distribution=distribution_right, randomization=randomization)
    x_mapped = ppf_precipitation_hurdle_model(q, fit=fit_left, distribution=distribution_left)
    return x_mapped


# Censored model: rain values of zero (or below censoring_value) are treated as censored values
def fit_censored_gamma(x: np.ndarray, nr_censored_x: int, min_x: float) -> tuple:
    """
    Fits a censored gamma distribution to left-censored data with `nr_censored_x` censored observations.

    Parameters
    ----------
    x : np.ndarray
        Non-censored values on which to fit the censored gamma.
    nr_censored_x : tuple
        Number of censored observations in the dataset on which to fit the censored gamma.
    min_x : tuple
        Censoring value.

    Returns
    -------
    tuple
        Parameter estimates for the gamma distribution.
    """

    def neg_log_likelihood(params, x, nr_censored_x, min_x) -> float:
        return -np.sum(gamma.logpdf(x, a=params[0], scale=params[1])) - nr_censored_x * np.log(
            gamma.cdf(min_x, a=params[0], scale=params[1])
        )

    optimizer_result = scipy.optimize.minimize(
        neg_log_likelihood,
        x0=np.array([1, 1]),
        args=(x, nr_censored_x, min_x),
        options={"maxiter": 1000, "disp": False},
        method="nelder-mead",
        tol=1e-8,
    )
    return (
        optimizer_result.x[0],
        0,
        optimizer_result.x[1],
    )  # location was fixed to zero


def fit_precipitation_censored_gamma(data, censoring_value):
    """
    Fits a censored gamma distribution to precipitation data where everything under `censored_value` is assumed to be
        a censored observation. This is useful when a slightly higher threshold is used to account for the drizzle effect in climate models.

    Parameters
    ----------
    data : np.ndarray
        Data on which to fit the censored gamma distribution.
    censoring_value : tuple
        Value under which observations are assumed to be censored

    Returns
    -------
    tuple
        Parameter estimates for the gamma distribution.
    """
    noncensored_data = data[data > censoring_value]
    return fit_censored_gamma(noncensored_data, data.shape[0] - noncensored_data.shape[0], censoring_value)


def quantile_mapping_precipitation_censored_gamma(
    x: np.ndarray, censoring_value: float, fit_right: tuple, fit_left: tuple
) -> np.ndarray:
    """
    Applies quantile mapping between two precipitation censored gamma models.
    The values x are first mapped onto quantiles q in [0, 1] using the cdf of fit_right and then
        to new values using the ppf/inverse cdf of fit_left. Values under censoring_value are first
        randomized on the right and then thresholded and set to zero again on the left

    Parameters
    ----------
    x : np.ndarray
        Values for which the cdf shall be evaluated.
    censoring_value : tuple
        Value under which observations are assumed to be censored
    fit_right : tuple
        Parameters controling the censored model on the right side of the quantile mapping.
        Return value of fit_precipitation_censored_gamma.
    fit_left : tuple
        Parameters controling the censored model on the left side of the quantile mapping
        Return value of fit_precipitation_censored_gamma.

    Returns
    -------
    np.ndarray
        Array containing the quantile mapped values for x.
    """
    x_randomized = np.where(x < censoring_value, np.random.uniform(0, censoring_value), x)
    q = gamma.cdf(x_randomized, *fit_right)
    x_mapped = gamma.ppf(q, *fit_left)
    return np.where(x_mapped < censoring_value, 0, x_mapped)


"""----- Other helpers -----"""


def IECDF(x):
    """
    Get the inverse empirical cdf of an array of data:

    x = np.random.random(1000)
    iecdf = IECDF(x)
    iecdf(0.2)

    Up to numerical accuracy this returns the same as np.quantile(x, q, method = "inverted_cdf") but is much faster.

    Parameters
    ----------
    x : array
        Array containing values for which the empirical cdf shall be calculated.

    Returns
    -------
    lambda
        Function to calculate the inverse empirical cdf-value for a given quantile q.
    """
    y = np.sort(x)
    n = y.shape[0]
    return lambda q: y[np.floor((n - 1) * q).astype(int)]


# TODO: this implementation is much faster, because np.quantile(x,p,method = "inverted_cdf") takes a bit.
# However it is slightly akward.
def iecdf(x, p, method="inverted_cdf", **kwargs):
    """
    Return the values of the the inverse empirical cdf of x evaluated at p:

    x = np.random.random(1000)
    p = np.linspace(0, 1, 100)
    iecdf(x, p)

    The call is delegated to np.quantile with the method-argument determining whether eg. interpolation is used.

    Parameters
    ----------
    x : array
        Array containing values with which the inverse empirical cdf is defined.
    p : array
        Array containing values between [0, 1] for which the inverse empirical cdf is evaluated.
    method : string
        Method string for np.quantile
    **kwargs

    Returns
    -------
    array
        Values of the inverse empirical cdf of x evaluated at p.
    """
    if method == "inverted_cdf":
        # This method is much faster actually than using np.quantile.
        # Although it is slightly akward for the sake of efficiency we refer to IECDF.
        iecdf = IECDF(x)
        return iecdf(p)
    else:
        return np.quantile(x, p, method=method, **kwargs)


def ecdf(x: np.array, y: np.array, method: str) -> np.array:
    """
    Return the values of the empirical cdf of x evaluated at y:

    x = np.random.random(1000)
    y = np.random.random(100)
    ecdf(x, y)

    Three methods exist determined by method.
        - method = "kernel_density": A kernel density estimate of the ecdf is used, using scipy.stat.rv_histogram.
        - method = "linear_interpolation": Linear interpolation is used, starting from a grid of cdf-values.
        - method = "step_function": The classical step-function.

    Parameters
    ----------
    x : array
        Array containing values with which the empirical cdf is defined.
    y : array
        Array containing values on which the empirical cdf is evaluated.
    method : string
        Method with which the ecdf is calculated. One of ["kernel_density", "linear_interpolation", "step_function"].

    Returns
    -------
    array
        Values of the empirical cdf of x evaluated at y.
    """
    if method == "kernel_density":
        rv_histogram_fit = scipy.stats.rv_histogram(np.histogram(x, bins="auto"))
        return rv_histogram_fit.cdf(y)
    elif method == "linear_interpolation":
        p_grid = np.linspace(0.0, 1.0, x.size)
        q_vals_for_p_grid = np.quantile(x, p_grid)
        return np.interp(x, q_vals_for_p_grid, p_grid)
    elif method == "step_function":
        step_function = statsmodels.distributions.empirical_distribution.ECDF(x)
        return step_function(y)
    else:
        raise ValueError('method needs to be one of ["kernel_density", "linear_interpolation", "step_function"] ')
