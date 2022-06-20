from multiprocessing.sharedctypes import Value

import numpy as np
import scipy.optimize
import scipy.stats
import statsmodels.distributions.empirical_distribution
from matplotlib.pyplot import step
from scipy.stats import gamma

"""----- Precipitation helpers -----"""
# TODO: in gamma.fit shall we specify gamma.fit(floc = 0) so keep loc fixed at zero?


# Hurdle model: two step process
# Binomial if it rains and then amounts how much. P(X = 0) = p0, P(0 < X <= x) = p0 + (1-p0) F_A(x)
def fit_precipitation_hurdle_model(data, distribution=scipy.stats.gamma):
    rainy_days = data[data != 0]

    p0 = 1 - rainy_days.shape[0] / data.shape[0]
    fit_rainy_days = distribution.fit(rainy_days)

    return (p0, fit_rainy_days)


def cdf_precipitation_hurdle_model(
    x, fit, distribution=scipy.stats.gamma, randomization=False
):
    p0 = fit[0]
    fit_rainy_days = fit[1]

    if not randomization:
        return np.where(
            x == 0, p0, p0 + (1 - p0) * distribution.cdf(x, *fit_rainy_days)
        )
    else:
        return np.where(
            x == 0,
            np.random.uniform(0, p0),
            p0 + (1 - p0) * distribution.cdf(x, *fit_rainy_days),
        )


def ppf_precipitation_hurdle_model(q, fit, distribution=scipy.stats.gamma):
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
    q = cdf_precipitation_hurdle_model(
        x, fit=fit_right, distribution=distribution_right, randomization=randomization
    )
    x_mapped = ppf_precipitation_hurdle_model(
        q, fit=fit_left, distribution=distribution_left
    )
    return x_mapped


# Censored model: rain values of zero (or below censoring_value) are treated as censored values
def fit_censored_gamma(x, nr_censored_x, min_x):
    def neg_log_likelihood(params, x, nr_censored_x, min_x) -> float:
        return -np.sum(
            gamma.logpdf(x, a=params[0], scale=params[1])
        ) - nr_censored_x * np.log(gamma.cdf(min_x, a=params[0], scale=params[1]))

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
    noncensored_data = data[data > censoring_value]
    return fit_censored_gamma(
        noncensored_data, data.shape[0] - noncensored_data.shape[0], censoring_value
    )


def quantile_mapping_precipitation_censored_gamma(
    x, censoring_value, fit_right, fit_left
):
    x_randomized = np.where(
        x < censoring_value, np.random.uniform(0, censoring_value), x
    )
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


# TODO: this implementation is much faster, because np.quantile(x,p,method = "inverted_cdf") takes a bit. However it is slightly akward.
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

    Three methods exist determined by method. Either a kernel density estimate of the ecdf is used, using scipy.stat.rv_histogram (method = "kernel_density"). Alternatively linear interpolation is possible, starting from a grid of cdf-values (method = "linear_interpolation"). And finally the classical step-function is possible (method = "step_function").

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
        raise ValueError(
            'method needs to be one of ["kernel_density", "linear_interpolation", "step_function"] '
        )
