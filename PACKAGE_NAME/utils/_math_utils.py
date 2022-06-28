# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""math_helpers module - helpers used by different debiasers"""

from abc import ABC, abstractmethod
from typing import Union

import attrs
import numpy as np
import scipy.optimize
import scipy.stats
import statsmodels.distributions.empirical_distribution
from scipy.stats import gamma


class StatisticalModel(ABC):

    """
    Abstract functionality to wrap an arbitrary statistical model given by a fit-method, a cdf and a ppf.
    This can be used to pass a self-defined model that is fitted in a debiaser to each location for eg. quantile-mapping.
    In principle this is similar to scipy.stats.rv_continuous, however the user has the option to provide an own fit-method. Thus this is able to represent a broader class of statistical models
    """

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> tuple:
        """
        Fits a statistical model and returns parameter estimates.

        Parameters
        ----------
        data : np.ndarray
            Array containing values on which the model is to fit.
        Returns
        -------
        tuple
            Tuple containing parameter estimates.
        """
        pass

    @abstractmethod
    def cdf(self, x: np.ndarray, fit: tuple, **kwargs) -> np.ndarray:
        """
        Returns cdf-values of a vector x for the cdf of a statistical model.

        Parameters
        ----------
        x : np.ndarray
            Values for which the cdf shall be evaluated.
        fit : tuple
            Parameters controling the model fit. Return value of fit.

        Returns
        -------
        np.ndarray
            Array containing cdf-values for x.
        """
        pass

    @abstractmethod
    def ppf(self, q: np.ndarray, fit: tuple, **kwargs) -> np.ndarray:
        """
        Returns ppf (quantile / inverse cdf)-values of a vector q for the cdf of a statistical model.

        Parameters
        ----------
        q : np.ndarray
            Values for which the ppf shall be evaluated.
        fit : tuple
            Parameters controling the model fit. Return value of fit.

        Returns
        -------
        np.ndarray
            Array containing ppf-values for x.
        """
        pass


"""----- Precipitation helpers -----"""


@attrs.define
class gen_PrecipitationHurdleModel(StatisticalModel):
    """
    Represents a precipitation hurdle model.

    A hurdle-model is a two-step process: binomially it is determined if it rains (with probability p0 of no rain) and
        then we assume that theamounts follow a given distribution (often gamma) described by a cdf F_A. Mathematically:

    P(X = 0) = p0,
    P(0 < X <= x) = p0 + (1-p0) F_A(x)

    Parameters
    ----------
    self.distribution : scipy.stats.rv_continuous
        Distribution assumed for the precipitation amounts.
    self.randomization : bool
        Whether cdf-values for x == 0 (no rain) shall be randomized uniformly within (0, p0).
        Helps for quantile mapping and controlling the zero-inflation.

    """

    distribution: scipy.stats.rv_continuous = attrs.field(
        default=scipy.stats.gamma, validator=attrs.validators.instance_of(scipy.stats.rv_continuous)
    )
    cdf_randomization: bool = attrs.field(default=True, validator=attrs.validators.instance_of(bool))

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fits a precipitation hurdle model and returns parameter estimates.


        Parameters
        ----------
        data : np.ndarray
            Array containing precipitation values.

        Returns
        -------
        tuple
            Tuple containing parameter estimates: (p0, tuple of parameter estimates for the amounts-distribution).
        """
        rainy_days = data[data != 0]

        p0 = 1 - rainy_days.shape[0] / data.shape[0]
        fit_rainy_days = self.distribution.fit(rainy_days)

        return (p0, fit_rainy_days)

    def cdf(self, x: np.ndarray, fit: tuple) -> np.ndarray:
        """
        Returns cdf-values of a vector x for the cdf of a precipitation hurdle-model. If self.cdf_randomization = True then cdf-values for x == 0
            (no rain) are randomized between (0, p0).

        Parameters
        ----------
        x : np.ndarray
            Values for which the cdf shall be evaluated.
        fit : tuple
            Parameter controling the hurdle model: (p0, tuple of parameter estimates for the amounts-distribution).
            Return value of fit.

        Returns
        -------
        np.ndarray
            Array containing cdf-values for x.
        """

        p0 = fit[0]
        fit_rainy_days = fit[1]

        if not self.randomization:
            return np.where(x == 0, p0, p0 + (1 - p0) * self.distribution.cdf(x, *fit_rainy_days))
        else:
            return np.where(
                x == 0,
                np.random.uniform(0, p0),
                p0 + (1 - p0) * self.distribution.cdf(x, *fit_rainy_days),
            )

    def ppf(self, q: np.ndarray, fit: tuple) -> np.ndarray:
        """
        Returns ppf (quantile / inverse cdf)-values of a vector q for the cdf of a precipitation hurdle-model.

        Parameters
        ----------
        q : np.ndarray
            Values for which the ppf shall be evaluated.
        fit : tuple
            Parameter controling the hurdle model: (p0, tuple of parameter estimates for the amounts-distribution).
            Return value of fit.

        Returns
        -------
        np.ndarray
            Array containing cdf-values for x.
        """
        p0 = fit[0]
        fit_rainy_days = fit[1]

        return np.where(q > p0, self.distribution.ppf((q - p0) / (1 - p0), *fit_rainy_days), 0)


PrecipitationHurdleModelGamma = gen_PrecipitationHurdleModel()
PrecipitationHurdleModelGammaWithoutCDFRandomization = gen_PrecipitationHurdleModel(cdf_randomization=False)


@attrs.define
class gen_PrecipitationGammaLeftCensoredModel(StatisticalModel):
    """
    Represents a left censored precipitation gamma model.
    A left censored gamma model is a gamma distribution where all values under a given threshold are censored: not observed. Those are represented by zero
    This is useful when a slightly higher threshold is used to account for the drizzle effect in climate models.

    In the cdf before calculating all values below the censoring value are first randomized between (0, censoring_value). In the ppf values below
        the censoring_value are again set to zero. This handles possible inflation in quantile mapping by the censoring-value.

    Parameters
    ----------
    self.censoring_value : float
        Value under which observations are censored.
    """

    censoring_value: bool = attrs.field(validator=attrs.validators.instance_of((float)), converter=float)

    @staticmethod
    def _fit_censored_gamma(x: np.ndarray, nr_censored_x: int, min_x: float) -> tuple:
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

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fits a censored gamma distribution to precipitation data where everything under `self.censoring_value` is assumed to be
            a censored observation.

        Parameters
        ----------
        data : np.ndarray
            Data on which to fit the censored gamma distribution.

        Returns
        -------
        tuple
            Parameter estimates for the gamma distribution.
        """
        noncensored_data = data[data > self.censoring_value]
        return gen_PrecipitationGammaLeftCensoredModel._fit_censored_gamma(
            noncensored_data, data.size - noncensored_data.size, self.censoring_value
        )

    def cdf(self, x: np.ndarray, fit: tuple) -> np.ndarray:
        """
        Returns cdf-values of a vector x for the cdf of a precipitation left censored gamma-model.
        Values x below the censoring value (mainly zeros) are first randomized between (0, censoring_value) before calculating the gamma-cdf.

        Parameters
        ----------
        x : np.ndarray
            Values for which the cdf shall be evaluated.
        fit : tuple
            Parameters controling the censored gamma-distribution: shape and scale

        Returns
        -------
        np.ndarray
            Array containing cdf-values for x.
        """
        x = np.where(x < self.censoring_value, np.random.uniform(0, self.censoring_value), x)
        return scipy.stats.gamma.cdf(x, *fit)

    def ppf(self, q: np.ndarray, fit: tuple) -> np.ndarray:
        """
        Returns ppf (quantile / inverse cdf)-values of a vector q for the cdf of a precipitation left-censored gamma model.
        Values generated by the gamma ppf below the censoring value are  set to zero.

        Parameters
        ----------
        q : np.ndarray
            Values for which the ppf shall be evaluated.
        fit : tuple
            Parameters controling the censored gamma-distribution: shape and scale

        Returns
        -------
        np.ndarray
            Array containing cdf-values for x.
        """
        vals = scipy.stats.gamma.ppf(q, *fit)
        return np.where(vals < self.censoring_value, 0, vals)


PrecipitationGammaLeftCensoredModel_5mm_threshold = gen_PrecipitationGammaLeftCensoredModel(0.05)

# TODO: think about possibility of kwargs and censoring_value outside of class: as kwarg

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
