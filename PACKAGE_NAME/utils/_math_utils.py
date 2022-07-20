# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""math_helpers module - helpers used by different debiasers"""

from abc import ABC, abstractmethod

import attrs
import numpy as np
import scipy.optimize
import scipy.stats
import statsmodels.distributions.empirical_distribution


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
    def cdf(self, x: np.ndarray, *fit: tuple, **kwargs) -> np.ndarray:
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
    def ppf(self, q: np.ndarray, *fit: tuple, **kwargs) -> np.ndarray:
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


class _gen_PrecipitationPlaceholder(StatisticalModel):
    def ppf(self, q: np.ndarray, *fit: tuple, **kwargs) -> np.ndarray:
        raise NotImplementedError("A concrete precipitation statistical modle needs to be implemented")

    def cdf(self, x: np.ndarray, *fit: tuple, **kwargs) -> np.ndarray:
        raise NotImplementedError("A concrete precipitation statistical modle needs to be implemented")

    def fit(self, data: np.ndarray, **kwargs) -> tuple:
        raise NotImplementedError("A concrete precipitation statistical modle needs to be implemented")


PrecipitationPlaceholder = _gen_PrecipitationPlaceholder()


@attrs.define
class gen_PrecipitationIgnoreZeroValuesModel(StatisticalModel):
    """
    Represents a precipitation model where zero values are ignored and a cdf is only fitted to amounts.
    In the cdf zero values are mapped to -np.inf and in the inverse cdf values of -np.inf are mapped to zero.

    Parameters
    ----------
    self.distribution : scipy.stats.rv_continuous
        Distribution assumed for the precipitation amounts.
    """

    distribution: scipy.stats.rv_continuous = attrs.field(
        default=scipy.stats.gamma, validator=attrs.validators.instance_of(scipy.stats.rv_continuous)
    )

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fits a precipitation model to the amounts (ignoring zero values).


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
        fit_rainy_days = self.distribution.fit(rainy_days)

        return fit_rainy_days

    def cdf(self, x: np.ndarray, *fit: tuple) -> np.ndarray:
        """
        Returns cdf-values for the precipitation amounts and -np.inf for zero.

        Parameters
        ----------
        x : np.ndarray
            Values for which the cdf shall be evaluated.
        fit : tuple
            Parameters controling the amounts model.
            Return value of fit.

        Returns
        -------
        np.ndarray
            Array containing cdf-values for x.
        """

        return np.where(
            x == 0,
            -np.inf,
            self.distribution.cdf(x, *fit),
        )

    def ppf(self, q: np.ndarray, *fit: tuple) -> np.ndarray:
        """
        Returns ppf (quantile / inverse cdf)-values of a vector q of the amounts ppf and 0 for -np.inf.

        Parameters
        ----------
        q : np.ndarray
            Values for which the ppf shall be evaluated.
        fit : tuple
            Parameters controling the amounts model.
            Return value of fit.

        Returns
        -------
        np.ndarray
            Array containing cdf-values for x.
        """

        return np.where(q != -np.inf, self.distribution.ppf(q, *fit), 0)


PrecipitationGammaModelIgnoreZeroValues = gen_PrecipitationIgnoreZeroValuesModel(distribution=scipy.stats.gamma)


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

    def cdf(self, x: np.ndarray, *fit: tuple) -> np.ndarray:
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

        if not self.cdf_randomization:
            return np.where(x == 0, p0, p0 + (1 - p0) * self.distribution.cdf(x, *fit_rainy_days))
        else:
            return np.where(
                x == 0,
                np.random.uniform(0, p0),
                p0 + (1 - p0) * self.distribution.cdf(x, *fit_rainy_days),
            )

    def ppf(self, q: np.ndarray, *fit: tuple) -> np.ndarray:
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
            return -np.sum(scipy.stats.gamma.logpdf(x, a=params[0], scale=params[1])) - nr_censored_x * np.log(
                scipy.stats.gamma.cdf(min_x, a=params[0], scale=params[1])
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

    def cdf(self, x: np.ndarray, *fit: tuple) -> np.ndarray:
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

    def ppf(self, q: np.ndarray, *fit: tuple) -> np.ndarray:
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


def iecdf(x: np.ndarray, p: np.ndarray, method: str = "inverted_cdf", **kwargs):
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


def ecdf(x: np.array, y: np.array, method: str = "step_function") -> np.array:
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
    x : np.ndarray
        Array containing values with which the empirical cdf is defined.
    y : np.ndarray
        Array containing values on which the empirical cdf is evaluated.
    method : str
        Method with which the ecdf is calculated. One of ["kernel_density", "linear_interpolation", "step_function"].

    Returns
    -------
    np.ndarray
        Values of the empirical cdf of x evaluated at y.
    """
    if method == "kernel_density":
        rv_histogram_fit = scipy.stats.rv_histogram(np.histogram(x, bins="auto"))
        return rv_histogram_fit.cdf(y)
    elif method == "linear_interpolation":
        p_grid = np.linspace(0.0, 1.0, x.size)
        q_vals_for_p_grid = np.quantile(x, p_grid)
        return np.interp(y, q_vals_for_p_grid, p_grid)
    elif method == "step_function":
        step_function = statsmodels.distributions.empirical_distribution.ECDF(x)
        return step_function(y)
    else:
        raise ValueError('method needs to be one of ["kernel_density", "linear_interpolation", "step_function"] ')


def quantile_map_non_parametically(
    x: np.ndarray,
    y: np.ndarray,
    vals: np.ndarray,
    ecdf_method: str = "step_function",
    iecdf_method: str = "inverted_cdf",
    **kwargs,
) -> np.ndarray:
    """
    Quantiles maps a vector of values vals using empirical distributions defined by vectors x and y.
    Quantiles of values in vals are first found using the ecdf of the values in x. Afterwards they are transformed onto y using the empirical inverse cdf of y.

    Parameters:
        x: np.ndarray
            Values defining an empirical distribution with whose ecdf the quantiles are transformed.
        y: np.ndarray
            Values defining an empirical distribution with whose iecdf the quantiles are transformed.
        vals: np.ndarray
            Values to quantile map non parametically.
        ecdf_method: str
            Method to use for the ecdf (transformation of x). Passed to ecdf.
        iecdf_method: str
            Method to use for the iecdf (transformation of the quantiles). Passed to iecdf.
        **kwargs:
            Passed to iecdf.
    """

    return iecdf(y, ecdf(x, vals, method=ecdf_method), method=iecdf_method, **kwargs)


def quantile_map_non_parametically_with_constant_extrapolation(
    x: np.ndarray,
    y: np.ndarray,
    vals: np.ndarray,
    ecdf_method: str = "step_function",
    iecdf_method: str = "inverted_cdf",
    **kwargs,
) -> np.ndarray:
    """
    Quantiles maps a vector of values vals using empirical distributions defined by vectors x and y with constant extrapolation: values above the range of x are corrected using the correction for the maximum quantile of x. Values below the range of x are corrected using the correction for the minimum quantile of x.
    Quantiles of values in vals are first found using the ecdf of the values in x. Afterwards they are transformed onto y using the empirical inverse cdf of y.

    Parameters:
        x: np.ndarray
            Values defining an empirical distribution with whose ecdf the quantiles are transformed.
        y: np.ndarray
            Values defining an empirical distribution with whose iecdf the quantiles are transformed.
        vals: np.ndarray
            Values to quantile map non parametically.
        ecdf_method: str
            Method to use for the ecdf (transformation of x). Passed to ecdf.
        iecdf_method: str
            Method to use for the iecdf (transformation of the quantiles). Passed to iecdf.
        **kwargs:
            Passed to iecdf.
    """
    mapped_vals = quantile_map_non_parametically(
        x=x, y=y, vals=vals, ecdf_method=ecdf_method, iecdf_method=iecdf_method, **kwargs
    )

    vals_under = vals < (x_min := x.min())
    vals_above = vals > (x_max := x.max())
    correction_zero_and_one = np.array([np.min(y), np.max(y)]) - np.array(
        [x_min, x_max]
    )  # iecdf(y, np.array([0, 1]), method = "linear") - iecdf(x, np.array([0,1]), method = "linear")

    mapped_vals[vals_under] = vals[vals_under] + correction_zero_and_one[0]
    mapped_vals[vals_above] = vals[vals_above] + correction_zero_and_one[1]

    return mapped_vals


def _isimip_quantile_map_x_on_y_non_parametically(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p_x = (scipy.stats.rankdata(x) - 1.0) / x.size
    p_y = np.linspace(0.0, 1.0, y.size, dtype=y.dtype)
    z = np.interp(p_x, p_y, np.sort(y)).astype(y.dtype)
    return z


def quantile_map_x_on_y_non_parametically(
    x: np.ndarray,
    y: np.ndarray,
    mode: str = "normal",
    ecdf_method: str = "step_function",
    iecdf_method: str = "inverted_cdf",
    **kwargs,
) -> np.ndarray:
    """
    Quantiles maps a vector of values x onto a vector of values y.
    If mode = "normal" this is done using the normal quantile_map_non_parametically-function.
    If mode = "isimipv3.0" the scipy.stats.rankdata function is used for the quantiles and linear interpolation to find the corresponding quantile mapped values. In this case the ecdf_method and iecdf_method arguments are ignored.

    Parameters:
        x: np.ndarray
            Values to quantile map non parametically using their ecdf.
        y: np.ndarray
            Values defining an empirical distribution with whose iecdf the quantiles are transformed.
        mode: str
            One of ["normal", "isimipv3.0"]. Determines the mode of quantile mapping.
            If mode = "normal" this is done using the empirical cdf and inverse cdf.
        ecdf_method: str
            Method to use for the ecdf (transformation of x). Passed to ecdf.
        iecdf_method: str
            Method to use for the iecdf (transformation of the quantiles). Passed to iecdf.
        **kwargs:
            Passed to iecdf.
    """

    if mode == "normal":
        return quantile_map_non_parametically(
            x=x, y=y, vals=x, ecdf_method=ecdf_method, iecdf_method=iecdf_method, **kwargs
        )
    elif mode == "isimipv3.0":
        return _isimip_quantile_map_x_on_y_non_parametically(x, y)
    else:
        raise ValueError('mode needs to be one of ["normal", "isimipv3.0"]')
