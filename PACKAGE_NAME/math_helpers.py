import numpy as np
import scipy.optimize
import scipy.stats
from scipy.stats import gamma

"""----- Precipitation-helpers -----"""
# TODO: in gamma.fit shall we specify gamma.fit(floc = 0) so keep loc fixed at zero?

# Hurdle model: two step process: binomial if it rains and then amounts how much. P(X = 0) = p0, P(0 < X <= x) = p0 + (1-p0) F_A(x)
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

    return np.where(q > p0, distribution.ppf((q - p0) / (1 - p0)), 0)


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
