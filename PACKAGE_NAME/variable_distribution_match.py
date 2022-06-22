import scipy.stats

standard_distributions = {
    "temp": scipy.stats.norm,
    "temperature": scipy.stats.norm,
    "precip": scipy.stats.gamma,
    "precipitation": scipy.stats.gamma,
}
