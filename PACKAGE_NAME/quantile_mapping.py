import scipy
import numpy as np
import warnings

from scipy.stats import norm, gamma
from debiaser import Debiaser
from math_helpers import fit_precipitation_hurdle_model, quantile_mapping_precipitation_hurdle_model, fit_precipitation_censored_gamma, quantile_mapping_precipitation_censored_gamma 
from variable_distribution_match import standard_distributions

class QuantileMapping(Debiaser):
    def __init__(self, delta_type, variable = None, distribution = None, precip_model_type = "censored", precip_hurdle_randomization = False, precip_censoring_value = 0.1):
        
        if distribution is None and variable is None:
            raise ValueError("Either distribution or variable need to be specified")
        
        if distribution is None:
            distribution = standard_distributions[variable]

        if not isinstance(distribution, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)):
            raise TypeError("Wrong type for distribution. Needs to be scipy.stats.rv_continuous or scipy.stats.rv_discrete") 
        
        if not delta_type in ["additive", "multiplicative", "none"]:
            raise ValueError("Wrong value for delta_type. Needs to be one of ['additive', 'multiplicative', 'none']")
        
        if not precip_model_type in ["censored", "hurdle"]:
            raise ValueError("Wrong value for precip_model_type. Needs to be one of ['censored', 'hurdle']")
                
        
        if variable is None:
            variable = "unknown"
            
        self.variable = variable
        self.distribution = distribution
        self.delta_type = delta_type
        
        # Precipitation
        if variable in ["precip", "precipitation"] and precip_model_type == "censored":
            warnings.warn("Only the gamma distribution is supported for a censored precipitation model")
            
        self.precip_model_type = precip_model_type
        self.precip_censoring_value = precip_censoring_value
        self.precip_hurdle_randomization = precip_hurdle_randomization
    
    def apply_location_precip_hurdle(self, obs, cm_hist, cm_future):
        fit_obs = fit_precipitation_hurdle_model(obs, self.distribution)
        fit_cm_hist = fit_precipitation_hurdle_model(cm_hist, self.distribution)

        if self.delta_type == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return quantile_mapping_precipitation_hurdle_model(cm_future - delta, fit_cm_hist, fit_obs, self.distribution, self.distribution, self.precip_hurdle_randomization) + delta
        elif self.delta_type == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return quantile_mapping_precipitation_hurdle_model(cm_future / delta, fit_cm_hist, fit_obs, self.distribution, self.distribution, self.precip_hurdle_randomization) * delta
        elif self.delta_type == "none":
            return quantile_mapping_precipitation_hurdle_model(cm_future, fit_cm_hist, fit_obs, self.distribution, self.distribution, self.precip_hurdle_randomization)
        else:
            raise ValueError("self.delta_type has wrong value. Needs to be one of ['additive', 'multiplicative', 'none']")

    def apply_location_precip_censored(self, obs, cm_hist, cm_future):
        fit_obs = fit_precipitation_censored_gamma(obs, self.precip_censoring_value)
        fit_cm_hist = fit_precipitation_censored_gamma(cm_hist, self.precip_censoring_value)

        if self.delta_type == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return quantile_mapping_precipitation_censored_gamma(cm_future - delta, self.precip_censoring_value, fit_cm_hist, fit_obs) + delta
        elif self.delta_type == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return quantile_mapping_precipitation_censored_gamma(cm_future / delta, self.precip_censoring_value, fit_cm_hist, fit_obs) * delta
        elif self.delta_type == "none":
            return quantile_mapping_precipitation_censored_gamma(cm_future, self.precip_censoring_value, fit_cm_hist, fit_obs)
        else:
            raise ValueError("self.delta_type has wrong value. Needs to be one of ['additive', 'multiplicative', 'none']")

    def standard_qm(self, x, fit_cm_hist, fit_obs):
        return self.distribution.ppf(self.distribution.cdf(x, *fit_cm_hist), *fit_obs)
    
    def apply_location_standard(self, obs, cm_hist, cm_future):
        fit_obs = self.distribution.fit(cm_hist)
        fit_cm_hist = self.distribution.fit(cm_hist)
        
        if self.delta_type == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return self.standard_qm(cm_future - delta, fit_cm_hist, fit_obs) + delta
        elif self.delta_type == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return self.standard_qm(cm_future / delta, fit_cm_hist, fit_obs) * delta
        elif self.delta_type == "none":
            return self.standard_qm(cm_future, fit_cm_hist, fit_obs)
        else:
            raise ValueError("self.delta_type has wrong value. Needs to be one of ['additive', 'multiplicative', 'none']")
    
    def apply_location(self, obs, cm_hist, cm_future):   
        if self.variable in ["precipitation", "precip"]:
            if self.precip_model_type == "censored":
                return self.apply_location_precip_censored(obs, cm_hist, cm_future)
            elif self.precip_model_type == "hurdle":
                return self.apply_location_precip_censored(obs, cm_hist, cm_future)
            else: 
                raise ValueError("Invalid self.precip_model_type. Needs to be one of ['censored', 'hurdle']")
        else:
            return self.apply_location_standard(obs, cm_hist, cm_future)

    def apply(self, obs, cm_hist, cm_future):
        print("----- Running debiasing -----")
        Debiaser.check_inputs(obs, cm_hist, cm_future)
        return Debiaser.map_over_locations(self.apply_location, obs, cm_hist, cm_future, cm_future.shape[0])
    
    """
    def cache_location(self, obs, cm_hist, cm_future):
        fit_obs = self.distribution.fit(cm_hist)
        fit_cm_hist = self.distribution.fit(cm_future)
        
        return lambda obs, cm_hist, cm_future: self.distribution.ppf(self.distribution.cdf(cm_future, *fit_cm_hist), *fit_obs)
    """