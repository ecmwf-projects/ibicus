import scipy
import numpy as np

from debiaser import Debiaser

class QuantileMapping(Debiaser):
    def __init__(self, distribution, delta_type):
        if not isinstance(distribution, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)):
            raise TypeError("Wrong type for distribution. Needs to be scipy.stats.rv_continuous or scipy.stats.rv_discrete") 
        
        if not delta_type in ["additive", "multiplicative", "none"]:
            raise ValueError("Wrong value for delta_type. Needs to be one of ['additive', 'multiplicative', 'none']")
        
        self.distribution = distribution
        self.delta_type = delta_type
        
        return None
    
    def apply_location(self, obs, cm_hist, cm_future):     
        fit_obs = self.distribution.fit(cm_hist)
        fit_cm_hist = self.distribution.fit(cm_hist)
        
        if self.delta_type == "additive":
            delta = np.mean(cm_future) - np.mean(cm_hist)
            return self.distribution.ppf(self.distribution.cdf(cm_future - delta, *fit_cm_hist), *fit_obs) + delta
        elif self.delta_type == "multiplicative":
            delta = np.mean(cm_future) / np.mean(cm_hist)
            return self.distribution.ppf(self.distribution.cdf(cm_future / delta, *fit_cm_hist), *fit_obs) * delta
        elif self.delta_type == "none":
            return self.distribution.ppf(self.distribution.cdf(cm_future, *fit_cm_hist), *fit_obs)
        else:
            raise ValueError("self.delta_type has wrong value. Needs to be one of ['additive', 'multiplicative', 'none']")

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