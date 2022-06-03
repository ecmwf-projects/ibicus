import scipy
import numpy as np

from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.signal import detrend
from debiaser import Debiaser

# Reference Cannon et al. 2015
class Switanek2017(Debiaser):
    def __init__(self):
        pass
    
    @staticmethod
    def apply_cdf_thresholding(cdf, cdf_threshold = 0.0001):
        return np.maximum(np.minimum(cdf, 1-cdf_threshold), cdf_threshold)

    def apply_location_temp(self, obs, cm_hist, cm_future): 
        
        # Step 1
        obs_detrended = detrend(obs, type = "constant")
        cm_hist_detrended = detrend(cm_hist, type = "constant")
        cm_future_detrended = detrend(cm_future, type = "constant")
        
        # Step 2
        fit_obs_detrended = norm.fit(obs_detrended)
        fit_cm_hist_detrended = norm.fit(cm_hist_detrended)
        fit_cm_future_detrended = norm.fit(cm_future_detrended)
        
        argsort_cm_future = np.argsort(cm_future_detrended)
        
        cdf_vals_obs_detrended_thresholded = Switanek2017.apply_cdf_thresholding(np.sort(norm.cdf(obs_detrended)))
        cdf_vals_cm_hist_detrended_thresholded = Switanek2017.apply_cdf_thresholding(np.sort(norm.cdf(cm_hist_detrended)))
        cdf_vals_cm_future_detrended_thresholded = Switanek2017.apply_cdf_thresholding(norm.cdf(cm_future_detrended)[argsort_cm_future])
        
        # interpolate cdf-values for obs and mod to the length of the scenario
        cdf_vals_obs_detrended_thresholded_intpol = np.interp(
            np.linspace(1, len(obs), len(cm_future)),
            np.linspace(1, len(obs), len(obs)),
            cdf_vals_obs_detrended_thresholded
        )
        cdf_vals_cm_hist_detrended_thresholded_intpol = np.interp(
            np.linspace(1, len(cm_hist), len(cm_future)),
            np.linspace(1, len(cm_hist), len(cm_hist)),
            cdf_vals_cm_hist_detrended_thresholded
        )

        # Step 3
        scaling = (norm.ppf(cdf_vals_cm_future_detrended_thresholded, *fit_cm_future_detrended) - norm.ppf(cdf_vals_cm_future_detrended_thresholded, *fit_cm_hist_detrended)) * fit_obs_detrended[1]/fit_cm_hist_detrended[1]
        
        # Step 4
        recurrence_interval_obs = 1/(0.5 - np.abs(cdf_vals_obs_detrended_thresholded_intpol - 0.5))
        recurrence_interval_cm_hist = 1/(0.5 - np.abs(cdf_vals_cm_hist_detrended_thresholded_intpol - 0.5))
        recurrence_interval_cm_future = 1/(0.5 - np.abs(cdf_vals_cm_future_detrended_thresholded - 0.5))
        
        # Step 5
        recurrence_interval_scaled = np.maximum(1, recurrence_interval_obs*recurrence_interval_cm_future/recurrence_interval_cm_hist)
        cdf_scaled = Switanek2017.apply_cdf_thresholding(0.5 + np.sign(cdf_vals_obs_detrended_thresholded_intpol - 0.5)*np.abs(0.5 - 1/recurrence_interval_scaled))
        
        # Step 6
        bias_corrected = norm.ppf(cdf_scaled, *fit_obs_detrended) + scaling
        
        # Step 7
        return bias_corrected[argsort_cm_future] + (cm_future - cm_future_detrended)
        
        # Open things:
        # - Not entirely clear what this code does. Different than from paper:
        # adapted_cdf = np.sign(obs_cdf_shift) * (
        #        1. - 1. / (obs_inverse * sce_inverse / mod_inverse))
        # adapted_cdf[adapted_cdf < 0] += 1.
        # - Not entirely clear why the mean of the scaled values is subtracted and then shift with obs mean pluc difference of future and historical. Not in paper:
        # xvals -= xvals.mean()
        # xvals += obs_mean + (sce_mean - mod_mean)
        # - Not clear why scenario mean subtracted here again when trend readded:
        # correction += sce_diff - sce_mean

    def apply(self, obs, cm_hist, cm_future):
        print("----- Running debiasing -----")
        Debiaser.check_inputs(obs, cm_hist, cm_future)
        return Debiaser.map_over_locations(self.apply_location_temp, obs, cm_hist, cm_future, cm_future.shape[0])
        
        