import numpy as np
import scipy
from debiaser import Debiaser
from statsmodels.distributions.empirical_distribution import ECDF


# Reference Cannon et al. 2015
class DeltaQuantileMapping(Debiaser):
    def __init__(self, distribution, time_window_length):
        if not isinstance(
            distribution, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)
        ):
            raise TypeError(
                "Wrong type for distribution. Needs to be scipy.stats.rv_continuous or scipy.stats.rv_discrete"
            )

        if not isinstance(time_window_length, int):
            raise TypeError("Wrong type for time_window_length. Needs to int")

        if not time_window_length > 0:
            raise ValueError("Wrong value for time_window_length. Needs to be > 0")

        self.distribution = distribution
        self.time_window_length = time_window_length

    def apply_location(self, obs, cm_hist, cm_future):

        fit_obs = self.distribution.fit(cm_hist)
        fit_cm_hist = self.distribution.fit(cm_hist)

        # Q: what is the more efficient way to chunk and then return chunks of equal size to write a new array. That, solution below or totally different one?
        # T: time measurements are quite variable. More testing necessary. Which solution is cleaner?
        debiased_cm_list_of_time_windows = []
        for cm_future_time_window in np.array_split(
            cm_future, self.time_window_length, axis=0
        ):
            ecdf_cm_future_time_window = ECDF(cm_future_time_window)
            tau_t = ecdf_cm_future_time_window(cm_future_time_window)
            debiased_cm_list_of_time_windows.append(
                cm_future_time_window
                * self.distribution.ppf(tau_t, *fit_obs)
                / self.distribution.ppf(tau_t, *fit_cm_hist)
            )
        debiased_cm = np.concatenate(debiased_cm_list_of_time_windows)

        # Alternative:
        """
        debiased_cm = np.empty(cm_future.shape[0])
        for time_chunk in np.array_split(np.array(range(cm_future.shape[0])), self.time_window_length, axis=0):
            ecdf_cm_future_time_window = ECDF(cm_future[time_chunk])
            tau_t = ecdf_cm_future_time_window(cm_future[time_chunk])
                        
            debiased_cm[time_chunk] = cm_future[time_chunk] * self.distribution.ppf(tau_t, *fit_obs) / self.distribution.ppf(tau_t, *fit_cm_hist)
        """

        return debiased_cm
