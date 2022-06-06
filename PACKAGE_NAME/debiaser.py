# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
from tqdm import tqdm


# Debiaser class
class Debiaser():
    def __init__():
        return self
    
    def apply_location(self, obs, cm_hist, cm_future):
        raise NotImplementedError("apply_location is an abstract method which needs to be overriden in derived classes.")
        
    def apply(self, obs, cm_hist, cm_future):
        raise NotImplementedError("apply is an abstract method which needs to be overriden in derived classes.")
    

    # Input checks:
    @staticmethod
    def is_correct_type(df):
        return isinstance(df, np.ndarray)
    
    @staticmethod
    def has_correct_shape(df):
        if df.ndim == 3:
            return True
        else:
            return False
        
    @staticmethod
    def have_same_shape(obs, cm_hist, cm_future):
        if obs.shape[1:] == cm_hist.shape[1:] and obs.shape[1:] == cm_future.shape[1:]:
            return True
        else:
            return False

    @staticmethod
    def check_inputs(obs, cm_hist, cm_future):
        # correct type
        if not Debiaser.is_correct_type(obs):
            raise TypeError("Wrong type for obs. Needs to be np.ndarray") 
        
        # correct shape
        if not Debiaser.has_correct_shape(obs):
            raise ValueError("obs needs to have 3 dimensions: time, x, y")

        if not Debiaser.has_correct_shape(cm_hist):
            raise ValueError("cm_hist needs to have 3 dimensions: time, x, y")

        if not Debiaser.has_correct_shape(cm_future):
            raise ValueError("cm_future needs to have 3 dimensions: time, x, y")

        # have shame shape
        if not Debiaser.have_same_shape(obs, cm_hist, cm_future):
            raise ValueError("obs, cm_hist, cm_future need to have same (number of) spatial dimensions")

        return True

    # Helpers for downstream:
    @staticmethod
    def map_over_locations(fct, obs, cm_hist, cm_future, nr_timesteps_output = None):
        """
        Maps locationwise-defined function over all gridpoints.

        Parameters
        ----------
        fct : function
            Function to be mapped over all locations.
        obs : array
            Array of observations.
        cm_hist : array
            Array of climate model simulation data over historical period.
        cm_future : array
            Array of climate model simulation data over future period.
        nr_timesteps_output : int
            Output-length at individual location.
            
        Returns
        -------
        np.ndarray
            Numpy array containing locationwise output.
        """
        
        if nr_timesteps_output is None:
            nr_timesteps_output  = cm_future.shape[0]


        output = np.empty([nr_timesteps_output, obs.shape[1], obs.shape[2]])
        for i, j in tqdm(np.ndindex(obs.shape[1:]), total = np.prod(obs.shape[1:])):
            output[:, i, j] = fct(obs[:, i, j], cm_hist[:, i, j], cm_future[:, i, j])[0:nr_timesteps_output+1]
                
        return output 