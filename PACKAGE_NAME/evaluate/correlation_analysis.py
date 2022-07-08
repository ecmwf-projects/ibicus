# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

import seaborn
import pandas as pd

#import libpysal
from libpysal.weights import lat2W
from esda.moran import Moran

from pylab import *
from scipy.ndimage import measurements

from statsmodels.tsa.stattools import adfuller

variable_dictionary = {
        "tas": {
        "distribution": scipy.stats.norm,
        "trend_preservation": "additive",
        "detrending": True,
        "name": '2m daily mean air temperature (K)',
        "high_threshold": 295,
        "low_threshold": 273,
        "unit": 'K'
    },
    "pr": {
        "distribution": scipy.stats.gamma,
        "trend_preservation": "mixed",
        "detrending": False,
        "name": 'Total precipitation (m/day)',
        "high_threshold": 0.0004,
        "low_threshold": 0.00001,
        "unit": 'm/day'
    }
}




def calculate_moransi_spatial(dataset):
    
    moransi = np.zeros(dataset.shape[0])
    
    for i in range(dataset.shape[0]):
    
        Z = dataset[i,:,:]
        # Create the matrix of weigthts - runing w.neighbors seems to imply that neighbors 
        # are those neighboring cells sharing an edge, pay attention here as result of Moran's I
        # depends heavily on the choice of w
        w = lat2W(Z.shape[0], Z.shape[1])
        # Crate the pysal Moran object 
        mi = Moran(Z, w)
        moransi[i] = mi.I

    return(moransi)



def rmse_spatial_correlation(variable, name_BC, data_obs, data_raw, data_bc):

  rmsd_raw = np.zeros((data_obs.shape[1], data_obs.shape[2]))
  rmsd_bc = np.zeros((data_obs.shape[1], data_obs.shape[2]))

  for a in range(data_obs.shape[1]):
    for b in range(data_obs.shape[2]):

        corr_matrix_obs = np.zeros((data_obs.shape[1], data_obs.shape[2]))
        corr_matrix_raw = np.zeros((data_raw.shape[1], data_raw.shape[2]))
        corr_matrix_bc = np.zeros((data_bc.shape[1], data_bc.shape[2]))

        for i in range(data_obs.shape[1]):
          for j in range(data_obs.shape[2]):
            corr_matrix_obs[i,j] = np.corrcoef(data_obs[:,a,b], data_obs[:,i,j])[0,1]
            corr_matrix_raw[i,j] = np.corrcoef(data_raw[:,a,b], data_raw[:,i,j])[0,1]
            corr_matrix_bc[i,j] = np.corrcoef(data_bc[:,a,b], data_bc[:,i,j])[0,1]

        rmsd_raw[a,b] = sqrt(mean_squared_error(corr_matrix_obs, corr_matrix_raw))
        rmsd_bc[a,b] = sqrt(mean_squared_error(corr_matrix_obs, corr_matrix_bc))
    
    array1 = np.transpose(np.array([['Raw']*len(np.ndarray.flatten(rmsd_raw)), 
                                    np.transpose(np.ndarray.flatten(rmsd_raw))]))
    array2 = np.transpose(np.array([[name_BC]*len(np.ndarray.flatten(rmsd_bc)), 
                                    np.transpose(np.ndarray.flatten(rmsd_bc))]))
    
    arrays = np.concatenate((array1, array2))
    
    spatial_corr = pd.DataFrame(arrays, columns=['Correction Method', 'RMSE spatial correlation'])
    spatial_corr["RMSE spatial correlation"] = pd.to_numeric(spatial_corr["RMSE spatial correlation"])
    
    return(spatial_corr)


def rmse_spatial_correlation_boxplot(variable, dataset):
    
    fig = plt.figure(figsize=(8, 6))
    seaborn.violinplot(y='RMSE spatial correlation', x='RMSE spatial correlation', 
                   data=dataset, 
                   palette="colorblind")

    fig.suptitle('{} \n RMSE of spatial correlation matrices)'.format(variable_dictionary.get(variable).get('name')))

    return fig
       

