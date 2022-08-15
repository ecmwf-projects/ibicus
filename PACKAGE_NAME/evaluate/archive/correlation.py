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
#from libpysal.weights import lat2W
#from esda.moran import Moran

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




'''def calculate_moransi_spatial(dataset):
    
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

    return(moransi)'''



import math
import sklearn

def rmse_spatial_correlation_distribution(variable, data_obs, **kwargs):
    
    rmsd_array = np.empty((0, 2))
    
    for k in kwargs.keys():
        
        rmsd = np.zeros((data_obs.shape[1], data_obs.shape[2]))
        
        for a in range(data_obs.shape[1]):
            for b in range(data_obs.shape[2]):

                corr_matrix_obs = np.zeros((data_obs.shape[1], data_obs.shape[2]))
                corr_matrix_cm = np.zeros((data_obs.shape[1], data_obs.shape[2]))


                for i in range(data_obs.shape[1]):
                      for j in range(data_obs.shape[2]):
                            
                            corr_matrix_obs[i,j] = np.corrcoef(data_obs[:,a,b], data_obs[:,i,j])[0,1]
                            corr_matrix_cm[i,j] = np.corrcoef(kwargs[k][:,a,b], kwargs[k][:,i,j])[0,1]

                rmsd[a,b] = math.sqrt(sklearn.metrics.mean_squared_error(corr_matrix_obs, corr_matrix_cm))

        array = np.transpose(np.array([[k]*len(np.ndarray.flatten(rmsd)), 
                                    np.transpose(np.ndarray.flatten(rmsd))]))
        
        rmsd_array = np.append(rmsd_array, array, axis = 0)
    

    
    rmsd_data = pd.DataFrame(rmsd_array, columns=['Correction Method', 'RMSE spatial correlation'])
    rmsd_data["RMSE spatial correlation"] = pd.to_numeric(rmsd_data["RMSE spatial correlation"])
    
    return(rmsd_data)


def rmse_spatial_correlation_boxplot(variable, dataset):
    
    fig = plt.figure(figsize=(8, 6))
    seaborn.boxplot(y='RMSE spatial correlation',
                    x='Correction Method',
                   data=dataset, 
                   palette="colorblind")

    fig.suptitle('{} \n RMSE of spatial correlation matrices)'.format(variable_dictionary.get(variable).get('name')))

    return fig
       

