# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn

import math
import sklearn.metrics

from PACKAGE_NAME.variables import *



def rmse_spatial_correlation_distribution(variable, obs_data, **cm_data):

    rmsd_array = np.empty((0, 2))

    for k in cm_data.keys():

        rmsd = np.zeros((obs_data.shape[1], obs_data.shape[2]))

        for a in range(obs_data.shape[1]):
            for b in range(obs_data.shape[2]):

                corr_matrix_obs = np.zeros((obs_data.shape[1], obs_data.shape[2]))
                corr_matrix_cm = np.zeros((obs_data.shape[1], obs_data.shape[2]))

                for i in range(obs_data.shape[1]):
                    for j in range(obs_data.shape[2]):

                        corr_matrix_obs[i, j] = np.corrcoef(obs_data[:, a, b], obs_data[:, i, j])[0, 1]
                        corr_matrix_cm[i, j] = np.corrcoef(cm_data[k][:, a, b], cm_data[k][:, i, j])[0, 1]

                rmsd[a, b] = math.sqrt(sklearn.metrics.mean_squared_error(corr_matrix_obs, corr_matrix_cm))

        array = np.transpose(np.array([[k] * len(np.ndarray.flatten(rmsd)), np.transpose(np.ndarray.flatten(rmsd))]))

        rmsd_array = np.append(rmsd_array, array, axis=0)

    rmsd_data = pd.DataFrame(rmsd_array, columns=["Correction Method", "RMSE spatial correlation"])
    rmsd_data["RMSE spatial correlation"] = pd.to_numeric(rmsd_data["RMSE spatial correlation"])

    return rmsd_data


def rmse_spatial_correlation_boxplot(variable, dataset):

    fig = plt.figure(figsize=(8, 6))
    seaborn.boxplot(y="RMSE spatial correlation", x="Correction Method", data=dataset, palette="colorblind")

    fig.suptitle("{} ({}) \n RMSE of spatial correlation matrices)".format(map_variable_str_to_variable_class(variable).name, map_variable_str_to_variable_class(variable).unit))

    return fig

def calculate_multivariate_correlation_locationwise(variables, **kwargs):
    
    correlation_matrix = {}
    
    for k in kwargs.keys():
        
        variable1 = kwargs[k][0]
        variable2 = kwargs[k][1]
        
        correlation_matrix[k] = np.zeros((variable1.shape[1], variable1.shape[2]))
    
        for i in range(variable1.shape[1]):
                for j in range(variable1.shape[2]):
                            
                        correlation_matrix[k][i,j] = np.corrcoef(variable1[:,i,j].T, variable2[:,i,j].T)[0,1]

    axis_max = max(abs(max(np.ndarray.flatten(np.vstack(list(chain(*correlation_matrix.values())))))), 
                   abs(min(np.ndarray.flatten(np.vstack(list(chain(*correlation_matrix.values())))))))
    axis_min = -axis_max
    
    fig_width = 6*len(kwargs.keys())
    fig, ax = plt.subplots(1, len(kwargs.keys()), figsize=(fig_width, 5))
    fig.suptitle("Multivariate Correlation: {} and {}".format(variable_dictionary.get(variables[0]).get('name'),
                                                              variable_dictionary.get(variables[1]).get('name')))
    
    i=0
    for k in kwargs.keys():
        
        plot = ax[i].imshow(correlation_matrix[k], cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
        ax[i].set_title('{}'.format(k))
        fig.colorbar(plot, ax=ax[i])
        i = i+1
        
    return(fig)
