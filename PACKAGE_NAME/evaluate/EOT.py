# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import scipy
import scipy.stats

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

import seaborn
import pandas as pd


from pylab import *
from scipy.ndimage import measurements

from statsmodels.tsa.stattools import adfuller
from itertools import chain

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



def calculate_matrix(dataset, threshold, thresholdtype):
    
    """
    Converts data into 1-0 matrix of same dimensions, 1 if the value at this time and location is below / above 
    a specific threshold and 0 otherwise.

    Parameters
    ----------
    dataset : dataset to be analysed, numeric entries expected
    threshold: numeric threshold value
    thresholdtype: '>' if values above threshold are to be analysed, '<' values below threshold are to be analysed.
    """
    
    thresholds = np.copy(dataset)

    if thresholdtype=='>':

      thresholds = (thresholds > threshold).astype(int)

    elif thresholdtype=='<':

      thresholds = (thresholds < threshold).astype(int)
      
    else:
      print('Invalid threshold type')
      
    return(thresholds)



def calculate_probability_once(variable, data, thresholdname, thresholdsign):
    
    threshold_data = calculate_matrix(dataset=data, 
                                           threshold=variable_dictionary.get(variable).get(thresholdname), 
                                           thresholdtype=thresholdsign)
    
    threshold_probability = np.zeros((threshold_data.shape[1], threshold_data.shape[2]))
    
    for i in range(threshold_data.shape[1]):
      for j in range(threshold_data.shape[2]):

        threshold_probability[i, j] = np.sum(threshold_data[:, i, j])/threshold_data.shape[0]
        
    return(threshold_probability)   




def calculate_marginal_bias(variable, data_obs, **kwargs):
    
    high_bias = {}
    low_bias = {}
    
    high_obs = calculate_probability_once(variable = variable, 
                                              data = data_obs, 
                                              thresholdname = 'high_threshold', 
                                              thresholdsign = '>')
    low_obs = calculate_probability_once(variable = variable, 
                                             data = data_obs, 
                                             thresholdname = 'low_threshold', 
                                             thresholdsign = '<')
    
    for k in kwargs.keys():
        
        high_bias[k] = high_obs*365 - 365*calculate_probability_once(variable = variable, 
                                                          data = kwargs[k], 
                                                          thresholdname = 'high_threshold', 
                                                          thresholdsign = '>')
        low_bias[k] = low_obs*365 - 365*calculate_probability_once(variable = variable, 
                                                  data = kwargs[k], 
                                                  thresholdname = 'low_threshold', 
                                                  thresholdsign = '<')

    bias_array = np.empty((0, 3))
    length = len(np.ndarray.flatten(high_bias['raw']))
       
    for k in ('raw', *kwargs.keys()):
           
        bias_array = np.append(bias_array,
                                  np.transpose(np.array([[k]*length, ['High']*length, np.transpose(np.ndarray.flatten(high_bias[k]))])), 
                                  axis = 0)
        bias_array = np.append(bias_array,
                                  np.transpose(np.array([[k]*length, ['Low']*length, np.transpose(np.ndarray.flatten(low_bias[k]))])), 
                                  axis = 0)


    plot_data = pd.DataFrame(bias_array, columns=['Correction Method','Threshold', 'Mean bias (days/year)'])
    plot_data["Mean bias (days/year)"] = pd.to_numeric(plot_data["Mean bias (days/year)"])
    
    return(plot_data)



def plot_distribution_mean_bias_days(variable, dataset):
    
    fig = plt.figure(figsize=(10, 6))
    seaborn.violinplot(y='Mean bias (days/year)', x='Threshold', 
                   data=dataset, 
                   palette="colorblind",
                   hue='Correction Method')


    fig.suptitle('{} - Bias \n (high threshold = {}{}, low threshold = {}{})'.format(variable_dictionary.get(variable).get('name'), 
                                                                              variable_dictionary.get(variable).get('high_threshold'),
                                                                              variable_dictionary.get(variable).get('unit'),    
                                                                              variable_dictionary.get(variable).get('low_threshold'),
                                                                              variable_dictionary.get(variable).get('unit')))

    return fig





def plot_2d_distribution_mean_bias_days(variable, data_obs, **kwargs):
    
    high_bias = {}
    low_bias = {}
    
    high_obs = calculate_probability_once(variable = variable, 
                                              data = data_obs, 
                                              thresholdname = 'high_threshold', 
                                              thresholdsign = '>')
    low_obs = calculate_probability_once(variable = variable, 
                                             data = data_obs, 
                                             thresholdname = 'low_threshold', 
                                             thresholdsign = '<')
    
    for k in kwargs.keys():
        
        high_bias[k] = high_obs*365 - 365*calculate_probability_once(variable = variable, 
                                                          data = kwargs[k], 
                                                          thresholdname = 'high_threshold', 
                                                          thresholdsign = '>')
        low_bias[k] = low_obs*365 - 365*calculate_probability_once(variable = variable, 
                                                  data = kwargs[k], 
                                                  thresholdname = 'low_threshold', 
                                                  thresholdsign = '<')
        
        
    arrays_max = max(max(np.ndarray.flatten(np.vstack(list(chain(*high_bias.values()))))),
                   max(np.ndarray.flatten(np.vstack(list(chain(*low_bias.values()))))))
    
    arrays_min = min(min(np.ndarray.flatten(np.vstack(list(chain(*high_bias.values()))))),
                   min(np.ndarray.flatten(np.vstack(list(chain(*low_bias.values()))))))
    
    axis_max = max(abs(arrays_max), abs(arrays_min))
    axis_min = -axis_max
    
    number_biascorrections = len(kwargs.keys())
    fig_length = 6*number_biascorrections
    
    fig, ax = plt.subplots(number_biascorrections ,2, figsize=(12,fig_length))

    fig.suptitle("{} - mean days/year bias of EOT".format(variable_dictionary.get(variable).get('name')))
    
    i=0
    
    for k in kwargs.keys():
        
        plot1 = ax[i, 0].imshow(high_bias[k], cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
        ax[i, 0].set_title('{} \n {} > {} {}'.format(k, variable_dictionary.get(variable).get('name'),
                                                  variable_dictionary.get(variable).get('high_threshold'),
                                                  variable_dictionary.get(variable).get('unit')))
        fig.colorbar(plot1, ax=ax[i, 0])
        
        plot2 = ax[i, 1].imshow(low_bias[k], cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
        ax[i, 1].set_title('{} \n {} < {} {}'.format(k, variable_dictionary.get(variable).get('name'),
                                                  variable_dictionary.get(variable).get('low_threshold'),
                                                  variable_dictionary.get(variable).get('unit')))
        fig.colorbar(plot2, ax=ax[i, 1])
        
        i = i+1

    return(fig)







def compute_spell_length(variable, min_length, **kwargs):
    
    spell_length_low = {}
    spell_length_high = {}
    
    spell_length_array = np.empty((0, 3))
    
    for k in kwargs.keys():
        
        threshold_data_low = calculate_matrix(kwargs[k],  variable_dictionary.get(variable).get('low_threshold'), '<')
        threshold_data_high = calculate_matrix(kwargs[k], variable_dictionary.get(variable).get('high_threshold'), '>')
        
        spell_length_once_low = np.array([])
        spell_length_once_high = np.array([])
        
        for i in range(threshold_data_low.shape[1]):
          for j in range(threshold_data_low.shape[2]):
            N=0
            for t in range(threshold_data_low.shape[0]):
              if threshold_data_low[t, i, j]==1:
                  N=N+1
              elif (threshold_data_low[t, i, j]==0) and (N!=0):
                  spell_length_once_low = np.append(spell_length_once_low, N)
                  N=0
                  
        for i in range(threshold_data_high.shape[1]):
          for j in range(threshold_data_high.shape[2]):
            N=0
            for t in range(threshold_data_high.shape[0]):
              if threshold_data_high[t, i, j]==1:
                  N=N+1
              elif (threshold_data_high[t, i, j]==0) and (N!=0):
                  spell_length_once_high = np.append(spell_length_once_high, N)
                  N=0
                
        spell_length_once_low = spell_length_once_low[spell_length_once_low>min_length]
        spell_length_once_high = spell_length_once_high[spell_length_once_high>min_length]
        spell_length_high[k] = spell_length_once_high
        spell_length_low[k] = spell_length_once_low
        
        length_low = len(spell_length_once_low)
        length_high = len(spell_length_once_high)
        
        spell_length_array = np.append(spell_length_array,
                               np.transpose(np.array([[k]*length_high, ['High']*length_high, np.transpose(spell_length_high[k])])), 
                               axis = 0)
        spell_length_array = np.append(spell_length_array,
                               np.transpose(np.array([[k]*length_low, ['Low']*length_low, np.transpose(spell_length_low[k])])), 
                               axis = 0)
        
    plot_data = pd.DataFrame(spell_length_array, columns=['Correction Method','Threshold', 'Spell length'])
    plot_data["Spell length"] = pd.to_numeric(plot_data["Spell length"])
    
    return(plot_data)
        
        


def calculate_spatiotemporal_clusters(variable, **kwargs):
    
    clusters_array = np.empty((0, 3))
    
    for k in kwargs.keys():
        
        high = calculate_matrix(dataset = kwargs[k], 
                                threshold = variable_dictionary.get(variable).get('high_threshold'),
                                thresholdtype = '>')
        high_lw, high_num = measurements.label(high)
        area_high = measurements.sum(high, high_lw, index=arange(high_lw.max() + 1))
        
        low = calculate_matrix(dataset = kwargs[k], 
                                threshold = variable_dictionary.get(variable).get('low_threshold'),
                                thresholdtype = '<')
        low_lw, low_num = measurements.label(low)
        area_low = measurements.sum(low, low_lw, index=arange(low_lw.max() + 1))
        
        length_low = len(area_low)
        length_high = len(area_high)
    
        clusters_array = np.append(clusters_array,
                               np.transpose(np.array([[k]*length_high, ['High']*length_high, np.transpose(area_high)])), 
                               axis = 0)
        clusters_array = np.append(clusters_array,
                               np.transpose(np.array([[k]*length_low, ['Low']*length_low, np.transpose(area_low)])), 
                               axis = 0)


    spatiotemporal_clusters = pd.DataFrame(clusters_array, columns=['Correction Method','Threshold', 'Cluster size'])
    spatiotemporal_clusters["Cluster size"] = pd.to_numeric(spatiotemporal_clusters["Cluster size"])
    
    return(spatiotemporal_clusters)



def calculate_spatial_count_once(variable, data, thresholdname, thresholdsign):
    
    spatial_count = np.array([])

    number_gridpoints = data.shape[1]*data.shape[2]

    threshold_data = calculate_matrix(dataset=data, 
                                               threshold=variable_dictionary.get(variable).get(thresholdname), 
                                               thresholdtype=thresholdsign)

    for i in range(threshold_data.shape[0]):

        count = np.sum(threshold_data[i, :, :])/number_gridpoints
        spatial_count = np.append(spatial_count, count)

    spatial_count = spatial_count[spatial_count!=0]

    return(spatial_count)


def calculate_spatial_clusters(variable, **kwargs):
    
    clusters_array = np.empty((0, 3))
    
    for k in kwargs.keys():
        
        high = calculate_spatial_count_once(variable = 'tas', data = kwargs[k], 
                                                              thresholdname = 'high_threshold', 
                                                              thresholdsign = '>')
        low = calculate_spatial_count_once(variable = 'tas', data = kwargs[k], 
                                                              thresholdname = 'low_threshold', 
                                                              thresholdsign = '<')
        
        length_low = len(low)
        length_high = len(high)
        
        clusters_array = np.append(clusters_array,
                               np.transpose(np.array([[k]*length_high, ['High']*length_high, np.transpose(high)])), 
                               axis = 0)
        clusters_array = np.append(clusters_array,
                               np.transpose(np.array([[k]*length_low, ['Low']*length_low, np.transpose(low)])), 
                               axis = 0)

    spatial_clusters = pd.DataFrame(clusters_array, columns=['Correction Method','Threshold', 'Percent of area'])
    spatial_clusters["Percent of area"] = pd.to_numeric(spatial_clusters["Percent of area"])
    
    return(spatial_clusters)




def clusters_distribution(variable, plot_data, clustertype):

    #fig, ax = plt.subplots(1,1, figsize=(8,6))
    fig = plt.figure(figsize=(8,6))
    #fig.suptitle('This is a somewhat long figure title', fontsize=16)

    seaborn.displot(x=plot_data.keys()[2], 
                     col='Threshold',
                     data=plot_data, 
                     #kind = 'kde',
                     #common_norm = True,
                     #hist = False,
                     #kde = True,
                     #fill = False,
                     palette="colorblind",
                     hue='Correction Method',
                     element="step"
                     )
    return(fig)