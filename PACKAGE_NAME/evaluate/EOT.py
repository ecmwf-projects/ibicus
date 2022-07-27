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

# define thresholds

threshold_dictionary = {
    "frost": {
        "variable": 'tasmin',
        "variablename": '2m daily minimum air temperature (K)',
        "value": 273.15,
        "threshold_sign": 'lower',
        "name": 'Frost days'
    },
    "mean_warm_day": {
        "variable": 'tas',
        "variablename": '2m daily mean air temperature (K)',
        "value": 295,
        "threshold_sign": 'higher',
        "name": 'Warm days (mean)'
    },
    "mean_cold_day": {
        "variable": 'tas',
        "variablename": '2m daily mean air temperature (K)',
        "value": 273,
        "threshold_sign": 'lower',
        "name": 'Cold days (mean)'
    },
        "dry": {
        "variable": 'pr',
        "variablename": 'Precipitation',
        "value": 0.000001,
        "threshold_sign": 'lower',
        "name": 'Dry days (mean)'
    }
}



def calculate_matrix(dataset, thresholdname):
    
    """
    Converts data into 1-0 matrix of same dimensions, 1 if the value at this time and location is below / above 
    a specific threshold and 0 otherwise.

    Parameters
    ----------
    dataset : dataset to be analysed, numeric entries expected
    thresholdname: name of threshold specified in the threshold dictionary
    """
    
    # TO-DO: show error if thresholdname specified and dataset don't match  
    
    thresholds = np.copy(dataset)

    if threshold_dictionary.get(thresholdname).get('threshold_sign')=='higher':

      thresholds = (thresholds > threshold_dictionary.get(thresholdname).get('value')).astype(int)

    elif threshold_dictionary.get(thresholdname).get('threshold_sign')=='lower':

      thresholds = (thresholds < threshold_dictionary.get(thresholdname).get('value')).astype(int)
      
    else:
      print('Invalid threshold type')
      
    return(thresholds)



def calculate_probability_once(data, thresholdname):
    
    """
    Calculates the probability of exceeding the specified threshold, using the function calculate_matrix to calculate array of 1-0 entries first 
      
    Parameters
    ----------
    dataset : dataset to be analysed, numeric entries expected
    thresholdname: name of threshold specified in the threshold dictionary
    """
    
    threshold_data = calculate_matrix(data, thresholdname)
    
    threshold_probability = np.zeros((threshold_data.shape[1], threshold_data.shape[2]))
    
    for i in range(threshold_data.shape[1]):
      for j in range(threshold_data.shape[2]):

        threshold_probability[i, j] = np.sum(threshold_data[:, i, j])/threshold_data.shape[0]
        
    return(threshold_probability)   



def calculate_marginal_bias(thresholds, data_obs, **kwargs):
    
    threshold_obs = {}
    
    for i in range(len(thresholds)):
    
        threshold_obs[i] = calculate_probability_once(data = data_obs, 
                                                  thresholdname = thresholds[i])

    bias_array = np.empty((0, 3))
    length = len(np.ndarray.flatten(data_obs[1, : ,:]))
    
    for k in kwargs.keys():
        for i in range(len(thresholds)):
        
            bias = threshold_obs[i]*365 - 365*calculate_probability_once(data = kwargs[k], 
                                                          thresholdname = thresholds[i])

            bias_array = np.append(bias_array,
                                  np.transpose(np.array([[k]*length, 
                                                         [threshold_dictionary.get(thresholds[i]).get('name')]*length, 
                                                         np.transpose(np.ndarray.flatten(bias))])), 
                                  axis = 0)



    plot_data = pd.DataFrame(bias_array, columns=['Correction Method','Metric', 'Mean bias (days/year)'])
    plot_data["Mean bias (days/year)"] = pd.to_numeric(plot_data["Mean bias (days/year)"])
    
    return(plot_data)



def plot_distribution_mean_bias_days(variable, dataset):
    
    fig = plt.figure(figsize=(10, 6))
    seaborn.violinplot(y='Mean bias (days/year)', x='Metric', 
                   data=dataset, 
                   palette="colorblind",
                   hue='Correction Method')


    fig.suptitle('Distribution of exceedance over threshold bias \n Variable: {}'.format(variable_dictionary.get(variable).get('name')))
    return(fig)







def plot_2d_distribution_mean_bias_days(thresholdname, data_obs, **kwargs):
    
    threshold_obs = calculate_probability_once(data = data_obs, thresholdname = thresholdname)
    bias = {}
    
    for k in kwargs.keys():
        
        bias[k] = threshold_obs*365 - 365*calculate_probability_once(data = kwargs[k], 
                                                          thresholdname = thresholdname)
            
    axis_max = max(abs(max(np.ndarray.flatten(np.vstack(list(chain(*bias.values())))))), 
                   abs(min(np.ndarray.flatten(np.vstack(list(chain(*bias.values())))))))
    axis_min = -axis_max
    
    fig_width = 6*len(kwargs.keys())
    fig, ax = plt.subplots(1, len(kwargs.keys()), figsize=(fig_width, 5))
    fig.suptitle("{} - bias in days/year".format(threshold_dictionary.get(thresholdname).get('name')))
    
    i=0
    for k in kwargs.keys():
        
        plot = ax[i].imshow(bias[k], cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
        ax[i].set_title('{}'.format(k))
        fig.colorbar(plot, ax=ax[i])
        i = i+1
        
    return(fig)







def compute_spell_length(thresholdname, min_length, **kwargs):
    
    spell_length_array = np.empty((0, 3))
    
    for k in kwargs.keys():
        
        threshold_data = calculate_matrix(kwargs[k],  thresholdname)
        spell_length = np.array([])
        
        for i in range(threshold_data.shape[1]):
          for j in range(threshold_data.shape[2]):
            N=0
            for t in range(threshold_data.shape[0]):
              if threshold_data[t, i, j]==1:
                  N=N+1
              elif (threshold_data[t, i, j]==0) and (N!=0):
                  spell_length = np.append(spell_length, N)
                  N=0
  
        spell_length = spell_length[spell_length>min_length]
        
        spell_length_array = np.append(spell_length_array,
                               np.transpose(np.array([[k]*len(spell_length), 
                                                      [threshold_dictionary.get(thresholdname).get('name')]*len(spell_length), 
                                                      np.transpose(spell_length)])), 
                               axis = 0)
        
    plot_data = pd.DataFrame(spell_length_array, columns=['Correction Method','Threshold', 'Spell length'])
    plot_data["Spell length"] = pd.to_numeric(plot_data["Spell length"])
    
    return(plot_data)
        
        

def calculate_spatiotemporal_clusters(thresholdname, **kwargs):
    
    clusters_array = np.empty((0, 3))
    
    for k in kwargs.keys():
        
        threshold_data = calculate_matrix(dataset = kwargs[k], 
                                thresholdname = thresholdname)
        threshold_data_lw, threshold_data_num = measurements.label(threshold_data)
        area = measurements.sum(threshold_data, threshold_data_lw, index=arange(threshold_data_lw.max() + 1))

        clusters_array = np.append(clusters_array,
                               np.transpose(np.array([[k]*len(area), 
                                            [threshold_dictionary.get(thresholdname).get('name')]*len(area), 
                                            np.transpose(area)])), axis = 0)

    spatiotemporal_clusters = pd.DataFrame(clusters_array, columns=['Correction Method','Threshold', 'Cluster size'])
    spatiotemporal_clusters["Cluster size"] = pd.to_numeric(spatiotemporal_clusters["Cluster size"])
    
    return(spatiotemporal_clusters)



def calculate_spatial_clusters(thresholdname, **kwargs):
    
    clusters_array = np.empty((0, 3))
    
    for k in kwargs.keys():
        
        spatial_count = np.array([])

        number_gridpoints = kwargs[k].shape[1]*kwargs[k].shape[2]

        threshold_data = calculate_matrix(dataset=kwargs[k], thresholdname=thresholdname)

        for i in range(threshold_data.shape[0]):

            count = np.sum(threshold_data[i, :, :])/number_gridpoints
            spatial_count = np.append(spatial_count, count)

        spatial_count = spatial_count[spatial_count!=0]
        
        clusters_array = np.append(clusters_array,
                               np.transpose(np.array([[k]*len(spatial_count), 
                                                      [threshold_dictionary.get(thresholdname).get('name')]*len(spatial_count), 
                                                      np.transpose(spatial_count)])), axis = 0)

    spatial_clusters = pd.DataFrame(clusters_array, columns=['Correction Method','Threshold', 'Percent of area'])
    spatial_clusters["Percent of area"] = pd.to_numeric(spatial_clusters["Percent of area"])
    
    return(spatial_clusters)



def plot_clusters_distribution(thresholdname, plot_data, clustertype):

    seaborn.set_style('white')
    p = seaborn.displot(x=plot_data.keys()[2], 
                     data=plot_data, 
                     kind = 'kde',
                     palette="colorblind",
                     hue='Correction Method'
                     )
    p.fig.subplots_adjust(top=0.9)
    p.fig.suptitle("{} - {} distribution".format(threshold_dictionary.get(thresholdname).get('name'), clustertype))
