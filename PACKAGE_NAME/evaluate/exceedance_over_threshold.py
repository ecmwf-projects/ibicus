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



def EOT_calculate_matrix(dataset, threshold, thresholdtype):
    
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



def EOT_calculate_probability_once(variable, data, thresholdname, thresholdsign):
    
    threshold_data = EOT_calculate_matrix(dataset=data, 
                                           threshold=variable_dictionary.get(variable).get(thresholdname), 
                                           thresholdtype=thresholdsign)
    
    threshold_probability = np.zeros((threshold_data.shape[1], threshold_data.shape[2]))
    
    for i in range(threshold_data.shape[1]):
      for j in range(threshold_data.shape[2]):

        threshold_probability[i, j] = np.sum(threshold_data[:, i, j])/threshold_data.shape[0]
        
    return(threshold_probability)   




def EOT_probability(variable, data_obs, **kwargs):
    
    high_bias = {}
    low_bias = {}
    
    high_obs = EOT_calculate_probability_once(variable = variable, 
                                              data = data_obs, 
                                              thresholdname = 'high_threshold', 
                                              thresholdsign = '>')
    low_obs = EOT_calculate_probability_once(variable = variable, 
                                             data = data_obs, 
                                             thresholdname = 'low_threshold', 
                                             thresholdsign = '<')
    
    for k in kwargs.keys():
        
        high_bias[k] = high_obs*365 - 365*EOT_calculate_probability_once(variable = variable, 
                                                          data = kwargs[k], 
                                                          thresholdname = 'high_threshold', 
                                                          thresholdsign = '>')
        low_bias[k] = low_obs*365 - 365*EOT_calculate_probability_once(variable = variable, 
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


    plot_data = pd.DataFrame(bias_array, columns=['Correction Method','Exceedance over threshold', 'Mean bias (days/year)'])
    plot_data["Mean bias (days/year)"] = pd.to_numeric(plot_data["Mean bias (days/year)"])
    
    return(plot_data)



def EOT_mean_bias_days_violinplot(variable, dataset):
    
    fig = plt.figure(figsize=(8, 6))
    seaborn.violinplot(y='Mean bias (days/year)', x='Exceedance over threshold', 
                   data=dataset, 
                   palette="colorblind",
                   hue='Correction Method')


    fig.suptitle('{} - Bias \n (high threshold = {}{}, low threshold = {}{})'.format(variable_dictionary.get(variable).get('name'), 
                                                                              variable_dictionary.get(variable).get('high_threshold'),
                                                                              variable_dictionary.get(variable).get('unit'),    
                                                                              variable_dictionary.get(variable).get('low_threshold'),
                                                                              variable_dictionary.get(variable).get('unit')))

    return fig




def clusters_violinplot(variable, boxplot_data, clustertype, columnname):
    
    fig = plt.figure(figsize=(8, 6))
    seaborn.violinplot(y=columnname, x='Exceedance over threshold', 
                   data=boxplot_data, 
                   palette="colorblind",
                   hue='Correction Method')


    fig.suptitle('{} - {} clusters \n (high threshold = {}{}, low threshold = {}{})'.format(variable_dictionary.get(variable).get('name'),
                                                                                            clustertype,
                                                                              variable_dictionary.get(variable).get('high_threshold'),
                                                                              variable_dictionary.get(variable).get('unit'),    
                                                                              variable_dictionary.get(variable).get('low_threshold'),
                                                                              variable_dictionary.get(variable).get('unit')))

    return fig



def EOT_mean_bias_days_2dplot(variable, threshold_prob_obs, threshold_prob_raw, threshold_prob_bc, thresholdname, thresholdsign, name_BC):
    
    axis_max = max(np.concatenate((np.ndarray.flatten(threshold_prob_obs*100), 
                                   np.ndarray.flatten(threshold_prob_raw*100),
                                   np.ndarray.flatten(threshold_prob_bc*100),
                                   )))
    
    axis_min = min(np.concatenate((np.ndarray.flatten(threshold_prob_obs*100), 
                                   np.ndarray.flatten(threshold_prob_raw*100),
                                   np.ndarray.flatten(threshold_prob_bc*100),
                                   )))
    
    
    
    fig, ax = plt.subplots(1,3, figsize=(14,4))

    fig.suptitle("Bias in probability of {} {} {}".format(variable_dictionary.get(variablename).get('name'),
                                                          thresholdsign,
                                                          thresholdname))
    
    plot1 = ax[0].imshow(threshold_prob_obs*100, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
    ax[0].set_title('Observed')
    fig.colorbar(plot1, ax=ax[0])
    
    plot2 = ax[1].imshow(threshold_prob_raw*100, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
    ax[1].set_title('Raw CM')
    fig.colorbar(plot2, ax=ax[1])
    
    plot3 = ax[2].imshow(threshold_prob_bc*100, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
    ax[2].set_title('Debiased CM - {}'.format(name_BC))
    fig.colorbar(plot3, ax=ax[2])
    
    return(fig)



def EOT_compute_spell_length(variable, data, thresholdname, thresholdsign, min_length):

  spell_length = np.array([])
  
  threshold_data = EOT_calculate_matrix(dataset=data, 
                                           threshold=variable_dictionary.get(variable).get(thresholdname), 
                                           thresholdtype=thresholdsign)

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
 
  return(spell_length)



def EOT_spell_length_plot(variable, data_obs, data_raw, data_bc, BC_name, thresholdsign, thresholdtype, number_bins=9):
    
    fig = plt.figure(figsize=(8, 6))
    
    plt.hist(data_obs, bins=number_bins, label='Observed', alpha=0.5)
    plt.hist(data_raw, bins=number_bins, label='Raw', alpha=0.5)
    plt.hist(data_bc, bins=number_bins, label=BC_name, alpha=0.5)
    plt.xlabel("Spell length")
    plt.ylabel("Number of occurences")
    plt.title('Marginal spell length \n {} {} {} {}'.format(variable_dictionary.get(variable).get('name'), 
                                                         thresholdsign,
                                                         variable_dictionary.get(variable).get(thresholdtype),
                                                         variable_dictionary.get(variable).get('unit')))
    plt.legend()
    
    return(fig)





def EOT_spell_length_violinplot(variable, data_obs, data_raw, data_bc, name_BC, minimum_spell_length):
    
    high_obs_spell_length = EOT_compute_spell_length(variable = variable, data = data_obs, 
                                                 thresholdname = 'high_threshold', thresholdsign = '>', 
                                                 min_length = minimum_spell_length)

    low_obs_spell_length = EOT_compute_spell_length(variable = variable, data = data_obs, 
                                                 thresholdname = 'low_threshold', thresholdsign = '<', 
                                                 min_length = minimum_spell_length)

    high_raw_spell_length = EOT_compute_spell_length(variable = variable, data = data_raw, 
                                                 thresholdname = 'high_threshold', thresholdsign = '>', 
                                                 min_length = minimum_spell_length)

    low_raw_spell_length = EOT_compute_spell_length(variable = variable, data = data_raw, 
                                                 thresholdname = 'low_threshold', thresholdsign = '<', 
                                                 min_length = minimum_spell_length)

    high_bc_spell_length = EOT_compute_spell_length(variable = variable, data = data_bc, 
                                                 thresholdname = 'high_threshold', thresholdsign = '>', 
                                                 min_length = minimum_spell_length)

    low_bc_spell_length = EOT_compute_spell_length(variable = variable, data = data_bc, 
                                                 thresholdname = 'low_threshold', thresholdsign = '<', 
                                                 min_length = minimum_spell_length)
    
    print('High {} spell-length - Observations compared to Raw CM \n KS-test \n'.format(variable_dictionary.get(variable).get('name')), 
          'Maximum distance between cdf functions:',
          scipy.stats.ks_2samp(high_obs_spell_length, high_raw_spell_length)[0],
      "\n p-value =", scipy.stats.ks_2samp(high_obs_spell_length, high_raw_spell_length)[1])
    if(scipy.stats.ks_2samp(high_obs_spell_length, high_raw_spell_length)[1] >= 0.05):
        print('There is an absence of evidence to conclude that the two samples were not drawn from the same distribution. \n')
    else:
        print('There is significant evidence that the two samples were not drawn from the same distribution. \n')
        
    print('Low {} spell-length - Observations compared to Raw CM \n KS-test \n'.format(variable_dictionary.get(variable).get('name')), 
          'Maximum distance between cdf functions:',
          scipy.stats.ks_2samp(low_obs_spell_length, low_raw_spell_length)[0],
      "\n p-value =", scipy.stats.ks_2samp(low_obs_spell_length, low_raw_spell_length)[1])
    if(scipy.stats.ks_2samp(low_obs_spell_length, low_raw_spell_length)[1] >= 0.05):
        print('There is an absence of evidence to conclude that the two samples were not drawn from the same distribution. \n')
    else:
        print('There is significant evidence that the two samples were not drawn from the same distribution. \n')
        
    print('High {} spell-length - Observations compared to debiased CM ({}) \n KS-test \n'.format(variable_dictionary.get(variable).get('name'), name_BC), 
          'Maximum distance between cdf functions:',
          scipy.stats.ks_2samp(high_obs_spell_length, high_bc_spell_length)[0],
      "\n p-value =", scipy.stats.ks_2samp(high_obs_spell_length, high_bc_spell_length)[1])
    if(scipy.stats.ks_2samp(high_obs_spell_length, high_bc_spell_length)[1] >= 0.05):
        print('There is an absence of evidence to conclude that the two samples were not drawn from the same distribution. \n')
    else:
        print('There is significant evidence that the two samples were not drawn from the same distribution. \n')
        
    print('Low {} spell-length - Observations compared to deabised CM ({}) \n KS-test \n'.format(variable_dictionary.get(variable).get('name'), name_BC), 
          'Maximum distance between cdf functions:',
          scipy.stats.ks_2samp(low_obs_spell_length, low_bc_spell_length)[0],
      "\n p-value =", scipy.stats.ks_2samp(low_obs_spell_length, low_bc_spell_length)[1])
    if(scipy.stats.ks_2samp(low_obs_spell_length, low_bc_spell_length)[1] >= 0.05):
        print('There is an absence of evidence to conclude that the two samples were not drawn from the same distribution. \n')
    else:
        print('There is significant evidence that the two samples were not drawn from the same distribution. \n')
        
    
    
    array1 = np.transpose(np.array([['Observed']*len(np.ndarray.flatten(high_obs_spell_length)), 
                                    ['High']*len(np.ndarray.flatten(high_obs_spell_length)), 
                                    np.transpose(np.ndarray.flatten(high_obs_spell_length))]))
    array2 = np.transpose(np.array([['Observed']*len(np.ndarray.flatten(low_obs_spell_length)), 
                                    ['Low']*len(np.ndarray.flatten(low_obs_spell_length)), 
                                    np.transpose(np.ndarray.flatten(low_obs_spell_length))]))
    array3 = np.transpose(np.array([['Raw']*len(np.ndarray.flatten(high_raw_spell_length)), 
                                    ['High']*len(np.ndarray.flatten(high_raw_spell_length)), 
                                    np.transpose(np.ndarray.flatten(high_raw_spell_length))]))
    array4 = np.transpose(np.array([['Raw']*len(np.ndarray.flatten(low_raw_spell_length)), 
                                    ['Low']*len(np.ndarray.flatten(low_raw_spell_length)), 
                                    np.transpose(np.ndarray.flatten(low_raw_spell_length))]))
    array5 = np.transpose(np.array([[name_BC]*len(np.ndarray.flatten(high_bc_spell_length)), 
                                    ['High']*len(np.ndarray.flatten(high_bc_spell_length)), 
                                    np.transpose(np.ndarray.flatten(high_bc_spell_length))]))
    array6 = np.transpose(np.array([[name_BC]*len(np.ndarray.flatten(low_bc_spell_length)), 
                                    ['Low']*len(np.ndarray.flatten(low_bc_spell_length)), 
                                    np.transpose(np.ndarray.flatten(low_bc_spell_length))]))
    
    arrays = np.concatenate((array1, array2, array3, array4, array5, array6))

    boxplot_data = pd.DataFrame(arrays, columns=['Correction Method','Exceedance over threshold', 'Spell length (days)'])
    boxplot_data["Spell length (days)"] = pd.to_numeric(boxplot_data["Spell length (days)"])

    fig = plt.figure(figsize=(8, 6))
    seaborn.violinplot(y='Spell length (days)', x='Exceedance over threshold', 
                   data=boxplot_data, 
                   palette="colorblind",
                   hue='Correction Method')


    fig.suptitle('{} - Spell length \n (high threshold = {}{}, low threshold = {}{})'.format(variable_dictionary.get(variable).get('name'), 
                                                                              variable_dictionary.get(variable).get('high_threshold'),
                                                                              variable_dictionary.get(variable).get('unit'),    
                                                                              variable_dictionary.get(variable).get('low_threshold'),
                                                                              variable_dictionary.get(variable).get('unit')))




    return fig

    


def calculate_spatiotemporal_clusters(variable, name_BC, data_obs, data_raw, data_bc):

    high_obs = EOT_calculate_matrix(dataset = data_obs, 
                                               threshold = variable_dictionary.get(variable).get('high_threshold'),
                                               thresholdtype = '>')
    high_obs_lw, high_obs_num = measurements.label(high_obs)
    high_obs_area = measurements.sum(high_obs, high_obs_lw, index=arange(high_obs_lw.max() + 1))
    
    
    low_obs = EOT_calculate_matrix(dataset = data_obs, 
                                               threshold = variable_dictionary.get(variable).get('low_threshold'),
                                               thresholdtype = '<')
    low_obs_lw, low_obs_num = measurements.label(low_obs)
    low_obs_area = measurements.sum(low_obs, low_obs_lw, index=arange(low_obs_lw.max() + 1))
    
    
    high_raw = EOT_calculate_matrix(dataset = data_raw, 
                                               threshold = variable_dictionary.get(variable).get('high_threshold'),
                                               thresholdtype = '>')
    high_raw_lw, high_raw_num = measurements.label(high_raw)
    high_raw_area = measurements.sum(high_raw, high_raw_lw, index=arange(high_raw_lw.max() + 1))
    
    low_raw = EOT_calculate_matrix(dataset = data_raw, 
                                               threshold = variable_dictionary.get(variable).get('low_threshold'),
                                               thresholdtype = '<')
    low_raw_lw, low_raw_num = measurements.label(low_raw)
    low_raw_area = measurements.sum(low_raw, low_raw_lw, index=arange(low_raw_lw.max() + 1))
    
    high_bc = EOT_calculate_matrix(dataset = data_bc, 
                                               threshold = variable_dictionary.get(variable).get('high_threshold'),
                                               thresholdtype = '>')
    high_bc_lw, high_bc_num = measurements.label(high_bc)
    high_bc_area = measurements.sum(high_bc, high_bc_lw, index=arange(high_bc_lw.max() + 1)) 
    
    low_bc = EOT_calculate_matrix(dataset = data_bc, 
                                               threshold = variable_dictionary.get(variable).get('low_threshold'),
                                               thresholdtype = '<')
    low_bc_lw, low_bc_num = measurements.label(low_bc)
    low_bc_area = measurements.sum(low_bc, low_bc_lw, index=arange(low_bc_lw.max() + 1))
    
    array1 = np.transpose(np.array([['Observed']*len(np.ndarray.flatten(high_obs_area)), 
                                    ['High']*len(np.ndarray.flatten(high_obs_area)), 
                                    np.transpose(np.ndarray.flatten(high_obs_area))]))
    array2 = np.transpose(np.array([['Observed']*len(np.ndarray.flatten(low_obs_area)), 
                                    ['Low']*len(np.ndarray.flatten(low_obs_area)), 
                                    np.transpose(np.ndarray.flatten(low_obs_area))]))
    array3 = np.transpose(np.array([['Raw']*len(np.ndarray.flatten(high_raw_area)), 
                                    ['High']*len(np.ndarray.flatten(high_raw_area)), 
                                    np.transpose(np.ndarray.flatten(high_raw_area))]))
    array4 = np.transpose(np.array([['Raw']*len(np.ndarray.flatten(low_raw_area)), 
                                    ['Low']*len(np.ndarray.flatten(low_raw_area)), 
                                    np.transpose(np.ndarray.flatten(low_raw_area))]))
    array5 = np.transpose(np.array([[name_BC]*len(np.ndarray.flatten(high_bc_area)), 
                                    ['High']*len(np.ndarray.flatten(high_bc_area)), 
                                    np.transpose(np.ndarray.flatten(high_bc_area))]))
    array6 = np.transpose(np.array([[name_BC]*len(np.ndarray.flatten(low_bc_area)), 
                                    ['Low']*len(np.ndarray.flatten(low_bc_area)), 
                                    np.transpose(np.ndarray.flatten(low_bc_area))]))
    
    arrays = np.concatenate((array1, array2, array3, array4, array5, array6))

    spatiotemporal_clusters = pd.DataFrame(arrays, columns=['Correction Method','Exceedance over threshold', 'Cluster size'])
    spatiotemporal_clusters["Cluster size"] = pd.to_numeric(spatiotemporal_clusters["Cluster size"])
    
    return(spatiotemporal_clusters)



def calculate_spatial_count_once(variable, data, thresholdname, thresholdsign):
    
    spatial_count = np.array([])

    number_gridpoints = data.shape[1]*data.shape[2]

    threshold_data = EOT_calculate_matrix(dataset=data, 
                                               threshold=variable_dictionary.get(variable).get(thresholdname), 
                                               thresholdtype=thresholdsign)

    for i in range(threshold_data.shape[0]):

        count = np.sum(threshold_data[i, :, :])/number_gridpoints
        spatial_count = np.append(spatial_count, count)

    spatial_count = spatial_count[spatial_count!=0]

    return(spatial_count)


def EOT_spatial_compounding(variable, name_BC, data_obs, data_raw, data_bc):
    
    high_obs_spatial_count = calculate_spatial_count_once(variable = 'tas', data = data_obs, 
                                                          thresholdname = 'high_threshold', 
                                                          thresholdsign = '>')
    
    low_obs_spatial_count = calculate_spatial_count_once(variable = 'tas', data = data_obs, 
                                                          thresholdname = 'low_threshold', 
                                                          thresholdsign = '<')
    
    high_raw_spatial_count = calculate_spatial_count_once(variable = 'tas', data = data_raw, 
                                                          thresholdname = 'high_threshold', 
                                                          thresholdsign = '>')
    
    low_raw_spatial_count = calculate_spatial_count_once(variable = 'tas', data = data_raw, 
                                                          thresholdname = 'low_threshold', 
                                                          thresholdsign = '<')
    
    high_bc_spatial_count = calculate_spatial_count_once(variable = 'tas', data = data_bc, 
                                                          thresholdname = 'high_threshold', 
                                                          thresholdsign = '>')
    
    low_bc_spatial_count = calculate_spatial_count_once(variable = 'tas', data = data_bc, 
                                                          thresholdname = 'low_threshold', 
                                                          thresholdsign = '<')

    array1 = np.transpose(np.array([['Observed']*len(np.ndarray.flatten(high_obs_spatial_count)), 
                                    ['High']*len(np.ndarray.flatten(high_obs_spatial_count)), 
                                    np.transpose(np.ndarray.flatten(high_obs_spatial_count))]))
    array2 = np.transpose(np.array([['Observed']*len(np.ndarray.flatten(low_obs_spatial_count)), 
                                    ['Low']*len(np.ndarray.flatten(low_obs_spatial_count)), 
                                    np.transpose(np.ndarray.flatten(low_obs_spatial_count))]))
    array3 = np.transpose(np.array([['Raw']*len(np.ndarray.flatten(high_raw_spatial_count)), 
                                    ['High']*len(np.ndarray.flatten(high_raw_spatial_count)), 
                                    np.transpose(np.ndarray.flatten(high_raw_spatial_count))]))
    array4 = np.transpose(np.array([['Raw']*len(np.ndarray.flatten(low_raw_spatial_count)), 
                                    ['Low']*len(np.ndarray.flatten(low_raw_spatial_count)), 
                                    np.transpose(np.ndarray.flatten(low_raw_spatial_count))]))
    array5 = np.transpose(np.array([[name_BC]*len(np.ndarray.flatten(high_bc_spatial_count)), 
                                    ['High']*len(np.ndarray.flatten(high_bc_spatial_count)), 
                                    np.transpose(np.ndarray.flatten(high_bc_spatial_count))]))
    array6 = np.transpose(np.array([[name_BC]*len(np.ndarray.flatten(low_bc_spatial_count)), 
                                    ['Low']*len(np.ndarray.flatten(low_bc_spatial_count)), 
                                    np.transpose(np.ndarray.flatten(low_bc_spatial_count))]))
    
    arrays = np.concatenate((array1, array2, array3, array4, array5, array6))

    spatial_clusters = pd.DataFrame(arrays, columns=['Correction Method','Exceedance over threshold', 'Percent of area'])
    spatial_clusters["Percent of area"] = pd.to_numeric(spatial_clusters["Percent of area"])
    
    return(spatial_clusters)

