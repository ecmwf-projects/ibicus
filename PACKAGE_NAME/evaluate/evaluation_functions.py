# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy
import scipy.stats

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





# marginal evaluation

def histogram_plot(variable, dataset_obs, dataset_cm, dataset_cm_bc, bin_number=100):
    
  # requires one dimensional array as input
  fig, ax = plt.subplots(1,2, figsize=(12,5))
    
  fig.suptitle("{}".format(variable_dictionary.get(variable).get('name')))
  
  ax[0].hist(dataset_obs, bins=bin_number, alpha=0.5, label='Observed')
  ax[0].hist(dataset_cm, bins=bin_number, alpha=0.5, label='CM without BC') 
  ax[0].set_title("Not Bias Corrected")
  ax[0].legend()
  
  ax[1].hist(dataset_obs, bins=bin_number, alpha=0.5, label='Observed')
  ax[1].hist(dataset_cm_bc, bins=bin_number, alpha=0.5, label='CM with BC') 
  ax[1].set_title("Bias Corrected")
  ax[1].legend()
  
  return fig


def marginal_bias_plot(variable, dataset_obs, dataset_cm, dataset_cm_bc):

  marginal_mean_obs = np.mean(dataset_obs, axis=0)
  marginal_10pc_obs = np.quantile(dataset_obs, 0.1, axis=0)
  marginal_90pc_obs = np.quantile(dataset_obs, 0.9, axis=0)

  marginal_mean_cm = np.mean(dataset_cm, axis=0)
  marginal_10pc_cm = np.quantile(dataset_cm, 0.1, axis=0)
  marginal_90pc_cm = np.quantile(dataset_cm, 0.9, axis=0)
  
  marginal_mean_cm_bc = np.mean(dataset_cm_bc, axis=0)
  marginal_10pc_cm_bc = np.quantile(dataset_cm_bc, 0.1, axis=0)
  marginal_90pc_cm_bc = np.quantile(dataset_cm_bc, 0.9, axis=0)
  
  mean_bias_cm = 100*(marginal_mean_obs - marginal_mean_cm)/marginal_mean_obs
  lowpc_bias_cm = 100*(marginal_10pc_obs - marginal_10pc_cm)/marginal_10pc_obs
  highpc_bias_cm = 100*(marginal_90pc_obs - marginal_90pc_cm)/marginal_90pc_cm
  
  mean_bias_cm_bc = 100*(marginal_mean_obs - marginal_mean_cm_bc)/marginal_mean_obs
  lowpc_bias_cm_bc = 100*(marginal_10pc_obs - marginal_10pc_cm_bc)/marginal_10pc_obs
  highpc_bias_cm_bc = 100*(marginal_90pc_obs - marginal_90pc_cm_bc)/marginal_90pc_cm
  
  
  axis_max = max(np.concatenate((np.ndarray.flatten(mean_bias_cm), 
                                 np.ndarray.flatten(lowpc_bias_cm),
                                 np.ndarray.flatten(highpc_bias_cm),
                                 np.ndarray.flatten(mean_bias_cm_bc),
                                 np.ndarray.flatten(lowpc_bias_cm_bc),
                                 np.ndarray.flatten(highpc_bias_cm_bc),
                                 )))
  
  axis_min = min(np.concatenate((np.ndarray.flatten(mean_bias_cm), 
                                 np.ndarray.flatten(lowpc_bias_cm),
                                 np.ndarray.flatten(highpc_bias_cm),
                                 np.ndarray.flatten(mean_bias_cm_bc),
                                 np.ndarray.flatten(lowpc_bias_cm_bc),
                                 np.ndarray.flatten(highpc_bias_cm_bc),
                                 )))

  fig, ax = plt.subplots(2,3, figsize=(15,10))

  fig.suptitle('{} - Bias'.format(variable_dictionary.get(variable).get('name')))

  plot1 = ax[0, 0].imshow(mean_bias_cm, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
  ax[0, 0].set_title('Mean % bias - not BC')
  fig.colorbar(plot1, ax=ax[0, 0])

  plot2 = ax[0, 1].imshow(lowpc_bias_cm, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
  ax[0, 1].set_title('10th percentile % bias - not BC')
  fig.colorbar(plot2, ax=ax[0, 1])

  plot3 = ax[0, 2].imshow(highpc_bias_cm, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
  ax[0, 2].set_title('95th percentile % bias - not BC')
  fig.colorbar(plot3, ax=ax[0, 2])
  
  plot4 = ax[1, 0].imshow(mean_bias_cm_bc, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
  ax[1, 0].set_title('Mean % bias - BC')
  fig.colorbar(plot4, ax=ax[1, 0])

  plot5 = ax[1, 1].imshow(lowpc_bias_cm_bc, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
  ax[1, 1].set_title('10th percentile % bias - BC')
  fig.colorbar(plot5, ax=ax[1, 1])

  plot6 = ax[1, 2].imshow(highpc_bias_cm_bc, cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
  ax[1, 2].set_title('95th percentile % bias - BC')
  fig.colorbar(plot6, ax=ax[1, 2])


  return fig




def marginal_bias_violinplot(variable, dataset_obs, dataset_cm, dataset_cm_bc, name_BC):

  marginal_mean_obs = np.mean(dataset_obs, axis=0)
  marginal_10pc_obs = np.quantile(dataset_obs, 0.1, axis=0)
  marginal_90pc_obs = np.quantile(dataset_obs, 0.9, axis=0)

  marginal_mean_cm = np.mean(dataset_cm, axis=0)
  marginal_10pc_cm = np.quantile(dataset_cm, 0.1, axis=0)
  marginal_90pc_cm = np.quantile(dataset_cm, 0.9, axis=0)
  
  marginal_mean_cm_bc = np.mean(dataset_cm_bc, axis=0)
  marginal_10pc_cm_bc = np.quantile(dataset_cm_bc, 0.1, axis=0)
  marginal_90pc_cm_bc = np.quantile(dataset_cm_bc, 0.9, axis=0)
  
  mean_bias_cm = 100*(marginal_mean_obs - marginal_mean_cm)/marginal_mean_obs
  lowpc_bias_cm = 100*(marginal_10pc_obs - marginal_10pc_cm)/marginal_10pc_obs
  highpc_bias_cm = 100*(marginal_90pc_obs - marginal_90pc_cm)/marginal_90pc_cm
  
  mean_bias_cm_bc = 100*(marginal_mean_obs - marginal_mean_cm_bc)/marginal_mean_obs
  lowpc_bias_cm_bc = 100*(marginal_10pc_obs - marginal_10pc_cm_bc)/marginal_10pc_obs
  highpc_bias_cm_bc = 100*(marginal_90pc_obs - marginal_90pc_cm_bc)/marginal_90pc_cm
    
  # re-organize data for grouped boxplot
  length = len(np.ndarray.flatten(mean_bias_cm))
  array1 = np.transpose(np.array([['Raw']*length, ['Mean']*length, np.transpose(np.ndarray.flatten(mean_bias_cm))]))
  array2 = np.transpose(np.array([['Raw']*length, ['10pc']*length, np.transpose(np.ndarray.flatten(lowpc_bias_cm))]))
  array3 = np.transpose(np.array([['Raw']*length, ['90pc']*length, np.transpose(np.ndarray.flatten(highpc_bias_cm))]))

  array4 = np.transpose(np.array([[name_BC]*length, ['Mean']*length, np.transpose(np.ndarray.flatten(mean_bias_cm_bc))]))
  array5 = np.transpose(np.array([[name_BC]*length, ['10pc']*length, np.transpose(np.ndarray.flatten(lowpc_bias_cm_bc))]))
  array6 = np.transpose(np.array([[name_BC]*length, ['90pc']*length, np.transpose(np.ndarray.flatten(highpc_bias_cm_bc))]))
    
  arrays = np.concatenate((array1, array2, array3, array4, array5, array6))

  boxplot_data = pd.DataFrame(arrays, columns=['Correction Method','Metric', 'Percentage bias'])
  boxplot_data["Percentage bias"] = pd.to_numeric(boxplot_data["Percentage bias"])

  fig = plt.figure(figsize=(8, 6))
  seaborn.violinplot(y='Percentage bias', x='Metric', 
                 data=boxplot_data, 
                 palette="colorblind",
                 hue='Correction Method')


  fig.suptitle('{} - Bias'.format(variable_dictionary.get(variable).get('name')))

  return fig


# trend analysis

def calculate_trend(data_hist, data_future, datatype):

    trend_additive = np.empty((0, 3))
    trend_multiplicative = np.empty((0, 3))
    
    
    for i in range(data_hist.shape[1]):
        for j in range(data_hist.shape[2]):
            
            for q in range(1, 10):
                
              trend_additive = np.append(trend_additive, 
                                         [[datatype, q/10,np.quantile(data_future[:, i, j], q/10) - np.quantile(data_hist[:, i, j], q/10)]],
                                                                     axis = 0)
              trend_multiplicative = np.append(trend_additive, 
                                         [[datatype, q/10,np.quantile(data_future[:, i, j], q/10) / np.quantile(data_hist[:, i, j], q/10)]],
                                                                     axis = 0)
                                         
    return(trend_additive, trend_multiplicative)

def calculate_full_trend_matrix(variable, data_obs_hist, data_obs_future, data_raw_hist, data_raw_future, data_bc_hist, data_bc_future, trendtype, trendtype_number):
    
    trend_obs = calculate_trend(data_obs_hist, data_obs_future, 'Observations')[trendtype_number]
    trend_raw = calculate_trend(data_raw_hist, data_raw_future, 'Raw')[trendtype_number]
    trend_bc = calculate_trend(data_bc_hist, data_bc_future, 'QM')[trendtype_number]
    
    trend = np.concatenate((trend_obs, trend_raw, trend_bc))
    
    boxplot_data = pd.DataFrame(trend, columns=['Correction Method','Quantile', trendtype])
    boxplot_data[trendtype] = pd.to_numeric(boxplot_data[trendtype])

    fig = plt.figure(figsize=(10, 6))
    seaborn.boxplot(y=trendtype, x='Quantile', 
                 data=boxplot_data, 
                 palette="colorblind",
                 hue='Correction Method')
    fig.suptitle('{} - {}'.format(variable_dictionary.get(variable).get('name'), trendtype))

    return(fig)    



# EOT analysis

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




def EOT_probability(variable, name_BC, data_obs, data_raw, data_bc):
    
    high_obs = EOT_calculate_probability_once(variable = variable, 
                                              data = data_obs, 
                                              thresholdname = 'high_threshold', 
                                              thresholdsign = '>')
    low_obs = EOT_calculate_probability_once(variable = variable, 
                                             data = data_obs, 
                                             thresholdname = 'low_threshold', 
                                             thresholdsign = '<')
    high_raw = EOT_calculate_probability_once(variable = variable, 
                                              data = data_raw, 
                                              thresholdname = 'high_threshold', 
                                              thresholdsign = '>')
    low_raw = EOT_calculate_probability_once(variable = variable, 
                                             data = data_raw, 
                                             thresholdname = 'low_threshold', 
                                             thresholdsign = '<')
    high_bc = EOT_calculate_probability_once(variable = variable, 
                                             data = data_bc, 
                                             thresholdname = 'high_threshold', 
                                             thresholdsign = '>')
    low_bc = EOT_calculate_probability_once(variable = variable, 
                                            data = data_bc, 
                                            thresholdname = 'low_threshold', 
                                            thresholdsign = '<')
    
    low_bias_raw = low_obs*365 - low_raw*365
    low_bias_bc = low_obs*365 - low_bc*365
    
    high_bias_raw = high_obs*365 - high_raw*365
    high_bias_bc = high_obs*365 - high_bc*365
      
    # re-organize data for grouped boxplot
    length = len(np.ndarray.flatten(low_bias_raw))
    
    array1 = np.transpose(np.array([['Raw']*length, ['High']*length, np.transpose(np.ndarray.flatten(high_bias_raw))]))
    array2 = np.transpose(np.array([['Raw']*length, ['Low']*length, np.transpose(np.ndarray.flatten(low_bias_raw))]))
    array3 = np.transpose(np.array([[name_BC]*length, ['High']*length, np.transpose(np.ndarray.flatten(high_bias_bc))]))
    array4 = np.transpose(np.array([[name_BC]*length, ['Low']*length, np.transpose(np.ndarray.flatten(low_bias_bc))]))
      
    arrays = np.concatenate((array1, array2, array3, array4))

    plot_data = pd.DataFrame(arrays, columns=['Correction Method','Exceedance over threshold', 'Mean bias (days/year)'])
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


def clusters_histogram(variable, plot_data, clustertype, columnname):

    fig = plt.figure(figsize=(8, 6))
    seaborn.displot(x=columnname, 
                     col='Exceedance over threshold',
                     data=plot_data, 
                     palette="colorblind",
                     hue='Correction Method',
                     kind='kde'
                     )
    return (fig)



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



def EOT_spell_length_plot(variable, data_obs, data_raw, data_bc, BC_name, number_bins=9, minimum_spell_length=2):
    
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
    
    
    
    fig1 = plt.figure(figsize=(8, 6))
    
    plt.hist(high_obs_spell_length, bins=number_bins, label='Observed', alpha=0.3)
    plt.hist(high_raw_spell_length, bins=number_bins, label='Raw', alpha=0.3)
    plt.hist(high_bc_spell_length, bins=number_bins, label=BC_name, alpha=0.3)
    plt.xlabel("Spell length")
    plt.ylabel("Number of occurences")
    plt.title('Spell length \n Days with {} > {} {}'.format(variable_dictionary.get(variable).get('name'), 
                                                         variable_dictionary.get(variable).get('high_threshold'),
                                                         variable_dictionary.get(variable).get('unit')))
    plt.legend()
    
    fig2 = plt.figure(figsize=(8, 6))
    
    plt.hist(low_obs_spell_length, bins=number_bins, label='Observed', alpha=0.5)
    plt.hist(low_raw_spell_length, bins=number_bins, label='Raw', alpha=0.5)
    plt.hist(low_bc_spell_length, bins=number_bins, label=BC_name, alpha=0.5)
    plt.xlabel("Spell length")
    plt.ylabel("Number of occurences")
    plt.title('Spell length \n Days with {} < {} {}'.format(variable_dictionary.get(variable).get('name'), 
                                                         variable_dictionary.get(variable).get('low_threshold'),
                                                         variable_dictionary.get(variable).get('unit')))
    plt.legend()
    
    return(fig1, fig2)





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


# test assumptions

def goodness_of_fit_aic(variable, dataset, distribution_name = 'default'):

    if distribution_name =='default':
        distribution = variable_dictionary.get(variable).get('distribution')
    else:
        distribution = distribution_name

    aic = np.empty((0, 3))

    for i in range(dataset.shape[1]):
        for j in range(dataset.shape[2]):

            fit = distribution.fit(dataset[:, i, j])
            
            k = len(fit)
            logLik = np.sum(distribution.logpdf(dataset[:, i, j], *fit))
            aic_location = 2*k - 2*(logLik)
            
            aic = np.append(aic,[[i,j, aic_location]],axis = 0)
            
    aic_dataframe = pd.DataFrame(aic, columns=['x', 'y', 'AIC'])

    return(aic_dataframe) 
  

def goodness_of_fit_plot_worst_fit(variable, dataset, aic, data_type, distribution_name='default', number_bins = 100):
    
    from scipy.stats import norm
    
    if distribution_name =='default':
        distribution = variable_dictionary.get(variable).get('distribution')
    else:
        distribution = distribution_name

    x_location = aic.loc[aic['AIC'].idxmax()]['x']
    y_location = aic.loc[aic['AIC'].idxmax()]['y']
    data_slice = dataset[:, int(x_location), int(y_location)]
    
    fit = distribution.fit(data_slice)
    
    fig = plt.figure(figsize=(8, 6))
  
    plt.hist(data_slice, bins=number_bins, density=True, label=data_type, alpha=0.5)
    xmin, xmax = plt.xlim()
    
    x = np.linspace(xmin, xmax, 100)
    p = distribution.pdf(x, *fit)
    
    plt.plot(x, p, 'k', linewidth=2)
    title = "{} {}, distribution = {} \n Location = ({}, {})".format(data_type,
                                                                    variable_dictionary.get(variable).get('name'),
                                                                    distribution_name,
                                                                    x_location, y_location)
    plt.title(title)
    
    return(fig)

  




def goodness_of_fit_plot_quantiles(dataset, variable, data_type, distribution_name='default'):
    
    from scipy.stats import norm
    
    if distribution_name =='default':
        distribution = variable_dictionary.get(variable).get('distribution')
    else:
        distribution = distribution_name

    fit = distribution.fit(dataset)
    q = distribution.cdf(dataset, *fit)
    
    q_normal = norm.ppf(q)
    
    fig, ax = plt.subplots(1,3, figsize=(14,4))

    fig.suptitle('{} - {}. Distribution = {}'.format(variable_dictionary.get(variable).get('name'), data_type, distribution_name))
    
    x = range(0, len(q))
    ax[0].plot(x, q)
    ax[0].set_title('Quantile Residuals - Timeseries')
    
    plot_acf(q, lags=1000, ax=ax[1])
    ax[1].set_title('Quantile Residuals - ACF')
    
    sm.qqplot(q_normal, line='45', ax=ax[2])
    ax[2].set_title('Normalized Quantile Residuals - QQ Plot')
    
    return(fig)


# correlation analysis

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



  
  




 



        


