# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy
import scipy.stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

import libpysal
from libpysal.weights import lat2W
from esda.moran import Moran



standard_variables_isimip = {
        "tas": {
        "lower_bound": -np.inf,
        "lower_threshold": -np.inf,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.norm,
        "trend_preservation": "additive",
        "detrending": True,
        "name": '2m daily mean air temperature (K)',
        "high_threshold": 295,
        "low_threshold": 273
    },
    "pr": {
        "lower_bound": 0,
        "lower_threshold": 0.1 / 86400,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.gamma,
        "trend_preservation": "mixed",
        "detrending": False,
        "name": 'Total precipitation (m/day)',
        "high_threshold": 0.0004,
        "low_threshold": 0.00001
    }
}


# High-level functions that execute chunks of the analysis

#def assumptions_evaluation(stationarity=TRUE, goodness_of_fit=TRUE):

#def marginal_evaluation(histogram=TRUE, 2d_marginal_bias=TRUE):
    
#def non_marginal_evaluation(temporal=TRUE, spatial=TRUE, multivariate=TRUE):     

#def threshold_exceedance_evaluation():

    # Individual evaluation steps that can be turned on or of in high-level functions

#def histogram_eval():
        





# Functions that actually do the work

def timeseries_gridpoint_plot(dataset1, dataset2, label1, label2, variablename, long, lat):
    
    # input are two three-dimensional datasets
    fig = plt.figure()
    plt.plot(dataset1[:, long, lat], label=label1)
    plt.plot(dataset2[:, long,lat], label=label2)
    plt.xlabel("timesteps")
    plt.ylabel(variablename)
    plt.title('Timeseries {}, grid point ({}, {})'.format(variablename, long, lat))
    plt.legend()
    return fig


def timeseries_plot(dataset1, dataset2, label1, label2, titlename):

  fig = plt.figure()
  plt.plot(dataset1, label=label1)
  plt.plot(dataset2, label=label2)
  plt.xlabel("timesteps")
  plt.ylabel(titlename)
  plt.title(titlename)
  plt.legend()
  #plt.show()
  return fig
  
  
def histogram_plot(variable, dataset1, dataset2, label1, label2):
    
  # requires one dimensional array as input
  fig = plt.figure()
  plt.hist(dataset1, bins=100, alpha=0.5, label=label1)
  plt.hist(dataset2, bins=100, alpha=0.5, label=label2) 
  plt.xlabel(standard_variables_isimip.get(variable).get('name'))
  plt.ylabel("Number of occurences")
  plt.title("Histogram of {}, entire area".format(standard_variables_isimip.get(variable).get('name')))
  plt.legend()
  #plt.show()
  return fig


def marginal_bias_plot(dataset1, dataset2, variablename):

  marginal_mean1 = np.mean(dataset1, axis=0)
  marginal_5pc1 = np.quantile(dataset1, 0.05, axis=0)
  marginal_95pc1 = np.quantile(dataset1, 0.95, axis=0)

  marginal_mean2 = np.mean(dataset2, axis=0)
  marginal_5pc2 = np.quantile(dataset2, 0.05, axis=0)
  marginal_95pc2 = np.quantile(dataset2, 0.95, axis=0)

  # plot

  fig, ax = plt.subplots(1,3, figsize=(14,4))

  fig.suptitle('{} - Bias'.format(standard_variables_isimip.get(variablename).get('name')))

  plot1 = ax[0].imshow(100*(marginal_mean1 - marginal_mean2)/marginal_mean1, cmap=plt.get_cmap('coolwarm'))
  ax[0].set_title('Mean - percentage bias')
  fig.colorbar(plot1, ax=ax[0])

  plot2 = ax[1].imshow(100*(marginal_5pc1 - marginal_5pc2)/marginal_5pc1, cmap=plt.get_cmap('coolwarm'))
  ax[1].set_title('5th percentile - percentage bias')
  fig.colorbar(plot2, ax=ax[1])

  plot3 = ax[2].imshow(100*(marginal_95pc1 - marginal_95pc2)/marginal_95pc1, cmap=plt.get_cmap('coolwarm'))
  ax[2].set_title('95th percentile - percentage bias')
  fig.colorbar(plot3, ax=ax[2])

  #plt.show()
  return fig
  
  
  
def goodness_of_fit(dataset, variablename, data_type):

    fit = standard_variables_isimip.get(variablename).get('distribution').fit(dataset)
    q = standard_variables_isimip.get(variablename).get('distribution').cdf(dataset, *fit)
    
    q_normal = norm.ppf(q)
    
    fig, ax = plt.subplots(1,3, figsize=(14,4))

    fig.suptitle('Goodness of fit evaluation - {} {}'.format(standard_variables_isimip.get(variablename).get('name'), data_type))
    
    x = range(0, len(q))
    ax[0].plot(x[0:8000], q[0:8000])
    ax[0].set_title('Quantile Residuals')
    
    plot_acf(q, lags=8000, ax=ax[1])
    ax[1].set_title('ACF {}')
    
    sm.qqplot(q_normal, line='45', ax=ax[2])
    ax[2].set_title('QQ Plot {}')
  


def threshold_matrix(dataset, threshold, thresholdtype):
    
    thresholds = np.copy(dataset)

    if thresholdtype=='>':

      thresholds = (thresholds > threshold).astype(int)

    elif thresholdtype=='<':

      thresholds = (thresholds < threshold).astype(int)
      
    else:
      print('Invalid threshold type')
      
    return(thresholds)


def probability_extremes(threshold_data):
    
    threshold_probability = np.zeros((threshold_data.shape[1], threshold_data.shape[2]))
    
    for i in range(threshold_data.shape[1]):
      for j in range(threshold_data.shape[2]):

        threshold_probability[i, j] = np.sum(threshold_data[:, i, j])/threshold_data.shape[0]
        
    return(threshold_probability)    


def mean_bias_days_extremes(extreme_probability_data1, extreme_probability_data2):
    
    bias_days = (extreme_probability_data1*365 - extreme_probability_data2*365* extreme_probability_data2.shape[0])
    mean_bias_days = np.mean(bias_days)
    return(mean_bias_days)



def plot_bias_marginal_extremes(variablename, threshold_probability1, threshold_probability2, label1, label2, highlow):
    
    fig, ax = plt.subplots(1,3, figsize=(14,4))

    fig.suptitle("Bias in probability of {} {}".format(highlow, standard_variables_isimip.get(variablename).get('name')))
    
    plot1 = ax[0].imshow(threshold_probability1*100, cmap=plt.get_cmap('coolwarm'))
    ax[0].set_title(label1)
    fig.colorbar(plot1, ax=ax[0])
    
    plot2 = ax[1].imshow(threshold_probability2*100, cmap=plt.get_cmap('coolwarm'))
    ax[1].set_title(label2)
    fig.colorbar(plot2, ax=ax[1])
    
    plot3 = ax[2].imshow(100*(threshold_probability1 - threshold_probability2), cmap=plt.get_cmap('coolwarm'))
    ax[2].set_title('Difference')
    fig.colorbar(plot3, ax=ax[2])
    
    return(fig)



def compute_spell_length(threshold_data):

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
 
  return(spell_length)



def spell_length_plot(variablename, spell_length_data1, spell_length_data2, label1, label2, highlow, number_bins=9):
    
    plt.hist(spell_length_data1, bins=number_bins, label=label1, alpha=0.5)
    plt.hist(spell_length_data2, bins=number_bins, label=label2, alpha=0.5)
    plt.xlabel("Spell length")
    plt.ylabel("Number of occurences")
    plt.title('{} {} days spell length distribution'.format(highlow, standard_variables_isimip.get(variablename).get('name')))
    plt.legend()
    plt.show()
    



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


def plot_moransi_spatial(moransi_data1, moransi_data2, label1, label2, titlename):
    
    plt.plot(moransi_data1, label=label1)
    plt.plot(moransi_data2, label=label2)
    plt.legend()
    plt.title(titlename)
    plt.show()  

