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
import seaborn
import scipy
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

  
def plot_histogram(variable, data_obs, data_raw, bin_number=100, **kwargs): 

      """
      Plots histogram over full area covered. Expects a one-dimensional array as input, so 2d lat-long array has to be flattened using
      for example np.ndarray.flatten. This plot will be more meaningful for smaller areas.
    
      Parameters
      ----------
      variable: str, variable name is standard form (i.e. 'tas', 'pr', etc)
      data_obs : flattened entry of all observed values over the area, numeric entries expected
      data_raw : flattened entry of all 'raw', i.e. not biased corrected, climate simulations over the area, numeric entries expected
      bin_number: integer expected, number of bins of the plotted histogram, default set to 100
      **kwargs: flattened, bias corrected data sets. To be given in the form bias_correction_name = bias_corrected_dataset, the latter being
      of the same form as data_obs and data_raw, numeric entries expected.
      """
        
      number_biascorrections = len(kwargs.keys())
      figure_length = 5 + number_biascorrections*5
      plot_number = number_biascorrections +1
    
      fig, ax = plt.subplots(1,plot_number, figsize=(figure_length,5))
        
      fig.suptitle("Distribution {} over entire area".format(variable_dictionary.get(variable).get('name')))
      
      ax[0].hist(data_obs, bins=bin_number, alpha=0.5, label='Observed')
      ax[0].hist(data_raw, bins=bin_number, alpha=0.5, label='Climate model') 
      ax[0].set_title("Not bias corrected")
      ax[0].legend()
      
      i=0
      for k in kwargs.keys():
          
          i=i+1
          ax[i].hist(data_obs, bins=bin_number, alpha=0.5, label='Observed')
          ax[i].hist(kwargs[k], bins=bin_number, alpha=0.5, label='Climate model') 
          ax[i].set_title("Bias corrected ({})".format(k))
          ax[i].legend()
      
      return fig


def plot_bias_spatial(variable, data_obs, data_raw, **kwargs):
    
    """
    Calculates the bias of the mean, 10th percentile and 90th percentile between the 
    observational and climate model data at each location and plots their spatial distribution. Function is intended to be applied to data in the validation
    period.

    Parameters
    ----------
    variable : str, variable name is standard form (i.e. 'tas', 'pr', etc)
    data_obs : three-dimensional array (time, latitude, longitude) of observational data in validation period, numeric entries expected
    data_raw : three-dimensional array (time, latitude, longitude) of raw (i.e. not bias corrected) climate model data in validation period, numeric entries expected
    **kwargs: three-dimensional array (time, latitude, longitude) of bias corrected data sets. To be given in the form bias_correction_name = bias_corrected_dataset, 
    the latter being of the same form as data_obs and data_raw, numeric entries expected.
    """

    number_biascorrections = len(kwargs.keys())
    fig_length = 5 + 5*number_biascorrections
    plot_number = number_biascorrections +1
    
    mean_obs = np.mean(data_obs, axis=0)
    lowpc_obs = np.quantile(data_obs, 0.1, axis=0)
    highpc_obs = np.quantile(data_obs, 0.9, axis=0)
    
    bias_mean = {}
    bias_lowpc = {}
    bias_highpc = {}
    
    bias_mean['raw'] = 100*(mean_obs - np.mean(data_raw, axis=0))/mean_obs
    bias_lowpc['raw'] = 100*(lowpc_obs - np.quantile(data_raw, 0.1, axis=0))/lowpc_obs
    bias_highpc['raw'] = 100*(highpc_obs - np.quantile(data_raw, 0.9, axis=0))/highpc_obs

    
    for k in kwargs.keys():
        
        bias_mean[str(k)] = 100*(mean_obs - np.mean(kwargs[k], axis=0))/mean_obs
        bias_lowpc[str(k)] = 100*(lowpc_obs - np.quantile(kwargs[k], 0.1, axis=0))/lowpc_obs
        bias_highpc[str(k)] = 100*(highpc_obs - np.quantile(kwargs[k], 0.9, axis=0))/highpc_obs

    arrays_max = max(max(np.ndarray.flatten(np.vstack(list(chain(*bias_mean.values()))))),
                   max(np.ndarray.flatten(np.vstack(list(chain(*bias_lowpc.values()))))),
                   max(np.ndarray.flatten(np.vstack(list(chain(*bias_highpc.values()))))))
    
    arrays_min = min(min(np.ndarray.flatten(np.vstack(list(chain(*bias_mean.values()))))),
                   min(np.ndarray.flatten(np.vstack(list(chain(*bias_lowpc.values()))))),
                   min(np.ndarray.flatten(np.vstack(list(chain(*bias_highpc.values()))))))
    
    axis_max = max(abs(arrays_max), abs(arrays_min))
    axis_min = -axis_max

    fig, ax = plt.subplots(plot_number, 3, figsize=(18,fig_length))

    fig.suptitle('{} - Bias'.format(variable_dictionary.get(variable).get('name')))
    
    i=0
    for k in ('raw', *kwargs.keys()):

        plot1 = ax[i, 0].imshow(bias_mean[k], cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
        ax[i, 0].set_title('% bias of mean \n {}'.format(k))
        fig.colorbar(plot1, ax=ax[i, 0])

        plot2 = ax[i, 1].imshow(bias_lowpc[k], cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
        ax[i, 1].set_title('% bias of 10th percentile \n {}'.format(k))
        fig.colorbar(plot2, ax=ax[i, 1])

        plot3 = ax[i, 2].imshow(bias_highpc[k], cmap=plt.get_cmap('coolwarm'), vmin = axis_min, vmax = axis_max)
        ax[i, 2].set_title('% bias of 90th percentile \n {}'.format(k))
        fig.colorbar(plot3, ax=ax[i, 2])
        
        i=i+1


    return fig






def plot_bias_distribution(variable, data_obs, data_raw, **kwargs):
    
    """
    Calculates the bias of the mean, 10th percentile and 90th percentile between the 
    observational and climate model data at each location and plots the distribution of this bias across locations. 
    Function is intended to be applied to data in the validation period.

    Parameters
    ----------
    variable : str, variable name is standard form (i.e. 'tas', 'pr', etc)
    data_obs : three-dimensional array (time, latitude, longitude) of observational data in validation period, numeric entries expected
    data_raw : three-dimensional array (time, latitude, longitude) of raw (i.e. not bias corrected) climate model data in validation period, numeric entries expected
    **kwargs: three-dimensional array (time, latitude, longitude) of bias corrected data sets. To be given in the form bias_correction_name = bias_corrected_dataset, 
    the latter being of the same form as data_obs and data_raw, numeric entries expected.
    """

    mean_obs = np.mean(data_obs, axis=0)
    lowpc_obs = np.quantile(data_obs, 0.1, axis=0)
    highpc_obs = np.quantile(data_obs, 0.9, axis=0)

    bias_mean = {}
    bias_lowpc = {}
    bias_highpc = {}
    
    bias_mean['raw'] = 100*(mean_obs - np.mean(data_raw, axis=0))/mean_obs
    bias_lowpc['raw'] = 100*(lowpc_obs - np.quantile(data_raw, 0.1, axis=0))/lowpc_obs
    bias_highpc['raw'] = 100*(highpc_obs - np.quantile(data_raw, 0.9, axis=0))/highpc_obs

    bias_array = np.empty((0, 3))
    
    for k in kwargs.keys():
        
            bias_mean[str(k)] = 100*(mean_obs - np.mean(kwargs[k], axis=0))/mean_obs

            if np.any(lowpc_obs)==0:
                bias_lowpc[str(k)]=lowpc_obs # TODO think of what to do here
            else:
                bias_lowpc[str(k)] = 100*(lowpc_obs - np.quantile(kwargs[k], 0.1, axis=0))/lowpc_obs

            bias_highpc[str(k)] = 100*(highpc_obs - np.quantile(kwargs[k], 0.9, axis=0))/highpc_obs
            

    for k in ('raw', *kwargs.keys()):
    
            length = len(np.ndarray.flatten(bias_mean['raw']))
    
        
            bias_array = np.append(bias_array,
                               np.transpose(np.array([[k]*length, ['Mean']*length, np.transpose(np.ndarray.flatten(bias_mean[k]))])), 
                               axis = 0)
            bias_array = np.append(bias_array,
                               np.transpose(np.array([[k]*length, ['10pc']*length, np.transpose(np.ndarray.flatten(bias_lowpc[k]))])), 
                               axis = 0)
            bias_array = np.append(bias_array,
                               np.transpose(np.array([[k]*length, ['90pc']*length, np.transpose(np.ndarray.flatten(bias_highpc[k]))])), 
                               axis = 0)
        

    boxplot_data = pd.DataFrame(bias_array, columns=['Correction Method','Metric', 'Percentage bias'])
    boxplot_data["Percentage bias"] = pd.to_numeric(boxplot_data["Percentage bias"])

    fig_width = 5 + 2*len(kwargs.keys())
    fig = plt.figure(figsize=(fig_width, 6))
    seaborn.violinplot(y='Percentage bias', x='Metric', 
                 data=boxplot_data, 
                 palette="colorblind",
                 hue='Correction Method')

    fig.suptitle('{} - Bias'.format(variable_dictionary.get(variable).get('name')))

    return fig

