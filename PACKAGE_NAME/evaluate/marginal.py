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

