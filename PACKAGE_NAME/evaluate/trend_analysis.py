# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd


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
        
        

