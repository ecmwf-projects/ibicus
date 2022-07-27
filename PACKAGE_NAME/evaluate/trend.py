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
import math
import scipy


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


def calculate_trend_once(data_hist, data_future, datatype):

    trend_additive = np.empty((0, 3))
    
    for i in range(data_hist.shape[1]):
        for j in range(data_hist.shape[2]):
                
            trend_additive = np.append(trend_additive, 
                                         [[datatype, '10% qn',np.quantile(data_future[:, i, j], 0.1) - np.quantile(data_hist[:, i, j], 0.1)]],
                                                                     axis = 0)
            trend_additive = np.append(trend_additive, 
                                         [[datatype, '90% qn',np.quantile(data_future[:, i, j], 0.9) - np.quantile(data_hist[:, i, j], 0.9)]],
                                                                     axis = 0)
            trend_additive = np.append(trend_additive, 
                                         [[datatype, 'Mean',np.mean(data_future[:, i, j]) - np.mean(data_hist[:, i, j])]],
                                                                     axis = 0)
    return(trend_additive)


def calculate_trend_bias(variable, **kwargs):
    
    trend_data = np.empty((0, 3))
    
    for k in kwargs.keys():
    
        trend = calculate_trend_once(*kwargs[k],k)
        trend_data = np.append(trend_data, trend, axis=0)
    
    boxplot_data = pd.DataFrame(trend_data, columns=['Correction Method','Metric', 'Additive Trend'])
    boxplot_data['Additive Trend'] = pd.to_numeric(boxplot_data['Additive Trend'])

    fig = plt.figure(figsize=(10, 6))
    seaborn.boxplot(y='Additive Trend', x='Metric', 
                 data=boxplot_data, 
                 palette="colorblind",
                 hue='Correction Method')
    fig.suptitle('{} - additive trend'.format(variable_dictionary.get(variable).get('name')))

    return(fig)  

        

