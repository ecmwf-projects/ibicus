# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import matplotlib.pyplot as plt
import scipy

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

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



def goodness_of_fit_aic(variable, dataset):

    aic = np.array([])

    for i in range(dataset.shape[1]):
        for j in range(dataset.shape[2]):

            fit = variable_dictionary.get(variable).get('distribution').fit(dataset[:, i, j])
            
            k = len(fit)
            logLik = np.sum(variable_dictionary.get(variable).get('distribution').logpdf(dataset[:, i, j], *fit))
            aic_location = 2*k - 2*(logLik)
            
            aic = np.append(aic, aic_location)
            
    
    return(aic) 
  


  
def goodness_of_fit_plot(dataset, variable, data_type):
    
    from scipy.stats import norm

    fit = variable_dictionary.get(variable).get('distribution').fit(dataset)
    q = variable_dictionary.get(variable).get('distribution').cdf(dataset, *fit)
    
    q_normal = norm.ppf(q)
    
    fig, ax = plt.subplots(1,3, figsize=(14,4))

    fig.suptitle('Goodness of fit evaluation - {} {}'.format(variable_dictionary.get(variable).get('name'), data_type))
    
    x = range(0, len(q))
    ax[0].plot(x, q)
    ax[0].set_title('Quantile Residuals')
    
    plot_acf(q, lags=1000, ax=ax[1])
    ax[1].set_title('ACF')
    
    sm.qqplot(q_normal, line='45', ax=ax[2])
    ax[2].set_title('QQ Plot')
    
    return(fig)