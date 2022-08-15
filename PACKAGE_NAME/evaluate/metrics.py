# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import scipy

variable_dictionary = {
    "tas": {
        "distribution": scipy.stats.norm,
        "trend_preservation": "additive",
        "detrending": True,
        "name": "2m daily mean air temperature (K)",
        "high_threshold": 295,
        "low_threshold": 273,
        "unit": "K",
    },
    "pr": {
        "distribution": scipy.stats.gamma,
        "trend_preservation": "mixed",
        "detrending": False,
        "name": "Total precipitation (m/day)",
        "high_threshold": 0.0004,
        "low_threshold": 0.00001,
        "unit": "m/day",
    },
}

metrics_dictionary = {
    "frost": {
        "variable": "tasmin",
        "variablename": "2m daily minimum air temperature (K)",
        "value": 273.15,
        "threshold_sign": "lower",
        "name": "Frost days",
    },
    "mean_warm_day": {
        "variable": "tas",
        "variablename": "2m daily mean air temperature (K)",
        "value": 295,
        "threshold_sign": "higher",
        "name": "Warm days (mean)",
    },
    "mean_cold_day": {
        "variable": "tas",
        "variablename": "2m daily mean air temperature (K)",
        "value": 273,
        "threshold_sign": "lower",
        "name": "Cold days (mean)",
    },
    "dry": {
        "variable": "pr",
        "variablename": "Precipitation",
        "value": 0.000001,
        "threshold_sign": "lower",
        "name": "Dry days (mean)",
    },
    "wet": {
        "variable": "pr",
        "variable_name": "Precipitation",
        "value": 1 / 86400,
        "threshold_sign": "higher",
        "name": "Wet days (daily total precipitation > 1 mm)",
    },
}


def calculate_eot_matrix(dataset: np.ndarray, threshold_name: str) -> np.ndarray:

    """
    Converts np.ndarray of input data (observations or climate projections) into 1-0 np.ndarray of same dimensions based on
    threshold value and sign. Assignes 1 if value is below/above specified threshold (exceedance over threshold - eot), 0 otherwise.

    Parameters
    ----------
    dataset: np.ndarray
        Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected
    threshold_name: str
        Name of threshold metric specified in the metrics dictionary
    """

    thresholds = np.copy(dataset)

    if metrics_dictionary.get(threshold_name).get("threshold_sign") == "higher":

        thresholds = (thresholds > metrics_dictionary.get(threshold_name).get("value")).astype(int)

    elif metrics_dictionary.get(threshold_name).get("threshold_sign") == "lower":

        thresholds = (thresholds < metrics_dictionary.get(threshold_name).get("value")).astype(int)

    else:
        raise ValueError('Invalid threshold sign. Modify threshold sign in metrics dictionary')

    return thresholds



def calculate_eot_probability(dataset: np.ndarray, threshold_name: str) -> np.ndarray:

    """
    Calculates the probability of exceeding a specified threshold at each location,
    building on the function calculate_matrix.
    
    Parameters
    ----------
    dataset: np.ndarray
        Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected
    threshold_name: str
        Name of threshold metric specified in the metrics dictionary
    """
    
    threshold_data = calculate_eot_matrix(dataset, threshold_name)

    threshold_probability = np.einsum('ijk -> jk', threshold_data)/threshold_data.shape[0]

    return threshold_probability
