# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import scipy.stats

isimip_2_5 = {
    "isimip_run": {
        # ISIMIP behavior
        "trend_removal_with_significance_test": True,  # >= v2.1, step 3
        # "powerlaw_exponent_step4": 1,  # >= v2.2: uniform distribution, step 4
        # "pseudo_future_observations_bounded_variables": "v2.3",  # >= v2.3: new equation, step 5
        "trend_transfer_only_for_values_within_threshold": True,  # >= v2.4,  step 5
        # math_helpers
        "ecdf_method": "linear_interpolation",
        "iecdf_method": "linear",
        # iteration
        "running_window_mode": True,  # >= v2.5
        "running_window_length": 31,  # >= v2.5
        "running_window_step_length": 1,  # >= v2.5
    },
    "variables": {
        "hurs": {
            "lower_bound": 0,
            "lower_threshold": 0.01,
            "upper_bound": 100,
            "upper_threshold": 99.99,
            "distribution": scipy.stats.beta,
            "trend_preservation_method": "bounded",
            "detrending": False,
            "event_likelihood_adjustment": False,  # >= v2.5 step 6
        },
        "pr": {
            "lower_bound": 0,
            "lower_threshold": 0.1 / 86400,
            "upper_bound": np.inf,
            "upper_threshold": np.inf,
            "distribution": scipy.stats.gamma,
            "trend_preservation_method": "mixed",
            "detrending": False,
            "event_likelihood_adjustment": False,  # >= v2.5 step 6
        },
        "prsnratio": {
            "lower_bound": 0,
            "lower_threshold": 0.0001,
            "upper_bound": 1,
            "upper_threshold": 0.9999,
            "distribution": scipy.stats.beta,
            "trend_preservation_method": "bounded",
            "detrending": False,
            "event_likelihood_adjustment": False,  # >= v2.5 step 6
        },
        "psl": {
            "lower_bound": -np.inf,
            "lower_threshold": -np.inf,
            "upper_bound": np.inf,
            "upper_threshold": np.inf,
            "distribution": scipy.stats.norm,
            "trend_preservation_method": "additive",
            "detrending": True,
            "event_likelihood_adjustment": False,  # >= v2.5 step 6
        },
        "rsds": {
            "lower_bound": 0,
            "lower_threshold": 0.0001,
            "upper_bound": 1,
            "upper_threshold": 0.9999,
            "distribution": scipy.stats.beta,
            "trend_preservation_method": "bounded",
            "detrending": False,
            "event_likelihood_adjustment": False,  # >= v2.5 step 6
        },
        "sfcWind": {
            "lower_bound": 0,
            "lower_threshold": 0.01,
            "upper_bound": np.inf,
            "upper_threshold": np.inf,
            "distribution": scipy.stats.exponweib,  # TODO: needs to be real weibull (log of exponweib)
            "trend_preservation_method": "mixed",
            "detrending": False,
            "event_likelihood_adjustment": False,  # >= v2.5 step 6
        },
        "tas": {
            "lower_bound": -np.inf,
            "lower_threshold": -np.inf,
            "upper_bound": np.inf,
            "upper_threshold": np.inf,
            "distribution": scipy.stats.norm,
            "trend_preservation_method": "additive",
            "detrending": True,
            "event_likelihood_adjustment": False,
            "reasonable_physical_range": [0, 400],  # TODO: needs to appear everywhere
        },
        "tasrange": {
            "lower_bound": 0,
            "lower_threshold": 0.01,
            "upper_bound": np.inf,
            "upper_threshold": np.inf,
            "distribution": scipy.stats.rice,
            "trend_preservation_method": "mixed",
            "detrending": False,
            "event_likelihood_adjustment": False,  # >= v2.5 step 6
        },
        "tasskew": {
            "lower_bound": 0,
            "lower_threshold": 0.0001,
            "upper_bound": 1,
            "upper_threshold": 0.9999,
            "distribution": scipy.stats.beta,
            "trend_preservation_method": "bounded",
            "detrending": False,
            "event_likelihood_adjustment": False,  # >= v2.5 step 6
        },
    },
}

standard_variables_isimip = {
    "hurs": {
        "lower_bound": 0,
        "lower_threshold": 0.01,
        "upper_bound": 100,
        "upper_threshold": 99.99,
        "distribution": scipy.stats.beta,
        "trend_preservation_method": "bounded",
        "detrending": False,
    },
    "pr": {
        "lower_bound": 0,
        "lower_threshold": 0.1 / 86400,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.gamma,
        "trend_preservation_method": "mixed",
        "detrending": False,
    },
    "prsnratio": {
        "lower_bound": 0,
        "lower_threshold": 0.0001,
        "upper_bound": 1,
        "upper_threshold": 0.9999,
        "distribution": scipy.stats.beta,
        "trend_preservation_method": "bounded",
        "detrending": False,
    },
    "psl": {
        "lower_bound": -np.inf,
        "lower_threshold": -np.inf,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.norm,
        "trend_preservation_method": "additive",
        "detrending": True,
    },
    "rsds": {
        "lower_bound": 0,
        "lower_threshold": 0.0001,
        "upper_bound": 1,
        "upper_threshold": 0.9999,
        "distribution": scipy.stats.beta,
        "trend_preservation_method": "bounded",
        "detrending": False,
    },
    "sfcWind": {
        "lower_bound": 0,
        "lower_threshold": 0.01,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.exponweib,  # TODO: needs to be real weibull (log of exponweib)
        "trend_preservation_method": "mixed",
        "detrending": False,
    },
    "tas": {
        "lower_bound": -np.inf,
        "lower_threshold": -np.inf,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.norm,
        "trend_preservation_method": "additive",
        "detrending": True,
        "reasonable_physical_range": [0, 400],  # TODO: needs to appear everywhere
    },
    "tasrange": {
        "lower_bound": 0,
        "lower_threshold": 0.01,
        "upper_bound": np.inf,
        "upper_threshold": np.inf,
        "distribution": scipy.stats.rice,
        "trend_preservation_method": "mixed",
        "detrending": False,
    },
    "tasskew": {
        "lower_bound": 0,
        "lower_threshold": 0.0001,
        "upper_bound": 1,
        "upper_threshold": 0.9999,
        "distribution": scipy.stats.beta,
        "trend_preservation_method": "bounded",
        "detrending": False,
    },
}

# TODO: keep track of isimip-modes somehow. Also think about init and how to change options
# I can use a dataclass and order of arguments + dict of arugments to map into that
