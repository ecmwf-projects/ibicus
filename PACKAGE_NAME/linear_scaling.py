import warnings

import numpy as np
import scipy

from .debiaser import Debiaser

standard_delta_types = {
    "temp": "additive",
    "pr": "multiplicative",
    "temp": "additive",
    "temp": "additive",
}


class LinearScaling(Debiaser):
    def __init__(self, variable=None, delta_type=None):

        # Checks
        if variable is not None:
            if variable not in standard_delta_types.keys():
                raise ValueError(
                    "variable needs to be one of %s" % standard_delta_types.keys()
                )

        if delta_type is not None:
            if delta_type not in ["additive", "multiplicative"]:
                raise ValueError(
                    'delta_type needs to be one of ["additive", "multiplicative"].'
                )

        # Parameter setting
        if variable is not None:
            if delta_type is not None:
                if delta_type != standard_delta_types.get(variable):
                    warnings.warn(
                        "Given delta type for variable is different from standard one."
                    )
                self.variable = variable
                self.delta_type = delta_type
            else:
                self.variable = variable
                self.delta_type = standard_delta_types.get(variable)
        else:
            if delta_type is not None:
                self.variable = "unknown"
                self.delta_type = delta_type
            else:
                raise ValueError(
                    "At least one of variable, delta_type needs to be specified."
                )

    def apply_location(self, obs, cm_hist, cm_future):
        if self.delta_type == "additive":
            return cm_future - (np.mean(cm_hist) - np.mean(obs))
        elif self.delta_type == "multiplicative":
            return cm_future * (np.mean(obs) / np.mean(cm_hist))
        else:
            raise ValueError(
                'self.delta_type needs to be one of ["additive", "multiplicative"].'
            )
