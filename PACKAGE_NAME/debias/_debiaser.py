# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from tqdm import tqdm

from ..variables import Variable, map_variable_str_to_variable_class


class Debiaser(ABC):
    def __init__(self, name):
        self.name = name

    # Constructors
    def from_variable(
        child_class,
        variable: Union[str, Variable],
        default_settings_variable: dict,
        default_settings_general: dict = {},
        **kwargs
    ):
        """
        Instanciates the class from a variable: either a string referring to a standard variable name or a Variable object.

        Parameters
        ----------
        child_class:
            Child class of debiaser to be instantiated.
        variable : Union[str, Variable]
            String or Variable object referring to standard meteorological variable for which default settings can be used.
        default_settings_variable : dict
            Dict of default settings for each variables. Has Variable-objects as keys (eg. tas, hurs) and dicts as values which map to the class parameters and store the default settings for these variables.
        default_settings_general : dict
            Default: {} (empty dict). Dict of general default settings (not variable specific) for the debiaser. Settings in here get overwritten by the variable specific ones.
        **kwargs:
            All other class attributes that shall be set and where the standard values for variable shall be overwritten.
        """

        if not isinstance(variable, Variable):
            variable_object = map_variable_str_to_variable_class(variable)
        else:
            variable_object = variable

        if variable_object not in default_settings_variable.keys():
            raise ValueError("No default settings exist for %s in debiaser %s" % (variable, child_class.__name__))

        parameters = {
            **default_settings_general,
            **default_settings_variable[variable_object],
            "variable": variable_object.name,
        }
        return child_class(**{**parameters, **kwargs})

    # Input checks:
    @staticmethod
    def is_correct_type(df):
        return isinstance(df, np.ndarray)

    @staticmethod
    def has_correct_shape(df):
        if df.ndim == 3:
            return True
        else:
            return False

    @staticmethod
    def have_same_shape(obs, cm_hist, cm_future):
        if obs.shape[1:] == cm_hist.shape[1:] and obs.shape[1:] == cm_future.shape[1:]:
            return True
        else:
            return False

    @staticmethod
    def contains_inf_nan(x):
        return any(np.isnan(x) | np.isinf(x))

    @staticmethod
    def check_inputs(obs, cm_hist, cm_future):
        # correct type
        if not Debiaser.is_correct_type(obs):
            raise TypeError("Wrong type for obs. Needs to be np.ndarray")

        # correct shape
        if not Debiaser.has_correct_shape(obs):
            raise ValueError("obs needs to have 3 dimensions: time, x, y")

        if not Debiaser.has_correct_shape(cm_hist):
            raise ValueError("cm_hist needs to have 3 dimensions: time, x, y")

        if not Debiaser.has_correct_shape(cm_future):
            raise ValueError("cm_future needs to have 3 dimensions: time, x, y")

        # have shame shape
        if not Debiaser.have_same_shape(obs, cm_hist, cm_future):
            raise ValueError("obs, cm_hist, cm_future need to have same (number of) spatial dimensions")
        """
        # contains inf, nan or na
        if Debiaser.contains_inf_nan(obs) or Debiaser.contains_inf_nan(cm_hist) or Debiaser.contains_inf_nan(cm_future):
            raise ValueError("One of obs, cm_hist, cm_future contains inf or nan values")
        """
        return True

    # Helpers
    @staticmethod
    def _unpack_iterable_args_and_get_locationwise_info(i, j, iterable_args):
        return {
            key: value[
                :,
                i,
                j,
            ]
            for key, value in iterable_args.items()
        }

    @staticmethod
    def map_over_locations(func, output_size, obs, cm_hist, cm_future, **kwargs):
        output = np.empty(output_size, dtype=cm_future.dtype)
        for i, j in tqdm(np.ndindex(obs.shape[1:]), total=np.prod(obs.shape[1:])):
            output[:, i, j] = func(obs[:, i, j], cm_hist[:, i, j], cm_future[:, i, j], **kwargs)
        return output

    # Apply functions:
    @abstractmethod
    def apply_location(self, obs, cm_hist, cm_future, **kwargs):
        pass

    def apply(self, obs, cm_hist, cm_future, **kwargs):
        print("----- Running debiasing -----")
        Debiaser.check_inputs(obs, cm_hist, cm_future)

        output = Debiaser.map_over_locations(
            self.apply_location, output_size=cm_future.shape, obs=obs, cm_hist=cm_hist, cm_future=cm_future, **kwargs
        )
        return output
