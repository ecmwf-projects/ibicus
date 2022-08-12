# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
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
            raise ValueError(
                "Unfortunately currently no default settings exist for the variable '%s' in the debiaser %s. You can set the required class parameters manually by using the class constructor. This also allows more fine-grained optimization of the debiaser."
                % (variable, child_class.__name__)
            )

        parameters = {
            **default_settings_general,
            **default_settings_variable[variable_object],
            "variable": variable_object.name,
        }
        return child_class(**{**parameters, **kwargs})

    # ----- Helpers: Input checks ----- #

    @staticmethod
    def is_correct_type(df):
        return isinstance(df, np.ndarray)

    @staticmethod
    def has_correct_shape(df):
        return df.ndim == 3

    @staticmethod
    def have_same_shape(obs, cm_hist, cm_future):
        return obs.shape[1:] == cm_hist.shape[1:] and obs.shape[1:] == cm_future.shape[1:]

    @staticmethod
    def contains_inf_nan(x):
        return np.any(np.logical_or(np.isnan(x), np.isinf(x)))

    @staticmethod
    def has_float_dtype(x):
        return np.issubdtype(x.dtype, np.floating)

    @staticmethod
    def is_masked_array(x):
        return isinstance(x, np.ma.core.MaskedArray)

    @staticmethod
    def masked_array_contains_invalid_values(x):
        return np.any(x.mask)

    # ----- Helpers: Input converters ----- #

    @staticmethod
    def convert_to_float_dtype(x):
        try:
            return x.astype(float)
        except:
            raise ValueError("Conversion to float not possible. Please use float datatype for obs, cm_hist, cm_future.")

    @staticmethod
    def fill_masked_array_with_nan(x):
        return x.filled(np.nan)

    # ----- Input checks ----- #

    @staticmethod
    def check_inputs_and_convert_if_possible(obs, cm_hist, cm_future):

        # correct type
        if not Debiaser.is_correct_type(obs):
            raise TypeError("Wrong type for obs. Needs to be np.ndarray")
        if not Debiaser.is_correct_type(cm_hist):
            raise TypeError("Wrong type for cm_hist. Needs to be np.ndarray")
        if not Debiaser.is_correct_type(cm_future):
            raise TypeError("Wrong type for cm_future. Needs to be np.ndarray")

        # correct dtype
        if not Debiaser.has_float_dtype(obs):
            logging.warning("obs does not have a float dtype. Attempting conversion.")
            obs = Debiaser.convert_to_float_dtype(obs)
        if not Debiaser.has_float_dtype(cm_hist):
            logging.warning("cm_hist does not have a float dtype. Attempting conversion.")
            cm_hist = Debiaser.convert_to_float_dtype(cm_hist)
        if not Debiaser.has_float_dtype(cm_future):
            logging.warning("cm_future does not have a float dtype. Attempting conversion.")
            cm_future = Debiaser.convert_to_float_dtype(cm_future)

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

        # contains inf or nan
        if Debiaser.contains_inf_nan(obs):
            logging.warning(
                "obs contains inf or nan values. Not all debiasers support missing values and their presence might lead to infs or nans inside of the debiased values. Consider infilling the missing values."
            )
        if Debiaser.contains_inf_nan(cm_hist):
            logging.warning(
                "cm_hist contains inf or nan values. Not all debiasers support missing values and their presence might lead to infs or nans inside of the debiased values. Consider infilling the missing values."
            )
        if Debiaser.contains_inf_nan(cm_future):
            logging.warning(
                "cm_future contains inf or nan values. Not all debiasers support missing values and their presence might lead to infs or nans inside of the debiased values. Consider infilling the missing values."
            )

        # masked arrays
        if Debiaser.is_masked_array(obs):
            if Debiaser.masked_array_contains_invalid_values(obs):
                logging.warning(
                    "obs is a masked array and contains cells with invalid data. Not all debiasers support invalid/missing values and their presence might lead to infs or nans inside the debiased values. Consider infilling them. For computation the masked values here are filled in by nan-values."
                )
            else:
                logging.info(
                    "obs is a masked array, but contains no invalid data. It is converted to a normal numpy array."
                )
            obs = Debiaser.fill_masked_array_with_nan(obs)
        if Debiaser.is_masked_array(cm_hist):
            if Debiaser.masked_array_contains_invalid_values(cm_hist):
                logging.warning(
                    "cm_hist is a masked array and contains cells with invalid data. Not all debiasers support invalid/missing values and their presence might lead to infs or nans inside the debiased values. Consider infilling them. For computation the masked values here are filled in by nan-values."
                )
            else:
                logging.info(
                    "cm_hist is a masked array, but contains no invalid data. It is converted to a normal numpy array."
                )
            cm_hist = Debiaser.fill_masked_array_with_nan(cm_hist)
        if Debiaser.is_masked_array(cm_future):
            if Debiaser.masked_array_contains_invalid_values(cm_future):
                logging.warning(
                    "cm_future is a masked array and contains cells with invalid data. Not all debiasers support invalid/missing values and their presence might lead to infs or nans inside the debiased values. Consider infilling them. For computation the masked values here are filled in by nan-values."
                )
            else:
                logging.info(
                    "cm_future is a masked array, but contains no invalid data. It is converted to a normal numpy array."
                )
            cm_future = Debiaser.fill_masked_array_with_nan(cm_future)

        return obs, cm_hist, cm_future

    # ----- Helpers ----- #

    @staticmethod
    def set_up_logging(verbosity):
        verbosity = verbosity.upper()

        if verbosity == "INFO":
            level = logging.INFO
        elif verbosity == "WARNING":
            level = logging.WARNING
        elif verbosity == "ERROR":
            level = logging.ERROR
        else:
            raise ValueError('verbosity needs to be one of ["INFO", "WARNING", "ERROR"]')

        logging.basicConfig(encoding="utf-8", level=level)
        logging.getLogger().setLevel(level)

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

    # ----- Apply functions ----- #

    @abstractmethod
    def apply_location(self, obs, cm_hist, cm_future, **kwargs):
        pass

    def apply(self, obs, cm_hist, cm_future, verbosity="INFO", **kwargs):
        """
        Applies the debiaser onto given data.
        """

        Debiaser.set_up_logging(verbosity)
        logging.info("----- Running debiasing for variable: %s -----" % self.variable)

        obs, cm_hist, cm_future = Debiaser.check_inputs_and_convert_if_possible(obs, cm_hist, cm_future)

        output = Debiaser.map_over_locations(
            self.apply_location, output_size=cm_future.shape, obs=obs, cm_hist=cm_hist, cm_future=cm_future, **kwargs
        )
        return output
