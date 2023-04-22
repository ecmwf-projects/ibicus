# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from typing import Optional, Union

import attrs
import numpy as np
from tqdm import tqdm

from ..utils import get_library_logger
from ..variables import Variable, map_variable_str_to_variable_class


@attrs.define(slots=False, kw_only=True)
class Debiaser(ABC):
    """
    A generic debiaser meant for subclassing. Provides functionality for individual debiasers and a unified interface to apply debiasing.

    The debiaser abstract class provides a unified interface to call the debiaser, as well as a vaeriety of setup tasks and input-checks. In order to subclass the :py:class:`debiaser`-class, the proposed debiaser needs to implement the :py:func:`from_variable` and :py:func:`apply_location` functions:

    - :py:func:`apply_location`: applies an initialised debiaser at one location. Arguments are 1d-vectors of obs, cm_hist, and cm_future representing observations, and climate model values during the reference (cm_hist) and future period (cm_future). Additionally ``kwargs`` passed to the debiaser :py:func:`apply`-function are passed down to the :py:func:`apply_location`-function.

    - :py:func:`from_variable`: initialises a debiaser with default arguments given a climatic variable either as ``str`` or member of the :py:class:`Variable`-class. ``kwargs`` are meant to overwrite default arguments for this variable. Given a `dict` of default arguments: with variables of the :py:class:`Variable` class as members and `dict` of default arguments as values the :py:func:`_from_variable`-function can be used.

     The :py:func:`apply` function, maps the debiaser :py:func:`apply_location` over locations. This allows to always initialise and apply debiasers follows:

    >>> debiaser = LinearScaling.from_variable("tas") # LinearScaling is a child-class of Debiaser
    >>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)

    or:

    >>> debiaser = CDFt.from_variable("pr", delta_shift="no_shift") # CDFt is a child-class of Debiaser
    >>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)


    Attributes
    ----------
    variable : str
        Variable that is meant to be debiased, by an initialisation of the debiaser. Default: ``"unknown"``.
    reasonable_physical_range : Optional[list]
        Reasonable physical range of the variable to debias in the form ``[lower_bound, upper_bound]``. It is checked against and warnings are raise if values fall outside the range. Default: ``None``.
    """

    variable: str = attrs.field(
        default="unknown", validator=attrs.validators.instance_of(str), eq=False
    )
    reasonable_physical_range: Optional[list] = attrs.field(default=None, eq=False)

    @reasonable_physical_range.validator
    def _validate_reasonable_physical_range(self, attribute, value):
        if value is not None:
            if len(value) != 2:
                raise ValueError(
                    "reasonable_physical_range should have only a lower and upper physical bound"
                )
            if not all(isinstance(elem, (int, float)) for elem in value):
                raise ValueError(
                    "reasonable_physical_range needs to be a list of floats"
                )
            if not value[0] < value[1]:
                raise ValueError(
                    "lower bounds needs to be smaller than upper bound in reasonable_physical_range"
                )

    # ----- Constructors ----- #
    # Helper for downstream classes
    def _from_variable(
        child_class,
        variable: Union[str, Variable],
        default_settings_variable: dict,
        experimental_default_setting_variable: dict = {},
        default_settings_general: dict = {},
        **kwargs,
    ):
        """
        Instanciates a class given by ``child_class`` from a variable: either a string referring to a standard variable name following the CMIP convention or a :py:class:`Variable` object.

        Parameters
        ----------
        child_class:
            Child class of debiaser to be instantiated.
        variable : Union[str, Variable]
            String or Variable object referring to standard meteorological variable for which default settings can be used.
        default_settings_variable : dict
            Dict of default settings for each variables. Has :py:class:`Variable`-objects as keys (eg. ``tas``, ``hurs``) and dicts as values which map to the class parameters and store the default settings for these variables.
        experimental_default_setting_variable : dict
            Dict of experimental default settings for variables. Same structure as ``default_settings_variable``, but keys should be mutually exclusive with those (or else a warning will be thrown and the standard default settings will be used). Throws a warning that the default settings for this variable are still experimental if used.
        default_settings_general : dict
            Dict of general default settings (not variable specific) for the debiaser. Settings in here get overwritten by the variable specific ones. Default: `{}` (empty dict).
        **kwargs:
            All other class attributes that shall be set and where the standard values for variable shall be overwritten.
        """

        logger = get_library_logger()

        # Check default and experimental default settings
        if (
            len(
                intersection := (
                    default_settings_variable.keys()
                    & experimental_default_setting_variable.keys()
                )
            )
            != 0
        ):
            logger.warning(
                f"Default and experimental default settings are not mutually exclusive for variables: {intersection} in debiaser {child_class.__name__}. Standard default settings are taken, but please review!"
            )

        # Get variable arguments
        if not isinstance(variable, Variable):
            variable_object = map_variable_str_to_variable_class(variable)
        else:
            variable_object = variable

        # If default settings exist
        if variable_object in default_settings_variable.keys():
            variable_settings = default_settings_variable[variable_object]
        else:
            # If experimental default settings exist
            if variable_object in experimental_default_setting_variable.keys():
                warnings.warn(
                    f"The default settings for variable {variable} in debiaser {child_class.__name__} are currently still experimental and may not have been evaluated in the peer-reviewed literature. Please review the results with care!",
                    stacklevel=2,
                )
                variable_settings = experimental_default_setting_variable[
                    variable_object
                ]
            else:
                raise ValueError(
                    f"Unfortunately currently no default settings exist for the variable {variable} in the debiaser {child_class.__name__}. You can set the required class parameters manually by using the class constructor."
                )

        # Instantiate class
        parameters = {
            "variable": variable_object.name,
            "reasonable_physical_range": variable_object.reasonable_physical_range,
            **default_settings_general,
            **variable_settings,
        }
        return child_class(**{**parameters, **kwargs})

    @classmethod
    @abstractmethod
    def from_variable(cls, variable: Union[str, Variable], **kwargs):
        """
        Instanciates the class from a variable: either a string referring to a standard variable name or a :py:class:`Variable` object.

        Parameters
        ----------
        variable : Union[str, Variable]
            String or Variable object referring to standard meteorological variable for which default settings can be used.
        **kwargs:
            All other class attributes that shall be set and where the standard values for variable shall be overwritten.

        Returns
        -------
        Debiaser
            Instance of the class for the given variable.
        """
        raise NotImplementedError(
            f"abstract classmethod from_variable of debiaser-class is not implemented class {cls.__name__} inheriting from debiaser-class. It needs to be overwritten in the child class."
        )

    # ----- Helpers: Input checks ----- #

    @staticmethod
    def _is_correct_type(df):
        return isinstance(df, np.ndarray)

    @staticmethod
    def _has_correct_shape(df):
        return df.ndim == 3

    @staticmethod
    def _have_same_shape(obs, cm_hist, cm_future):
        return (
            obs.shape[1:] == cm_hist.shape[1:] and obs.shape[1:] == cm_future.shape[1:]
        )

    @staticmethod
    def _contains_inf_nan(x):
        return np.any(np.logical_or(np.isnan(x), np.isinf(x)))

    def _not_if_or_nan_vals_outside_reasonable_physical_range(self, x):
        if self.reasonable_physical_range is not None:
            return not np.all(
                (x >= self.reasonable_physical_range[0])
                & (x <= self.reasonable_physical_range[1])
                | np.isinf(x)
                | np.isnan(x)
            )
        return False

    @staticmethod
    def _has_float_dtype(x):
        return np.issubdtype(x.dtype, np.floating)

    @staticmethod
    def _is_masked_array(x):
        return isinstance(x, np.ma.core.MaskedArray)

    @staticmethod
    def _masked_array_contains_invalid_values(x):
        return np.any(x.mask)

    # ----- Helpers: Input converters ----- #

    @staticmethod
    def _convert_to_float_dtype(x):
        try:
            return x.astype(float)
        except Exception:
            raise ValueError(
                "Conversion to float not possible. Please use float datatype for obs, cm_hist, cm_future."
            )

    @staticmethod
    def _fill_masked_array_with_nan(x):
        return x.filled(np.nan)

    # ----- Input checks ----- #

    def _check_inputs_and_convert_if_possible(self, obs, cm_hist, cm_future):
        # correct type
        if not Debiaser._is_correct_type(obs):
            raise TypeError("Wrong type for obs. Needs to be np.ndarray")
        if not Debiaser._is_correct_type(cm_hist):
            raise TypeError("Wrong type for cm_hist. Needs to be np.ndarray")
        if not Debiaser._is_correct_type(cm_future):
            raise TypeError("Wrong type for cm_future. Needs to be np.ndarray")

        # correct dtype
        if not Debiaser._has_float_dtype(obs):
            warnings.warn(
                "obs does not have a float dtype. Attempting conversion.", stacklevel=2
            )
            obs = Debiaser._convert_to_float_dtype(obs)
        if not Debiaser._has_float_dtype(cm_hist):
            warnings.warn(
                "cm_hist does not have a float dtype. Attempting conversion.",
                stacklevel=2,
            )
            cm_hist = Debiaser._convert_to_float_dtype(cm_hist)
        if not Debiaser._has_float_dtype(cm_future):
            warnings.warn(
                "cm_future does not have a float dtype. Attempting conversion.",
                stacklevel=2,
            )
            cm_future = Debiaser._convert_to_float_dtype(cm_future)

        # correct shape
        if not Debiaser._has_correct_shape(obs):
            raise ValueError("obs needs to have 3 dimensions: time, x, y")
        if not Debiaser._has_correct_shape(cm_hist):
            raise ValueError("cm_hist needs to have 3 dimensions: time, x, y")
        if not Debiaser._has_correct_shape(cm_future):
            raise ValueError("cm_future needs to have 3 dimensions: time, x, y")

        # have shame shape
        if not Debiaser._have_same_shape(obs, cm_hist, cm_future):
            raise ValueError(
                "obs, cm_hist, cm_future need to have same (number of) spatial dimensions. The arrays of obs, cm_hist and cm_future are assumed to have the following structure: [t, x, y] where t is the time dimension and x, y are spatial ones."
            )

        # contains inf or nan
        if Debiaser._contains_inf_nan(obs):
            warnings.warn(
                "obs contains inf or nan values. Not all debiasers support missing values and their presence might lead to infs or nans inside of the debiased values. Consider infilling the missing values.",
                stacklevel=2,
            )
        if Debiaser._contains_inf_nan(cm_hist):
            warnings.warn(
                "cm_hist contains inf or nan values. Not all debiasers support missing values and their presence might lead to infs or nans inside of the debiased values. Consider infilling the missing values.",
                stacklevel=2,
            )
        if Debiaser._contains_inf_nan(cm_future):
            warnings.warn(
                "cm_future contains inf or nan values. Not all debiasers support missing values and their presence might lead to infs or nans inside of the debiased values. Consider infilling the missing values.",
                stacklevel=2,
            )

        # in reasonable physical range:
        if self._not_if_or_nan_vals_outside_reasonable_physical_range(obs):
            warnings.warn(
                "obs contains values outside the reasonable physical range of %s for the variable: %s. This might be due to different units of to data problems. It is recommended to check the input."
                % (self.reasonable_physical_range, self.variable),
                stacklevel=2,
            )
        if self._not_if_or_nan_vals_outside_reasonable_physical_range(cm_hist):
            warnings.warn(
                "cm_hist contains values outside the reasonable physical range of %s for the variable: %s. This might be due to different units of to data problems. It is recommended to check the input."
                % (self.reasonable_physical_range, self.variable),
                stacklevel=2,
            )
        if self._not_if_or_nan_vals_outside_reasonable_physical_range(cm_future):
            warnings.warn(
                "cm_future contains values outside the reasonable physical range of %s for the variable: %s. This might be due to different units of to data problems. It is recommended to check the input."
                % (self.reasonable_physical_range, self.variable),
                stacklevel=2,
            )

        # masked arrays
        if Debiaser._is_masked_array(obs):
            if Debiaser._masked_array_contains_invalid_values(obs):
                warnings.warn(
                    "obs is a masked array and contains cells with invalid data. Not all debiasers support invalid/missing values and their presence might lead to infs or nans inside the debiased values. Consider infilling them. For computation the masked values here are filled in by nan-values.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "obs is a masked array, but contains no invalid data. It is converted to a normal numpy array.",
                    stacklevel=2,
                )
            obs = Debiaser._fill_masked_array_with_nan(obs)
        if Debiaser._is_masked_array(cm_hist):
            if Debiaser._masked_array_contains_invalid_values(cm_hist):
                warnings.warn(
                    "cm_hist is a masked array and contains cells with invalid data. Not all debiasers support invalid/missing values and their presence might lead to infs or nans inside the debiased values. Consider infilling them. For computation the masked values here are filled in by nan-values.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "cm_hist is a masked array, but contains no invalid data. It is converted to a normal numpy array.",
                    stacklevel=2,
                )
            cm_hist = Debiaser._fill_masked_array_with_nan(cm_hist)
        if Debiaser._is_masked_array(cm_future):
            if Debiaser._masked_array_contains_invalid_values(cm_future):
                warnings.warn(
                    "cm_future is a masked array and contains cells with invalid data. Not all debiasers support invalid/missing values and their presence might lead to infs or nans inside the debiased values. Consider infilling them. For computation the masked values here are filled in by nan-values.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "cm_future is a masked array, but contains no invalid data. It is converted to a normal numpy array.",
                    stacklevel=2,
                )
            cm_future = Debiaser._fill_masked_array_with_nan(cm_future)

        return obs, cm_hist, cm_future

    def _check_output(self, output):
        if Debiaser._contains_inf_nan(output):
            warnings.warn(
                "The debiaser output contains inf or nan values. This might be due to inf or nan values inside the input, or to a problem of the debiaser for the given dataset at hand. It is recommended to check the output carefully",
                stacklevel=2,
            )

        # in reasonable physical range:
        if self._not_if_or_nan_vals_outside_reasonable_physical_range(output):
            warnings.warn(
                "The debiaser output contains values outside the reasonable physical range of %s for the variable: %s. This might be due to values outside the range in the input, or to a problem of the debiaser for the given dataset at hand. It is recommended to check the output carefully."
                % (self.reasonable_physical_range, self.variable),
                stacklevel=2,
            )

    # ----- Helpers ----- #

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
    def _run_func_on_location_and_catch_error(
        obs, cm_hist, cm_future, func, failsafe=False, **kwargs
    ):
        # Note: obs, cm_hist, cm_future need to be the first arguments instead of func,
        # because pool.starmap is used for which the argument order matters.
        try:
            return func(obs, cm_hist, cm_future, **kwargs)
        except Exception as e:
            if failsafe:
                # Log error
                logger = get_library_logger()
                logger.error(
                    "ibicus encountered at error at runtime. Please check the output carefully!"
                )
                logger.error(e)
                # Return nan
                return np.nan
            else:
                # Raise error if not in failsafe mode
                raise

    @staticmethod
    def map_over_locations(
        func,
        output_size,
        obs,
        cm_hist,
        cm_future,
        progressbar=True,
        failsafe=False,
        **kwargs,
    ):
        output = np.empty(output_size, dtype=cm_future.dtype)

        indices = np.ndindex(obs.shape[1:])
        if progressbar:
            indices = tqdm(indices, total=np.prod(obs.shape[1:]))

        for i, j in indices:
            output[:, i, j] = Debiaser._run_func_on_location_and_catch_error(
                obs[:, i, j],
                cm_hist[:, i, j],
                cm_future[:, i, j],
                func,
                failsafe=failsafe,
                **kwargs,
            )
        return output

    @staticmethod
    def parallel_map_over_locations(
        func,
        output_size,
        obs,
        cm_hist,
        cm_future,
        nr_processes=4,
        failsafe=False,
        **kwargs,
    ):
        # compute results
        indices = [(i, j) for i in range(obs.shape[1]) for j in range(obs.shape[2])]
        with Pool(processes=nr_processes) as pool:
            result = pool.starmap(
                partial(
                    Debiaser._run_func_on_location_and_catch_error,
                    func=func,
                    failsafe=failsafe,
                    **kwargs,
                ),
                [
                    (obs[:, i, j], cm_hist[:, i, j], cm_future[:, i, j])
                    for (i, j) in indices
                ],
            )

        # fill output
        output = np.empty(output_size, dtype=cm_future.dtype)
        for k, index in enumerate(indices):
            output[:, index[0], index[1]] = result[k]

        # result = np.array(result)
        # result = np.moveaxis(result.reshape(obs.shape[1], obs.shape[2], obs.shape[0]), -1, 0)

        return output

    # ----- Apply functions ----- #

    @abstractmethod
    def apply_location(self, obs, cm_hist, cm_future, **kwargs):
        """
        Applies the debiaser at one location.

        Parameters
        ----------
        obs : np.ndarray
            1-dimensional numpy array of observations of the meteorological variable at one location.
        cm_hist : np.ndarray
            1-dimensional numpy array of values of the historical climate model run (run during the same or a similar period as observations) at one location.
        cm_future : np.ndarray
            1-dimensional numpy array of values of a climate model to debias (future run) at one location.

        Returns
        -------
        np.ndarray
            1-dimensional numpy array containing the debiased climate model values for the future run (cm_future).
        """
        pass

    def apply(
        self,
        obs,
        cm_hist,
        cm_future,
        progressbar=True,
        parallel=False,
        nr_processes=4,
        failsafe=False,
        **kwargs,
    ):
        """
        Applies the debiaser onto given data.

        Parameters
        ----------
        obs : np.ndarray
            3-dimensional numpy array of observations of the meteorological variable. The first dimension should correspond to temporal steps and the 2nd and 3rd one to locations.
        cm_hist : np.ndarray
            3-dimensional numpy array of values of a climate model run during the same or a similar period as observations (historical run). The first dimension should correspond to temporal steps and the 2nd and 3rd one to locations. Shape in the 2nd and 3rd dimension needs to be the same as for obs.
        cm_future : np.ndarray
            3-dimensional numpy array of values of a climate model to debias (future run).  The first dimension should correspond to temporal steps and the 2nd and 3rd one to locations. Shape in the 2nd and 3rd dimension needs to be the same as for obs.
        progressbar : bool = False
            Whether a progressbar shall be shown to indicate the debiaser status. Default: ``True``.
        parallel : bool = False
            Whether the debiasing shall be executed in parallel. No progressbar is shown in this case. Default: ``False``.
        nr_processes : int = 4
            Number of processes for parallel code execution. Default: 4.
        failsafe : bool = False
            Whether execution shall run in a failsafe mode. Debiasing then continues if it encounters an error at one location and returns ``np.nan`` at this location. Default: ``False``.

        Returns
        -------
        np.ndarray
            3-dimensional numpy array containing the debiased climate model values for the future run (cm_future). Has the spatial dimensions as obs, cm_hist, cm_future.
        """

        logger = get_library_logger()
        logger.info("----- Running debiasing for variable: %s -----" % self.variable)

        obs, cm_hist, cm_future = self._check_inputs_and_convert_if_possible(
            obs, cm_hist, cm_future
        )

        if parallel:
            if progressbar:
                warnings.warn("progressbar argument is ignored when parallel = True.")

            output = Debiaser.parallel_map_over_locations(
                self.apply_location,
                output_size=cm_future.shape,
                obs=obs,
                cm_hist=cm_hist,
                cm_future=cm_future,
                nr_processes=nr_processes,
                failsafe=failsafe,
                **kwargs,
            )
        else:
            output = Debiaser.map_over_locations(
                self.apply_location,
                output_size=cm_future.shape,
                obs=obs,
                cm_hist=cm_hist,
                cm_future=cm_future,
                progressbar=progressbar,
                failsafe=failsafe,
                **kwargs,
            )

        self._check_output(output)

        return output
