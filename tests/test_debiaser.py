# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for the input-checking, conversion and mapping helpers of the abstract
:py:class:`ibicus.debias.Debiaser` base class.

A concrete subclass (:py:class:`LinearScaling`) is used to exercise the shared
functionality implemented on the base class.
"""

import unittest
import warnings

import numpy as np

from ibicus.debias import Debiaser, LinearScaling


def _make_debiaser(reasonable_physical_range=None):
    return LinearScaling.from_variable(
        "tas", reasonable_physical_range=reasonable_physical_range
    )


class TestDebiaserInputChecks(unittest.TestCase):
    def test_is_correct_type(self):
        assert Debiaser._is_correct_type(np.zeros((3, 1, 1)))
        assert not Debiaser._is_correct_type([1, 2, 3])
        assert not Debiaser._is_correct_type(5)
        assert not Debiaser._is_correct_type("asdf")

    def test_has_correct_shape(self):
        assert Debiaser._has_correct_shape(np.zeros((3, 2, 2)))
        assert not Debiaser._has_correct_shape(np.zeros((3, 2)))
        assert not Debiaser._has_correct_shape(np.zeros(3))

    def test_have_same_shape(self):
        obs = np.zeros((10, 2, 3))
        cm_hist = np.zeros((20, 2, 3))
        cm_future = np.zeros((30, 2, 3))
        assert Debiaser._have_same_shape(obs, cm_hist, cm_future)

        # Different spatial dimensions
        assert not Debiaser._have_same_shape(obs, np.zeros((20, 2, 4)), cm_future)

    def test_contains_inf_nan(self):
        assert not Debiaser._contains_inf_nan(np.array([1.0, 2.0, 3.0]))
        assert Debiaser._contains_inf_nan(np.array([1.0, np.nan, 3.0]))
        assert Debiaser._contains_inf_nan(np.array([1.0, np.inf, 3.0]))

    def test_has_float_dtype(self):
        assert Debiaser._has_float_dtype(np.array([1.0, 2.0]))
        assert not Debiaser._has_float_dtype(np.array([1, 2]))

    def test_is_masked_array(self):
        assert Debiaser._is_masked_array(np.ma.array([1, 2], mask=[0, 1]))
        assert not Debiaser._is_masked_array(np.array([1, 2]))

    def test_masked_array_contains_invalid_values(self):
        assert Debiaser._masked_array_contains_invalid_values(
            np.ma.array([1, 2], mask=[0, 1])
        )
        assert not Debiaser._masked_array_contains_invalid_values(
            np.ma.array([1, 2], mask=[0, 0])
        )

    def test_not_if_or_nan_vals_outside_reasonable_physical_range(self):
        debiaser = _make_debiaser(reasonable_physical_range=[0.0, 10.0])

        # All in range
        assert not debiaser._not_if_or_nan_vals_outside_reasonable_physical_range(
            np.array([0.0, 5.0, 10.0])
        )
        # One out of range
        assert debiaser._not_if_or_nan_vals_outside_reasonable_physical_range(
            np.array([0.0, 5.0, 11.0])
        )
        # inf/nan are tolerated
        assert not debiaser._not_if_or_nan_vals_outside_reasonable_physical_range(
            np.array([0.0, np.nan, np.inf])
        )

        # No range set: always False
        debiaser_no_range = _make_debiaser(reasonable_physical_range=None)
        assert (
            not debiaser_no_range._not_if_or_nan_vals_outside_reasonable_physical_range(
                np.array([-1000.0, 1000.0])
            )
        )


class TestDebiaserConverters(unittest.TestCase):
    def test_convert_to_float_dtype(self):
        out = Debiaser._convert_to_float_dtype(np.array([1, 2, 3]))
        assert Debiaser._has_float_dtype(out)
        assert np.array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_convert_to_float_dtype_raises(self):
        with self.assertRaises(ValueError):
            Debiaser._convert_to_float_dtype(np.array(["a", "b"]))

    def test_fill_masked_array_with_nan(self):
        masked = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
        out = Debiaser._fill_masked_array_with_nan(masked)
        assert np.isnan(out[1])
        assert out[0] == 1.0 and out[2] == 3.0


class TestReasonablePhysicalRangeValidator(unittest.TestCase):
    def test_valid_range(self):
        debiaser = _make_debiaser(reasonable_physical_range=[0.0, 1.0])
        assert debiaser.reasonable_physical_range == [0.0, 1.0]

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            _make_debiaser(reasonable_physical_range=[0.0])
        with self.assertRaises(ValueError):
            _make_debiaser(reasonable_physical_range=[0.0, 1.0, 2.0])

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            _make_debiaser(reasonable_physical_range=["a", "b"])

    def test_lower_not_smaller_than_upper_raises(self):
        with self.assertRaises(ValueError):
            _make_debiaser(reasonable_physical_range=[5.0, 1.0])


class TestDebiaserMappingHelpers(unittest.TestCase):
    def test_unpack_iterable_args_and_get_locationwise_info(self):
        a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        b = np.arange(2 * 3 * 4).reshape(2, 3, 4) + 100
        out = Debiaser._unpack_iterable_args_and_get_locationwise_info(
            1, 2, {"a": a, "b": b}
        )
        assert np.array_equal(out["a"], a[:, 1, 2])
        assert np.array_equal(out["b"], b[:, 1, 2])

    def test_run_func_on_location_and_catch_error_success(self):
        def func(obs, cm_hist, cm_future):
            return obs + cm_hist + cm_future

        out = Debiaser._run_func_on_location_and_catch_error(
            np.array([1.0]), np.array([2.0]), np.array([3.0]), func
        )
        assert out == np.array([6.0])

    def test_run_func_on_location_and_catch_error_failsafe(self):
        def func(obs, cm_hist, cm_future):
            raise RuntimeError("boom")

        # failsafe returns nan instead of raising
        out = Debiaser._run_func_on_location_and_catch_error(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            func,
            failsafe=True,
        )
        assert np.isnan(out)

        # without failsafe the error propagates
        with self.assertRaises(RuntimeError):
            Debiaser._run_func_on_location_and_catch_error(
                np.array([1.0]),
                np.array([2.0]),
                np.array([3.0]),
                func,
                failsafe=False,
            )

    def test_map_over_locations(self):
        def func(obs, cm_hist, cm_future):
            return cm_future * 2

        obs = np.ones((5, 2, 3))
        cm_hist = np.ones((5, 2, 3))
        cm_future = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)

        out = Debiaser.map_over_locations(
            func, cm_future.shape, obs, cm_hist, cm_future, progressbar=False
        )
        assert out.shape == cm_future.shape
        assert np.array_equal(out, cm_future * 2)

    def test_map_over_locations_failsafe(self):
        def func(obs, cm_hist, cm_future):
            # Fail at one specific location
            if obs[0] == 1:
                raise RuntimeError("boom")
            return cm_future

        obs = np.zeros((3, 1, 2))
        obs[:, 0, 1] = 1  # this location fails
        cm_hist = np.zeros((3, 1, 2))
        cm_future = np.ones((3, 1, 2))

        out = Debiaser.map_over_locations(
            func,
            cm_future.shape,
            obs,
            cm_hist,
            cm_future,
            progressbar=False,
            failsafe=True,
        )
        # Working location keeps its value, failing one is nan
        assert np.array_equal(out[:, 0, 0], cm_future[:, 0, 0])
        assert np.all(np.isnan(out[:, 0, 1]))


class TestDebiaserApplyInputValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_apply_wrong_type_raises(self):
        debiaser = _make_debiaser()
        obs = [1, 2, 3]
        cm = np.zeros((10, 1, 1)) + 270
        with self.assertRaises(TypeError):
            debiaser.apply(obs, cm, cm, progressbar=False)

    def test_apply_wrong_shape_raises(self):
        debiaser = _make_debiaser()
        obs = np.zeros(10) + 270  # 1d, not 3d
        cm = np.zeros((10, 1, 1)) + 270
        with self.assertRaises(ValueError):
            debiaser.apply(obs, cm, cm, progressbar=False)

    def test_apply_mismatched_spatial_shape_raises(self):
        debiaser = _make_debiaser()
        obs = np.zeros((10, 2, 2)) + 270
        cm_hist = np.zeros((10, 2, 3)) + 270
        cm_future = np.zeros((10, 2, 2)) + 270
        with self.assertRaises(ValueError):
            debiaser.apply(obs, cm_hist, cm_future, progressbar=False)

    def test_apply_converts_integer_dtype_with_warning(self):
        debiaser = _make_debiaser()
        obs = (np.zeros((100, 1, 1)) + 270).astype(int)
        cm_hist = (np.zeros((100, 1, 1)) + 270).astype(int)
        cm_future = (np.zeros((100, 1, 1)) + 272).astype(int)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = debiaser.apply(obs, cm_hist, cm_future, progressbar=False)

        assert out.shape == cm_future.shape
        assert any("float dtype" in str(w.message) for w in caught)


if __name__ == "__main__":
    unittest.main()
