# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for the helpers in :py:mod:`ibicus.utils._utils`.
"""

import datetime
import logging
import unittest

import numpy as np
import pandas as pd

from ibicus.utils import (
    create_array_of_consecutive_dates,
    day,
    day_of_year,
    get_chunked_mean,
    get_library_logger,
    get_mask_for_unique_subarray,
    get_pr,
    get_prsn,
    get_prsnratio,
    get_tasmax,
    get_tasmin,
    get_tasmin_tasmax,
    get_tasrange,
    get_tasrange_tasskew,
    get_tasskew,
    get_verbosity_library_logger,
    get_yearly_means,
    get_years_and_yearly_means,
    interp_sorted_cdf_vals_on_given_length,
    month,
    season,
    set_verbosity_library_logger,
    sort_array_like_another_one,
    threshold_cdf_vals,
    year,
)
from ibicus.utils._utils import (
    _check_if_list_of_two_and_unpack_else_none,
    _get_library_name,
    _get_tasmax_from_tasmin_and_range,
    _unpack_df_of_numpy_arrays,
    check_time_information_and_raise_error,
    infer_and_create_time_arrays_if_not_given,
)


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_sort_array_like_another_one(self):
        x = np.random.random(1000)
        y = np.random.random(1000)
        x_sorted_like_y = sort_array_like_another_one(x, y)

        assert all(np.argsort(x_sorted_like_y) == np.argsort(y))

    def test_sort_array_like_another_one_keeps_values(self):
        x = np.random.random(100)
        y = np.random.random(100)
        x_sorted_like_y = sort_array_like_another_one(x, y)

        # The set of values is unchanged, only the order is.
        assert np.array_equal(np.sort(x), np.sort(x_sorted_like_y))

        # The biggest value of x sits at the position of the biggest value of y.
        assert x_sorted_like_y[np.argmax(y)] == x.max()
        assert x_sorted_like_y[np.argmin(y)] == x.min()


class TestArrayReductions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_get_chunked_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Chunks dividing the array evenly
        assert np.array_equal(get_chunked_mean(x, 2), np.array([1.5, 3.5, 5.5]))
        assert np.array_equal(get_chunked_mean(x, 3), np.array([2.0, 5.0]))

        # Whole array is one chunk
        assert np.array_equal(get_chunked_mean(x, x.size), np.array([np.mean(x)]))

        # Chunk size not dividing the array: last chunk is shorter
        out = get_chunked_mean(x, 4)
        assert np.allclose(out, np.array([np.mean(x[:4]), np.mean(x[4:])]))

    def test_get_yearly_means(self):
        x = np.array([1.0, 3.0, 10.0, 20.0, 5.0])
        years = np.array([2000, 2000, 2001, 2001, 2002])

        out = get_yearly_means(x, years)
        assert np.array_equal(out, np.array([2.0, 15.0, 5.0]))

        # Order follows np.unique (sorted unique years)
        years_unsorted = np.array([2002, 2002, 2000, 2000, 2001])
        x2 = np.array([5.0, 7.0, 1.0, 3.0, 9.0])
        assert np.array_equal(
            get_yearly_means(x2, years_unsorted), np.array([2.0, 9.0, 6.0])
        )

    def test_get_years_and_yearly_means(self):
        x = np.array([1.0, 3.0, 10.0, 20.0])
        years = np.array([2000, 2000, 2001, 2001])

        unique_years, means = get_years_and_yearly_means(x, years)
        assert np.array_equal(unique_years, np.array([2000, 2001]))
        assert np.array_equal(means, np.array([2.0, 15.0]))
        assert np.array_equal(means, get_yearly_means(x, years))


class TestCdfHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_threshold_cdf_vals(self):
        cdf_vals = np.array([0.0, 1e-12, 0.5, 1 - 1e-12, 1.0])

        out = threshold_cdf_vals(cdf_vals)

        # Nothing below cdf_threshold or above 1 - cdf_threshold
        assert out.min() >= 1e-10
        assert out.max() <= 1 - 1e-10

        # Values comfortably inside the range are untouched
        assert out[2] == 0.5

        # Boundaries are clipped to the threshold
        assert np.isclose(out[0], 1e-10)
        assert np.isclose(out[-1], 1 - 1e-10)

    def test_threshold_cdf_vals_custom_threshold(self):
        cdf_vals = np.array([0.0, 0.5, 1.0])
        out = threshold_cdf_vals(cdf_vals, cdf_threshold=0.01)
        assert np.allclose(out, np.array([0.01, 0.5, 0.99]))

    def test_interp_sorted_cdf_vals_on_given_length(self):
        cdf_vals = np.array([0.0, 0.5, 1.0])

        # Same length returns the same values
        out_same = interp_sorted_cdf_vals_on_given_length(cdf_vals, cdf_vals.size)
        assert np.allclose(out_same, cdf_vals)

        # Upsampling preserves endpoints and is monotonically increasing
        out_long = interp_sorted_cdf_vals_on_given_length(cdf_vals, 5)
        assert out_long.size == 5
        assert np.isclose(out_long[0], cdf_vals[0])
        assert np.isclose(out_long[-1], cdf_vals[-1])
        assert np.all(np.diff(out_long) >= 0)


class TestDatetimeFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_create_array_of_consecutive_dates(self):
        dates = create_array_of_consecutive_dates(5, "2000-02-27")

        assert dates.size == 5
        assert dates[0] == datetime.date(2000, 2, 27)
        # Leap year: 2000-02-29 exists, so 27, 28, 29, then March
        assert dates[-1] == datetime.date(2000, 3, 2)

        # Default start date
        default_dates = create_array_of_consecutive_dates(3)
        assert default_dates[0] == datetime.date(1950, 1, 1)
        assert default_dates.size == 3

        # Accepts a np.datetime64 start date directly
        dates_dt64 = create_array_of_consecutive_dates(2, np.datetime64("1980-06-01"))
        assert dates_dt64[0] == datetime.date(1980, 6, 1)

    def test_day(self):
        dates = create_array_of_consecutive_dates(5, "2000-02-27")
        assert np.array_equal(day(dates), np.array([27, 28, 29, 1, 2]))

        # np.datetime64 array path
        nd = np.array(["2000-01-15", "2000-12-31"], dtype="datetime64[D]")
        assert np.array_equal(day(nd), np.array([15, 31]))

    def test_month(self):
        dates = create_array_of_consecutive_dates(5, "2000-02-27")
        assert np.array_equal(month(dates), np.array([2, 2, 2, 3, 3]))

        nd = np.array(["2000-01-15", "2000-12-31"], dtype="datetime64[D]")
        assert np.array_equal(month(nd), np.array([1, 12]))

    def test_year(self):
        dates = create_array_of_consecutive_dates(3, "2000-12-30")
        assert np.array_equal(year(dates), np.array([2000, 2000, 2001]))

        nd = np.array(["1999-01-15", "2005-12-31"], dtype="datetime64[D]")
        assert np.array_equal(year(nd), np.array([1999, 2005]))

    def test_day_of_year(self):
        dates = np.array([datetime.date(2001, 1, 1), datetime.date(2001, 12, 31)])
        assert np.array_equal(day_of_year(dates), np.array([1, 365]))

        # Leap years have day 366
        nd = np.array(["2000-12-31"], dtype="datetime64[D]")
        assert np.array_equal(day_of_year(nd), np.array([366]))

    def test_season(self):
        dates = np.array(
            [
                datetime.date(2000, 1, 15),
                datetime.date(2000, 4, 15),
                datetime.date(2000, 7, 15),
                datetime.date(2000, 10, 15),
                datetime.date(2000, 12, 15),
            ]
        )
        assert np.array_equal(
            season(dates),
            np.array(["Winter", "Spring", "Summer", "Autumn", "Winter"]),
        )

    def test_datetime_functions_raise_on_invalid_objects(self):
        invalid = np.array([object()])
        for func in (day, month, year, day_of_year):
            with self.assertRaises(ValueError):
                func(invalid)


class TestTimeInformationHelpers(unittest.TestCase):
    def test_infer_and_create_time_arrays_if_not_given_all_none(self):
        obs = np.zeros(5)
        cm_hist = np.zeros(7)
        cm_future = np.zeros(3)

        time_obs, time_cm_hist, time_cm_future = (
            infer_and_create_time_arrays_if_not_given(obs, cm_hist, cm_future)
        )

        assert time_obs.size == obs.size
        assert time_cm_hist.size == cm_hist.size
        assert time_cm_future.size == cm_future.size

    def test_infer_and_create_time_arrays_if_not_given_partial(self):
        obs = np.zeros(5)
        cm_hist = np.zeros(7)
        cm_future = np.zeros(3)
        given_time_obs = create_array_of_consecutive_dates(5)

        time_obs, time_cm_hist, time_cm_future = (
            infer_and_create_time_arrays_if_not_given(
                obs, cm_hist, cm_future, time_obs=given_time_obs
            )
        )

        # The given array is passed through unchanged
        assert np.array_equal(time_obs, given_time_obs)
        # The missing ones are inferred
        assert time_cm_hist.size == cm_hist.size
        assert time_cm_future.size == cm_future.size

    def test_check_time_information_and_raise_error_valid(self):
        obs = np.zeros(5)
        cm_hist = np.zeros(7)
        cm_future = np.zeros(3)
        time_obs = np.zeros(5)
        time_cm_hist = np.zeros(7)
        time_cm_future = np.zeros(3)

        # Should not raise
        check_time_information_and_raise_error(
            obs, cm_hist, cm_future, time_obs, time_cm_hist, time_cm_future
        )

    def test_check_time_information_and_raise_error_invalid(self):
        obs = np.zeros(5)
        cm_hist = np.zeros(7)
        cm_future = np.zeros(3)

        with self.assertRaises(ValueError):
            check_time_information_and_raise_error(
                obs,
                cm_hist,
                cm_future,
                np.zeros(4),  # wrong size for obs
                np.zeros(7),
                np.zeros(3),
            )


class TestTemperatureVariableConversions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)
        cls.tasmin = np.random.uniform(250, 280, size=1000)
        cls.tasmax = cls.tasmin + np.random.uniform(1, 15, size=1000)
        cls.tas = (cls.tasmin + cls.tasmax) / 2

    def test_get_tasrange(self):
        tasrange = get_tasrange(self.tasmin, self.tasmax)
        assert np.allclose(tasrange, self.tasmax - self.tasmin)
        assert np.all(tasrange > 0)

    def test_get_tasskew(self):
        tasskew = get_tasskew(self.tas, self.tasmin, self.tasmax)
        assert np.allclose(
            tasskew, (self.tas - self.tasmin) / (self.tasmax - self.tasmin)
        )
        # tas is the midpoint, so skew should be 0.5
        assert np.allclose(tasskew, 0.5)

    def test_get_tasmin(self):
        tasrange, tasskew = get_tasrange_tasskew(self.tas, self.tasmin, self.tasmax)
        assert np.allclose(get_tasmin(self.tas, tasrange, tasskew), self.tasmin)

    def test_get_tasmax(self):
        tasrange, tasskew = get_tasrange_tasskew(self.tas, self.tasmin, self.tasmax)
        assert np.allclose(get_tasmax(self.tas, tasrange, tasskew), self.tasmax)

    def test_get_tasmin_tasmax(self):
        tasrange, tasskew = get_tasrange_tasskew(self.tas, self.tasmin, self.tasmax)
        tasmin, tasmax = get_tasmin_tasmax(self.tas, tasrange, tasskew)
        assert np.allclose(tasmin, self.tasmin)
        assert np.allclose(tasmax, self.tasmax)

    def test_get_tasrange_tasskew_roundtrip(self):
        tasrange, tasskew = get_tasrange_tasskew(self.tas, self.tasmin, self.tasmax)
        assert np.allclose(tasrange, get_tasrange(self.tasmin, self.tasmax))
        assert np.allclose(tasskew, get_tasskew(self.tas, self.tasmin, self.tasmax))

    def test_full_roundtrip_tas_tasmin_tasmax(self):
        # tas, tasmin, tasmax -> tasrange, tasskew -> tasmin, tasmax
        tasrange, tasskew = get_tasrange_tasskew(self.tas, self.tasmin, self.tasmax)
        tasmin, tasmax = get_tasmin_tasmax(self.tas, tasrange, tasskew)
        assert np.allclose(tasmin, self.tasmin)
        assert np.allclose(tasmax, self.tasmax)

    def test_get_tasmax_from_tasmin_and_range(self):
        tasrange = get_tasrange(self.tasmin, self.tasmax)
        assert np.allclose(
            _get_tasmax_from_tasmin_and_range(tasrange, self.tasmin), self.tasmax
        )


class TestPrecipitationVariableConversions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)
        cls.pr = np.random.uniform(0.1, 10, size=1000)
        cls.prsn = cls.pr * np.random.uniform(0, 1, size=1000)

    def test_get_prsnratio(self):
        prsnratio = get_prsnratio(self.pr, self.prsn)
        assert np.allclose(prsnratio, self.prsn / self.pr)
        assert np.all((prsnratio >= 0) & (prsnratio <= 1))

    def test_get_pr_roundtrip(self):
        prsnratio = get_prsnratio(self.pr, self.prsn)
        assert np.allclose(get_pr(self.prsn, prsnratio), self.pr)

    def test_get_prsn_roundtrip(self):
        prsnratio = get_prsnratio(self.pr, self.prsn)
        assert np.allclose(get_prsn(self.pr, prsnratio), self.prsn)


class TestMiscHelpers(unittest.TestCase):
    def test_get_mask_for_unique_subarray(self):
        x = np.array([1, 1, 2, 2, 3])
        mask = get_mask_for_unique_subarray(x)

        # The first occurrence of each value is kept
        assert np.array_equal(mask, np.array([True, False, True, False, True]))
        # Masked values form a set of unique values
        assert np.array_equal(np.unique(x[mask]), np.unique(x))
        assert x[mask].size == np.unique(x).size

    def test_check_if_list_of_two_and_unpack_else_none(self):
        assert _check_if_list_of_two_and_unpack_else_none([1, 2]) == (1, 2)
        assert _check_if_list_of_two_and_unpack_else_none((9, 8)) == (9, 8)
        assert _check_if_list_of_two_and_unpack_else_none(5) == (5, None)
        assert _check_if_list_of_two_and_unpack_else_none("abc") == ("abc", None)

        with self.assertRaises(ValueError):
            _check_if_list_of_two_and_unpack_else_none([1, 2, 3])

    def test_unpack_df_of_numpy_arrays(self):
        df = pd.DataFrame(
            {
                "label": ["a", "b"],
                "values": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            }
        )

        expanded = _unpack_df_of_numpy_arrays(df, "values")

        # Each numpy array is flattened into individual rows
        assert expanded.shape[0] == 4
        assert np.array_equal(
            np.sort(expanded["values"].to_numpy()), np.array([1.0, 2.0, 3.0, 4.0])
        )
        # The non-numpy column is repeated for each element
        assert list(expanded["label"]) == ["a", "a", "b", "b"]
        # The numpy column has been converted to numeric
        assert pd.api.types.is_numeric_dtype(expanded["values"])


class TestLoggingHelpers(unittest.TestCase):
    def test_get_library_name(self):
        assert _get_library_name() == "ibicus"

    def test_get_library_logger(self):
        logger = get_library_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "ibicus"

    def test_set_and_get_verbosity(self):
        original = get_verbosity_library_logger()
        try:
            set_verbosity_library_logger(logging.ERROR)
            assert get_verbosity_library_logger() == logging.ERROR

            set_verbosity_library_logger(logging.WARNING)
            assert get_verbosity_library_logger() == logging.WARNING
        finally:
            set_verbosity_library_logger(original)
