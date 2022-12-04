# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Demo tests for raw template.
"""

import unittest

import numpy as np
import scipy.stats

from ibicus.utils import (
    PrecipitationGammaLeftCensoredModel_5mm_threshold,
    PrecipitationGammaModelIgnoreZeroValues,
    PrecipitationHurdleModelGamma,
    StatisticalModel,
    ecdf,
    gen_PrecipitationGammaLeftCensoredModel,
    gen_PrecipitationHurdleModel,
    gen_PrecipitationIgnoreZeroValuesModel,
    iecdf,
    quantile_map_non_parametically,
    quantile_map_non_parametically_with_constant_extrapolation,
    quantile_map_x_on_y_non_parametically,
)


def nested_tuples_similar_up_to_diff(tuple1, tuple2, diff):
    if len(tuple1) != len(tuple2):
        return False

    for elem1, elem2 in zip(tuple1, tuple2):
        if isinstance(elem1, tuple):
            if isinstance(elem2, tuple):
                return nested_tuples_similar_up_to_diff(elem1, elem2, diff)
            else:
                return False
        if not np.abs(elem1 - elem2) < diff:
            return False
    return True


class TestStatisticalModel(unittest.TestCase):
    ESTIMATION_MAX_DIFF = 0.35

    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_abstract_StatisticalModel_class(self):
        with self.assertRaises(TypeError):
            StatisticalModel()

    def test_gen_PrecipitationIgnoreZeroValuesModel(self):
        model1 = gen_PrecipitationIgnoreZeroValuesModel()
        model2 = gen_PrecipitationIgnoreZeroValuesModel(distribution=scipy.stats.gamma)
        model3 = gen_PrecipitationIgnoreZeroValuesModel(
            distribution=scipy.stats.weibull_min, fit_kwds=None
        )
        model4 = gen_PrecipitationIgnoreZeroValuesModel(
            fit_kwds={"floc": 0, "fscale": 2}
        )

        data1 = np.concatenate(
            [scipy.stats.gamma.rvs(scale=2, a=3, size=1000), np.repeat(0, 100)]
        )
        data2 = np.concatenate(
            [scipy.stats.weibull_min.rvs(scale=2, c=3, size=1000), np.repeat(0, 100)]
        )

        # Test init
        assert model1 == model2
        assert model1 != model3
        assert model2 == PrecipitationGammaModelIgnoreZeroValues
        assert model1 != model4

        # Test fit
        fit1 = model1.fit(data1)
        fit2 = model3.fit(data2)
        fit3 = model1.fit(data1[data1 != 0])
        fit4 = model4.fit(data1)

        assert nested_tuples_similar_up_to_diff(
            fit1, (3, 0, 2), self.ESTIMATION_MAX_DIFF
        )
        assert fit1[1] == 0  # loc fixed to zero
        assert fit4[1] == 0 and fit4[2] == 2  # loc and scale fixed to zero and two
        assert nested_tuples_similar_up_to_diff(
            fit2, (3, 0, 2), self.ESTIMATION_MAX_DIFF
        )
        assert np.allclose(fit1, fit3)

        # Test cdf
        cdf_vals = model1.cdf(data1, *(2, 0, 3))
        assert np.array_equal(np.where(np.isinf(cdf_vals)), np.where(data1 == 0))

        # Test ppf
        q = np.concatenate([np.random.random(1000), np.repeat(-np.inf, 100)])
        ppf_vals = model1.ppf(q, *(2, 0, 3))

        assert np.array_equal(np.where(np.isinf(ppf_vals)), np.where(q == 0))

        # Test positive inf
        assert model1.ppf(np.inf, *(2, 0, 3)) != 0

        # Test ppf vs cdf
        assert np.allclose(data1, model1.ppf(model1.cdf(data1, *(2, 0, 3)), *(2, 0, 3)))

        # Test cdf vs ppf
        assert np.allclose(q, model1.cdf(model1.ppf(q, *(2, 0, 3)), *(2, 0, 3)))

    def test_gen_PrecipitationHurdleModel(self):
        model1 = gen_PrecipitationHurdleModel()
        model2 = gen_PrecipitationHurdleModel(distribution=scipy.stats.gamma)
        model3 = gen_PrecipitationHurdleModel(fit_kwds={"floc": 0, "fscale": 1})

        nr_dry_days = scipy.stats.binom.rvs(n=1000, p=0.6)
        data1 = np.concatenate(
            [
                scipy.stats.gamma.rvs(scale=2, a=3, size=1000 - nr_dry_days),
                np.repeat(0, nr_dry_days),
            ]
        )
        data2 = np.concatenate(
            [
                scipy.stats.gamma.rvs(scale=1, a=3, size=1000 - nr_dry_days),
                np.repeat(0, nr_dry_days),
            ]
        )

        # Test init
        assert model1 == model2
        assert model1 != model3
        assert model2 == PrecipitationHurdleModelGamma

        # Test fit
        fit1 = model1.fit(data1)
        fit2 = model1.fit(data1[data1 != 0])
        fit3 = model1.fit(data2)
        fit4 = model3.fit(data2)

        assert nested_tuples_similar_up_to_diff(
            fit1, (0.6, (3, 0, 2)), self.ESTIMATION_MAX_DIFF
        )
        assert fit1[1][1] == 0  # loc fixed to 0
        assert np.allclose(fit1[1], fit2[1])
        assert nested_tuples_similar_up_to_diff(fit3, fit4, self.ESTIMATION_MAX_DIFF)
        assert nested_tuples_similar_up_to_diff(
            fit3, (0.6, (3, 0, 1)), self.ESTIMATION_MAX_DIFF
        )

        # Test cdf
        cdf_vals = model1.cdf(data1, *(0.6, (3, 0, 2)))
        assert np.array_equal(np.where(cdf_vals <= 0.6), np.where(data1 == 0))

        # Test ppf
        q = np.random.random(1000)
        ppf_vals = model1.ppf(q, *(0.6, (3, 0, 2)))

        assert np.array_equal(np.where(ppf_vals == 0), np.where(q <= 0.6))

        # Test ppf vs cdf
        assert np.allclose(
            data1, model1.ppf(model1.cdf(data1, *(0.6, (3, 0, 2))), *(0.6, (3, 0, 2)))
        )

        # Test cdf vs ppf
        ppf_vals = model1.ppf(q, *(0.6, (3, 0, 2)))
        vals_not_zero = ppf_vals > 0
        assert np.allclose(
            q[vals_not_zero], model1.cdf(ppf_vals, *(0.6, (3, 0, 2)))[vals_not_zero]
        )

    def test_gen_PrecipitationGammaLeftCensoredModel(self):
        # Test init
        model1 = gen_PrecipitationGammaLeftCensoredModel()
        model2 = gen_PrecipitationGammaLeftCensoredModel(censoring_threshold=0.05)

        with self.assertRaises(ValueError):
            gen_PrecipitationGammaLeftCensoredModel(censoring_threshold=0.0)
        assert model1 != model2
        assert model2 == PrecipitationGammaLeftCensoredModel_5mm_threshold

        # Test fit
        data1 = scipy.stats.gamma.rvs(scale=2, a=3, size=1000)
        threshold = np.quantile(data1, 0.4)
        data2 = data1.copy()
        data1[data1 <= threshold] = 0

        model1 = gen_PrecipitationGammaLeftCensoredModel(censoring_threshold=threshold)

        fit1 = model1.fit(data1)
        fit2 = scipy.stats.gamma.fit(data2, floc=0)
        model2 = gen_PrecipitationGammaLeftCensoredModel(
            censoring_threshold=np.min(data2) / 2
        )
        fit3 = model2.fit(data2)

        assert nested_tuples_similar_up_to_diff(fit1, fit2, self.ESTIMATION_MAX_DIFF)
        assert nested_tuples_similar_up_to_diff(fit2, fit3, self.ESTIMATION_MAX_DIFF)
        assert nested_tuples_similar_up_to_diff(
            fit1, (3, 0, 2), self.ESTIMATION_MAX_DIFF
        )

        # Test cdf
        cdf_vals = model1.cdf(data1, *(3, 0, 2))
        assert np.array_equal(
            np.where(cdf_vals <= scipy.stats.gamma.cdf(threshold, *(3, 0, 2))),
            np.where(data1 <= threshold),
        )

        # Test ppf
        q = np.random.random(1000)
        ppf_vals = model1.ppf(q, *(3, 0, 2))

        assert np.array_equal(
            np.where(ppf_vals == 0),
            np.where(q <= scipy.stats.gamma.cdf(threshold, *(3, 0, 2))),
        )

        # Test ppf vs cdf
        assert np.allclose(data1, model1.ppf(model1.cdf(data1, *(3, 0, 2)), *(3, 0, 2)))

        # Test cdf vs ppf: values above threshold
        ppf_vals = model1.ppf(q, *(3, 0, 2))
        vals_above_threshold = ppf_vals > threshold
        assert np.allclose(
            q[vals_above_threshold],
            model1.cdf(ppf_vals, *(3, 0, 2))[vals_above_threshold],
        )


class TestConsistencyOfIecdfandEcdfMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

        cls.x = np.random.random(10000)
        cls.p = np.random.random(1000)
        sorted_p = np.sort(cls.p)
        cls.min_distance = np.max(
            np.abs(sorted_p[0 : (sorted_p.size - 1)] - sorted_p[1:])
        )

    def test_linear_interpolation_against_linear(self):
        assert all(
            np.abs(
                ecdf(
                    self.x,
                    iecdf(self.x, self.p, method="linear"),
                    method="linear_interpolation",
                )
                - self.p
            )
            < self.min_distance
        )

    def test_step_function_against_inverted_cdf(self):
        assert all(
            np.abs(
                ecdf(
                    self.x,
                    iecdf(self.x, self.p, method="inverted_cdf"),
                    method="step_function",
                )
                - self.p
            )
            < self.min_distance
        )

    def test_kernel_density_against_linear(self):
        assert all(
            np.abs(
                ecdf(
                    self.x,
                    iecdf(self.x, self.p, method="linear"),
                    method="kernel_density",
                )
                - self.p
            )
            < self.min_distance
        )

    def test_iecdf_options_against_kernel_density(self):
        for iecdf_option in [
            "inverted_cdf",
            "averaged_inverted_cdf",
            "closest_observation",
            "interpolated_inverted_cdf",
            "hazen",
            "weibull",
            "linear",
            "median_unbiased",
            "normal_unbiased",
        ]:
            assert all(
                np.abs(
                    ecdf(
                        self.x,
                        iecdf(self.x, self.p, method=iecdf_option),
                        method="kernel_density",
                    )
                    - self.p
                )
                < self.min_distance
            )


class TestOtherHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_quantile_map_non_parametically(self):
        vals = np.random.random(100)
        x = np.random.random(1000)
        y = np.random.normal(size=1000)

        # What to compare against
        sorted_vals = np.sort(vals)
        min_distance = np.max(
            np.abs(sorted_vals[0 : (sorted_vals.size - 1)] - sorted_vals[1:])
        )

        # Test inversion
        mapped_vals = quantile_map_non_parametically(x, y, vals)
        assert all(
            np.abs(vals - quantile_map_non_parametically(y, x, mapped_vals))
            < min_distance
        )

        # Test mapping on same vector with translation
        y = x + 100
        assert np.allclose(y, quantile_map_non_parametically(x, y, x))

    def test_quantile_map_non_parametically_with_constant_extrapolation(self):
        vals = np.random.random(100)
        x = np.random.random(1000)
        y = np.random.normal(size=1000)

        # Test inversion
        vals_within_xrange = vals[(vals <= x.max()) & (vals >= x.min())]

        # What to compare against
        sorted_vals = np.sort(vals_within_xrange)
        min_distance = np.max(
            np.abs(sorted_vals[0 : (sorted_vals.size - 1)] - sorted_vals[1:])
        )

        mapped_vals = quantile_map_non_parametically_with_constant_extrapolation(
            x, y, vals
        )
        assert all(
            np.abs(
                vals
                - quantile_map_non_parametically_with_constant_extrapolation(
                    y, x, mapped_vals
                )
            )
            < min_distance
        )

        # Test all values outside xrange have correction of maximum and minimum value respectively
        vals = np.random.uniform(size=100, low=-1, high=2)
        mask_vals_below_xrange = vals <= x.min()
        mask_vals_above_xrange = vals >= x.max()

        mapped_vals = quantile_map_non_parametically_with_constant_extrapolation(
            x, y, vals
        )
        correction_min_and_max_xval = (
            quantile_map_non_parametically_with_constant_extrapolation(
                x, y, np.array([x.min(), x.max()])
            )
            - np.array([x.min(), x.max()])
        )
        assert np.allclose(
            mapped_vals[mask_vals_below_xrange] - vals[mask_vals_below_xrange],
            correction_min_and_max_xval[0],
        )
        assert np.allclose(
            mapped_vals[mask_vals_above_xrange] - vals[mask_vals_above_xrange],
            correction_min_and_max_xval[1],
        )

    def test_quantile_map_x_on_y_non_parametically(self):
        x = np.random.random(1000)

        # Test mapping on same vector
        assert np.allclose(x, quantile_map_x_on_y_non_parametically(x, x))

        # Test mapping on same vector with translation
        y = x + 100
        assert np.allclose(y, quantile_map_x_on_y_non_parametically(x, y))

        # Test mapping on same vector squared
        y = np.square(x)
        assert np.allclose(y, quantile_map_x_on_y_non_parametically(x, y))
