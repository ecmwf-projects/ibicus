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

from PACKAGE_NAME.utils import ecdf, gen_PrecipitationGammaLeftCensoredModel, iecdf


class TestConsistencyOfIecdfandEcdfMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.random(10000)
        cls.p = np.random.random(1000)
        sorted_p = np.sort(cls.p)
        cls.min_distance = np.max(np.abs(sorted_p[0 : (sorted_p.size - 1)] - sorted_p[1:]))

    def test_linear_interpolation_against_linear(self):
        assert all(
            np.abs(ecdf(self.x, iecdf(self.x, self.p, method="linear"), method="linear_interpolation") - self.p)
            < self.min_distance
        )

    def test_step_function_against_inverted_cdf(self):
        assert all(
            np.abs(ecdf(self.x, iecdf(self.x, self.p, method="inverted_cdf"), method="step_function") - self.p)
            < self.min_distance
        )

    def test_kernel_density_against_linear(self):
        assert all(
            np.abs(ecdf(self.x, iecdf(self.x, self.p, method="linear"), method="kernel_density") - self.p)
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
                np.abs(ecdf(self.x, iecdf(self.x, self.p, method=iecdf_option), method="kernel_density") - self.p)
                < self.min_distance
            )


class Testgen_PrecipitationGammaLeftCensoredModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = (1, 0, 3)
        cls.x = scipy.stats.gamma.rvs(*cls.params, size=1000)
        cls.censoring_value = np.quantile(cls.x, 0.1)

        cls.PrecipitationGammaLeftCensoredModel = gen_PrecipitationGammaLeftCensoredModel(cls.censoring_value)
