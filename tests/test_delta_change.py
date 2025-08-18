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

from ibicus.debias import DeltaChange


def check_different_maximally_up_to_1(x, y):
    return np.abs(x - y) <= 1


def check_equal_up_to_distance(x, y, distance):
    return all(np.abs(x - y) < distance)


def get_min_distance_in_array(x):
    x = np.sort(x)
    return np.min(np.abs(x[0 : (x.size - 1)] - x[1:]))


class TestDeltaChange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_from_variable(self):

        # Check default arguments
        hurs = DeltaChange.from_variable("hurs")
        assert hurs.delta_type == "multiplicative"

        pr = DeltaChange.from_variable("pr")
        assert pr.delta_type == "multiplicative"

        psl = DeltaChange.from_variable("psl")
        assert psl.delta_type == "additive"

        rlds = DeltaChange.from_variable("rlds")
        assert rlds.delta_type == "additive"

        rsds = DeltaChange.from_variable("rsds")
        assert rsds.delta_type == "multiplicative"

        sfcWind = DeltaChange.from_variable("sfcWind")
        assert sfcWind.delta_type == "multiplicative"

        tas = DeltaChange.from_variable("tas")
        assert tas.delta_type == "additive"

        tasmin = DeltaChange.from_variable("tasmin")
        assert tasmin.delta_type == "additive"

        tasmax = DeltaChange.from_variable("tasmax")
        assert tasmax.delta_type == "additive"

    def test_tas_apply_location(self):
        debiaser = DeltaChange.from_variable("tas")

        n = 10000
        np.random.seed(1234)
        for mean_obs in [0, 5]:
            for scale_obs in [1.0, 2.0]:
                for bias in [0, 1, 2, 10]:
                    for scale_bias in [1.0, 1.5, 2.0]:
                        for trend in [0.0, 10.0, 20.0]:
                            for trend_scale in [1.0, 2.0]:

                                obs = np.random.normal(size=n) * scale_obs + mean_obs
                                cm_hist = (
                                    np.random.normal(size=n) * scale_obs * scale_bias
                                    + mean_obs
                                    + bias
                                )
                                cm_fut = (
                                    np.random.normal(size=n)
                                    * scale_obs
                                    * scale_bias
                                    * trend_scale
                                    + mean_obs
                                    + bias
                                    + trend
                                )

                                dc_obs = debiaser.apply_location(obs, cm_hist, cm_fut)
                                assert len(dc_obs) == len(obs)
                                assert np.abs(np.mean(dc_obs) - trend - mean_obs) < 0.5

    def test_pr_apply_location(self):

        # Systematic tests of tas
        debiaser = DeltaChange.from_variable("pr")

        n = 10000
        np.random.seed(1234)
        for scale_bias in [1.0, 1.5, 2.0]:
            for trend_scale in [1.0, 2.0]:

                obs = np.exp(np.random.normal(size=n))
                cm_hist = np.exp(np.random.normal(size=n)) * scale_bias
                cm_fut = np.exp(np.random.normal(size=n)) * scale_bias * trend_scale

                dc_obs = debiaser.apply_location(obs, cm_hist, cm_fut)
                assert len(dc_obs) == len(obs)
                assert np.abs(np.mean(dc_obs) / trend_scale - np.mean(obs)) < 0.1
