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

from ibicus.debias import (
    ECDFM,
    ISIMIP,
    CDFt,
    DeltaChange,
    LinearScaling,
    QuantileDeltaMapping,
    QuantileMapping,
    ScaledDistributionMapping,
)


class TestAllDebiasers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

    def test_from_variable_and_apply(self):
        debiaser_classes = [
            QuantileMapping,
            QuantileDeltaMapping,
            CDFt,
            ScaledDistributionMapping,
            ISIMIP,
            LinearScaling,
            ECDFM,
            DeltaChange,
        ]
        for debiaser_class in debiaser_classes:
            # debiaser = debiaser_class()
            # assert debiaser.variable == "unknown"

            # tas
            debiaser = debiaser_class.from_variable("tas")
            assert debiaser.variable == "Daily mean near-surface air temperature"

            obs = np.random.normal(size=(10000, 2, 2)) + 270
            cm_hist = np.random.normal(size=(10000, 2, 2)) + 270
            cm_future = np.random.normal(size=(10000, 2, 2)) + 270 + 2

            debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)
            assert debiased_cm_future.shape == cm_future.shape
            assert np.all(np.isfinite(debiased_cm_future))
            assert np.abs(np.mean(debiased_cm_future) - 2 - np.mean(obs)) < 0.1

            # pr
            debiaser = debiaser_class.from_variable("pr")
            assert debiaser.variable == "Daily mean precipitation flux"

            obs = np.exp(np.random.normal(size=(10000, 2, 2)))
            cm_hist = np.exp(np.random.normal(size=(10000, 2, 2)))
            cm_future = np.exp(np.random.normal(size=(10000, 2, 2))) * 2

            debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)
            assert debiased_cm_future.shape == cm_future.shape
            assert np.all(np.isfinite(debiased_cm_future))
            assert np.abs(np.mean(debiased_cm_future) / 2 - np.mean(obs)) < 0.1

    def test_trend_preservation_tas(self):

        debiasers = [
            QuantileMapping.from_variable("tas", running_window_mode=False),
            QuantileDeltaMapping.from_variable(
                "tas",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            CDFt.from_variable(
                "tas",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            ScaledDistributionMapping.from_variable("tas", running_window_mode=False),
            ISIMIP.from_variable("tas", running_window_mode=False),
            LinearScaling.from_variable("tas", running_window_mode=False),
            ECDFM.from_variable("tas", running_window_mode=False),
            DeltaChange.from_variable("tas", running_window_mode=False),
        ]

        for debiaser in debiasers:
            n = 1000
            np.random.seed(1234)
            for mean_obs in [5]:
                for scale_obs in [1.0, 2.0]:
                    for bias in [0, 2, 10]:
                        for scale_bias in [1.0, 2.0]:
                            for trend in [0.0, 10.0, 20.0]:
                                for trend_scale in [1.0, 2.0]:

                                    obs = (
                                        np.random.normal(size=n) * scale_obs + mean_obs
                                    )
                                    cm_hist = (
                                        np.random.normal(size=n)
                                        * scale_obs
                                        * scale_bias
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

                                    debiased_cm_fut = debiaser.apply_location(
                                        obs, cm_hist, cm_fut
                                    )
                                    assert (
                                        np.abs(
                                            np.mean(debiased_cm_fut) - trend - mean_obs
                                        )
                                        < 0.5
                                    )

    def test_trend_preservation_pr(self):

        debiasers = [
            QuantileMapping.from_variable("pr", running_window_mode=False),
            QuantileDeltaMapping.from_variable(
                "pr",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            CDFt.from_variable(
                "pr",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            ScaledDistributionMapping.from_variable("pr", running_window_mode=False),
            ISIMIP.from_variable("pr", running_window_mode=False),
            LinearScaling.from_variable("pr", running_window_mode=False),
            ECDFM.from_variable("pr", running_window_mode=False),
            DeltaChange.from_variable("pr", running_window_mode=False),
        ]

        for debiaser in debiasers:
            n = 1000
            np.random.seed(1234)
            for scale_obs in [1.0, 2.0]:
                for scale_bias in [1.0]:
                    for trend_scale in [1.0, 3.0]:

                        obs = np.exp(np.random.normal(size=n)) * scale_obs
                        cm_hist = (
                            np.exp(np.random.normal(size=n)) * scale_obs * scale_bias
                        )

                        cm_fut = (
                            np.exp(np.random.normal(size=n))
                            * scale_obs
                            * scale_bias
                            * trend_scale
                        )

                        debiased_cm_fut = debiaser.apply_location(obs, cm_hist, cm_fut)

                        assert (
                            np.abs(
                                np.mean(debiased_cm_fut) / trend_scale - np.mean(obs)
                            )
                            < 0.5
                        )

    def test_perfect_match_tas(self):
        debiasers = [
            QuantileMapping.from_variable("tas", running_window_mode=False),
            QuantileDeltaMapping.from_variable(
                "tas",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            CDFt.from_variable(
                "tas",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            ScaledDistributionMapping.from_variable("tas", running_window_mode=False),
            ISIMIP.from_variable("tas", running_window_mode=False),
            ECDFM.from_variable("tas", running_window_mode=False),
        ]
        for debiaser in debiasers:
            n = 10000
            np.random.seed(1234)
            for mean_obs in [270]:
                for scale_obs in [20.0, 30.0]:
                    for bias in [0, 2, 10]:
                        for scale_bias in [1.0, 2.0]:
                            obs = np.random.normal(size=n) * scale_obs + mean_obs
                            cm_hist = (
                                np.random.normal(size=n) * scale_obs * scale_bias
                                + mean_obs
                                + bias
                            )

                            deb = debiaser.apply_location(obs, cm_hist, cm_hist)
                            assert scipy.stats.kstest(obs, deb).pvalue > 0.05

    def test_perfect_match_pr(self):
        debiasers = [
            QuantileMapping.from_variable("pr", running_window_mode=False),
            CDFt.from_variable(
                "pr",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            QuantileDeltaMapping.from_variable(
                "pr",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            ScaledDistributionMapping.from_variable(
                "pr",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            ),
            ISIMIP.from_variable("pr", running_window_mode=False),
            ECDFM.from_variable("pr", running_window_mode=False),
        ]

        def generate_pr(size, prop_zeros=0.3):
            arr = np.zeros(size)
            arr[int(arr.size * prop_zeros) :] = scipy.stats.gamma(5, 1).rvs(
                size=arr.size - int(arr.size * prop_zeros)
            )
            np.random.shuffle(arr)
            return arr

        for debiaser in debiasers:
            n = 10000
            np.random.seed(1234)
            for mult_bias in [1, 1.1, 1.25, 1.5, 2.0, 5.0]:
                for prop_zeros_cm_hist in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]:
                    obs = generate_pr(n, 0.3)
                    cm_hist = generate_pr(n, prop_zeros_cm_hist) * mult_bias
                    deb = debiaser.apply_location(
                        obs.copy(), cm_hist.copy(), cm_hist.copy()
                    )
                    if not (
                        debiaser.__class__.__name__ == "ScaledDistributionMapping"
                        and prop_zeros_cm_hist >= 0.3
                    ):
                        # Perfect match is not ensured for SDM if the number of dry days in cm is larger than in obs
                        assert scipy.stats.kstest(obs, deb).pvalue > 0.05
