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

from PACKAGE_NAME.sample import speed_direction_to_uv


class TestSpeedDirectionToUV(unittest.TestCase):
    def test_zero_speed(self):
        # Test that a wind speed of 0 results in u and v values of 0
        self.assertEqual(speed_direction_to_uv(0, 10), (0, 0))

    def test_zero_direction(self):
        # Test that a wind direction of 0 results u==speed and v==0
        self.assertEqual(speed_direction_to_uv(10, 0), (10, 0))

    def test_180_direction(self):
        # Test that a wind direction of 180 results u==-speed and v==0
        wind_u, wind_v = speed_direction_to_uv(10, 180)
        self.assertEqual(wind_u, -10)
        self.assertAlmostEqual(wind_v, 0)

    def test_90_direction(self):
        # Test that a wind direction of 90 results u==0 and v==speed
        wind_u, wind_v = speed_direction_to_uv(10, 90)
        self.assertAlmostEqual(wind_u, 0)
        self.assertEqual(wind_v, 10)
