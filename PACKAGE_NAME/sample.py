# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
SAMPLE MODULE - demonstrates basic code style.
"""

import numpy as np


def speed_direction_to_uv(speed, direction):
    """
    Calculate wind u- and v-components from wind speed and direction.

    Parameters
    ----------
    speed : array
        Array containing value(s) of wind speed.
    wind_v : array
        Array containing values of wind (from) direction, in degrees.

    Returns
    -------
    tuple
        Tuple containing arrays of wind u- and v-components.
    """
    wind_u = speed * np.cos(np.deg2rad(direction))
    wind_v = speed * np.sin(np.deg2rad(direction))
    return wind_u, wind_v
