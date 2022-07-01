# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""
debias-module: provides functionality to debias climate models
"""

from ._debiaser import *
from ._delta_change import DeltaChange
from ._equidistant_cdf_matching import EquidistantCDFMatching
from ._isimip import ISIMIP
from ._linear_scaling import LinearScaling
from ._quantile_delta_mapping import QuantileDeltaMapping
from ._quantile_mapping import QuantileMapping
from ._scaled_distribution_mapping import ScaledDistributionMapping
