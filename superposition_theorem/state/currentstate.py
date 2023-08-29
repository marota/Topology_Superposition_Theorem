# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

from typing import List, Dict, Literal, Set, Tuple, Optional
import numpy as np
import copy

from ..staticgridprop import StaticGridProp


class CurrentState(object):
    """This class represents the continuous part of the grid state"""   
      
    def __init__(self, gridprop : StaticGridProp) -> None:
        self.ptr_gridprop = gridprop
        
        # actual state (flows, etc.)
        self.p_or = []
        self.p_ex = []
        self.theta_or = []
        self.theta_ex = []
        self.delta_theta = []
        
    def from_grid2op_obs(self, obs : "grid2op.Observation.BaseObservation") -> None:
        self.p_or = 1.0 * obs.p_or
        self.p_ex = 1.0 * obs.p_ex
        self.theta_or = 1.0 * obs.theta_or
        self.theta_ex = 1.0 * obs.theta_ex
        self.delta_theta = self.theta_or - self.theta_ex
