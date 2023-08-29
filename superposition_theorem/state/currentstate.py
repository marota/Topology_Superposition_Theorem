# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

import numpy as np

from ..staticgridprop import StaticGridProp


class CurrentState(object):
    """This class represents the continuous part of the grid state"""   
      
    def __init__(self, gridprop : StaticGridProp) -> None:
        self.ptr_gridprop = gridprop
        
        # actual state (flows, etc.)
        empty_array = np.ndarray((0, ))
        self.p_or : np.ndarray = empty_array
        self.p_ex : np.ndarray = empty_array
        self.theta_or : np.ndarray = empty_array
        self.theta_ex : np.ndarray = empty_array
        self.delta_theta : np.ndarray = empty_array
        self.load_p : np.ndarray = empty_array
        self.gen_p : np.ndarray = empty_array
        self.storage_p : np.ndarray = empty_array
        
    def from_grid2op_obs(self, obs : "grid2op.Observation.BaseObservation") -> None:
        self.p_or : np.ndarray = 1.0 * obs.p_or
        self.p_ex : np.ndarray = 1.0 * obs.p_ex
        self.theta_or : np.ndarray = 1.0 * obs.theta_or
        self.theta_ex : np.ndarray = 1.0 * obs.theta_ex
        self.delta_theta : np.ndarray = self.theta_or - self.theta_ex
        self.load_p : np.ndarray = 1.0 * obs.load_p
        self.gen_p : np.ndarray = 1.0 * obs.gen_p
        self.storage_p : np.ndarray = 1.0 * obs.storage_power
