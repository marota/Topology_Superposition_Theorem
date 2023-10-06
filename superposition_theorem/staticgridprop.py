# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem


class StaticGridProp(object):
    """This class represents the description of the "static" part of a powergrid. 
    
    It maps every modeled elements (line / trafo, load and generator at the time of writing)
    to the substation to which they are connected.
    
    This should not be modified, the grid is considered static for the superposition theorem to hold.
    """
    def __init__(self) -> None:
        self.line_or_subid = []
        self.line_ex_subid = []
        self.load_subid = []
        self.gen_subid = []    
        self.storage_subid = []    
        self.max_bus_per_sub = []
        self.bus_subid = []  # self.bus_subid[sub_id] gives the list of all the buses id (0, ... np.sum(max_bus_per_sub)) of this sub_id (0, ..., n_sub)
    
    def from_grid2op_obs(self, obs : "grid2op.Observation.BaseObservation") -> None:
        self.line_or_subid = 1 * type(obs).line_or_to_subid
        self.line_ex_subid = 1 * type(obs).line_ex_to_subid
        self.load_subid = 1 * type(obs).load_to_subid
        self.gen_subid = 1 * type(obs).gen_to_subid
        self.storage_subid = 1 * type(obs).storage_to_subid
        
        n_sub = type(obs).n_sub
        self.max_bus_per_sub = [2 for _ in range(n_sub)]
        self.bus_subid = [(i, i + n_sub) for i in range(n_sub)]
