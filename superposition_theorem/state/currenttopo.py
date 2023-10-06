# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

from typing import Optional, List, Tuple, Dict
import numpy as np
from ..staticgridprop import StaticGridProp


class CurrentTopo(object):
    """This class represents the "topology" part of the grid state"""   
    DISCO_BUS_ID = -1
    def __init__(self, gridprop : StaticGridProp) -> None:
        self.ptr_gridprop = gridprop
        self.ptr_bus_subid = gridprop.bus_subid
        
        self._line_or_bus = None
        self._line_ex_bus = None
        self._load_bus = None
        self._gen_bus = None
        self._storage_bus = None
    
    def _aux_from_g2op(self, DISCO_BUS_ID, obs_xxx_bus, obs_xxx_to_subid):
        return [self.ptr_bus_subid[sub_id][obs_xxx_bus[l_id] - 1] if obs_xxx_bus[l_id] > 0 else DISCO_BUS_ID for l_id, sub_id in enumerate(obs_xxx_to_subid)]
    
    def from_grid2op_obs(self, obs : "grid2op.Observation.BaseObservation") -> None:  
        t_obs = type(obs)  
        cls = type(self)
        self._line_or_bus = self._aux_from_g2op(cls.DISCO_BUS_ID, obs.line_or_bus, t_obs.line_or_to_subid)
        self._line_ex_bus = self._aux_from_g2op(cls.DISCO_BUS_ID, obs.line_ex_bus, t_obs.line_ex_to_subid)
        self._load_bus = self._aux_from_g2op(cls.DISCO_BUS_ID, obs.load_bus, t_obs.load_to_subid)
        self._gen_bus = self._aux_from_g2op(cls.DISCO_BUS_ID, obs.gen_bus, t_obs.gen_to_subid)
        self._storage_bus = self._aux_from_g2op(cls.DISCO_BUS_ID, obs.storage_bus, t_obs.storage_to_subid)

    def _aux_set_bus(self, sub_id, _xxx_bus, xxxs_id):
        """example: self._aux_set_bus(self._load_bus, loads_id)"""
        for l_id, loc_bus in xxxs_id:
            # loc_bus is either 1 or 2 but python is 0 based indexed, so I remove 1
            bid_this_sub = self.ptr_bus_subid[sub_id]  # bid = bus id
            this_bid = bid_this_sub[loc_bus - 1]
            _xxx_bus[l_id] = this_bid
            
    def set_bus(self, *,
                sub_id : int,
                lines_or_id : Optional[List[Tuple[int, int]]] = None,
                lines_ex_id : Optional[List[Tuple[int, int]]] = None,
                loads_id : Optional[List[Tuple[int, int]]]  = None,
                gens_id : Optional[List[Tuple[int, int]]]  = None,
                storages_id : Optional[List[Tuple[int, int]]]  = None) -> None:       
        """internal do not use, use the :class:`SubAction` to modify substations !""" 
        if loads_id is not None:
            self._aux_set_bus(sub_id, self._load_bus, loads_id)
        if gens_id is not None:
            self._aux_set_bus(sub_id, self._gen_bus, gens_id)
        if storages_id is not None:
            self._aux_set_bus(sub_id, self._storage_bus, storages_id)
        if lines_or_id is not None:
            self._aux_set_bus(sub_id, self._line_or_bus, lines_or_id)
        if lines_ex_id is not None:
            self._aux_set_bus(sub_id, self._line_ex_bus, lines_ex_id)
    
    def set_0_other_sub(self, sub_id):
        self._aux_0_other_sub(sub_id, self._line_or_bus, self.ptr_gridprop.line_or_subid)
        self._aux_0_other_sub(sub_id, self._line_ex_bus, self.ptr_gridprop.line_ex_subid)
        self._aux_0_other_sub(sub_id, self._load_bus, self.ptr_gridprop.load_subid)
        self._aux_0_other_sub(sub_id, self._gen_bus, self.ptr_gridprop.gen_subid)
        self._aux_0_other_sub(sub_id, self._storage_bus, self.ptr_gridprop.storage_subid)
        
    def _aux_0_other_sub(self, sub_id, _xxx_bus, xxx_subid):
        for el_id, el_sub in enumerate(xxx_subid):
            if el_sub != sub_id:
                _xxx_bus[el_id] = 0
        return _xxx_bus
    
    def _aux_make_local(self, el_id, global_bus, xxx_sub_id):
        """convert the "global id" in a grid2op "local" bus id"""
        sub_id = xxx_sub_id[el_id]
        where = np.where(np.array(self.ptr_bus_subid[sub_id]) == global_bus)[0]
        res = int(where[0]) + 1
        return res
    
    def _aux_get_grid2op_dict(self, dict_, key, _xxx_bus, xxx_sub_id):
        dict_[key] = [(el_id, self._aux_make_local(el_id, global_bus, xxx_sub_id)) for el_id, global_bus in enumerate(_xxx_bus) if global_bus != 0]
        
    def get_grid2op_dict(self) -> Dict:
        tmp = {}
        if self._line_or_bus:
            self._aux_get_grid2op_dict(tmp, "lines_or_id", self._line_or_bus, self.ptr_gridprop.line_or_subid)
        if self._line_ex_bus:
            self._aux_get_grid2op_dict(tmp, "lines_ex_id", self._line_ex_bus, self.ptr_gridprop.line_ex_subid)
        if self._load_bus:
            self._aux_get_grid2op_dict(tmp, "loads_id", self._load_bus, self.ptr_gridprop.load_subid)
        if self._gen_bus:
            self._aux_get_grid2op_dict(tmp, "generators_id", self._gen_bus, self.ptr_gridprop.gen_subid)
        if self._storage_bus:
            self._aux_get_grid2op_dict(tmp, "storages_id", self._storage_bus, self.ptr_gridprop.storage_subid)
        res = {'set_bus': tmp}
        return res
    
    def __hash__(self) -> int:
        res = hash((tuple(self._line_or_bus),
                    tuple(self._line_ex_bus),
                    tuple(self._load_bus),
                    tuple(self._gen_bus),
                    tuple(self._storage_bus)))
        return res

    def get_p(self, bus_id : int, current_state: "superposition_theorem.state.CurrentState") -> float:
        lor_id = np.where(np.array(self._line_or_bus) == bus_id)[0]
        lex_id = np.where(np.array(self._line_ex_bus) == bus_id)[0]
        load_id = np.where(np.array(self._load_bus) == bus_id)[0]
        gen_id = np.where(np.array(self._gen_bus) == bus_id)[0]
        sto_id = np.where(np.array(self._storage_bus) == bus_id)[0]
        res = current_state.get_p(lines_or_id=lor_id,
                                  lines_ex_id=lex_id,
                                  loads_id=load_id,
                                  gens_id=gen_id,
                                  storages_id=sto_id)
        return res
