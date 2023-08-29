# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

import copy
import numpy as np
from typing import Optional, List, Tuple, Dict, Literal, Set


from .staticgridprop import StaticGridProp
from .state import CurrentState, CurrentTopo

                                 
class SubAction(object):
    """This class represents an action on a given substation. 
    
    It cannot act on multiple substation at the moment.
    """
    OR_ID = 1
    EX_ID = 2
    def __init__(self,
                 gridprop: StaticGridProp,
                 current_state: CurrentState,
                 current_topo: CurrentTopo) -> None:
        self.ptr_gridprop = gridprop
        self.ptr_bus_subid = gridprop.bus_subid
        self.ptr_current_state = current_state
        self.ptr_current_topo = current_topo
        
        self._sub_id = None
        
        self._line_or_ex = None
        self._load_id = None  # loads connected to this sub
        self._gen_id = None  # gen connected to this sub
        self._storage_id = None  # storage connected to this sub
        
        # bus after the action
        self.bus_after_action = copy.deepcopy(self.ptr_current_topo)
        
    @property
    def sub_id(self):
        return self._sub_id

    @sub_id.setter
    def sub_id(self, value):
        raise RuntimeError("Impossible to change the sub_id like this. You need to call `act.set_subid(...)`")
        
    def set_subid(self, sub_id):
        return self.sub_modif(sub_id)
    
    def sub_modif(self, sub_id):
        """set the id of the substation modified"""
        self._sub_id = sub_id
        cls = type(self)
        
        self._load_id = np.zeros(self.ptr_gridprop.load_subid.size, dtype=int) 
        self._load_id[self.ptr_gridprop.load_subid == sub_id] = 1
        self._gen_id = np.zeros(self.ptr_gridprop.gen_subid.size, dtype=int) 
        self._gen_id[self.ptr_gridprop.gen_subid == sub_id] = 1
        self._storage_id = np.zeros(self.ptr_gridprop.storage_subid.size, dtype=int) 
        self._storage_id[self.ptr_gridprop.storage_subid == sub_id] = 1
        
        self._line_or_ex = np.zeros(self.ptr_gridprop.line_or_subid.size, dtype=int)  # 1 for or, 2 for ex, 0 for "not in sub"
        self._line_or_ex[self.ptr_gridprop.line_or_subid == sub_id] = cls.OR_ID
        self._line_or_ex[self.ptr_gridprop.line_ex_subid == sub_id] = cls.EX_ID        
        
        self.bus_after_action.set_0_other_sub(sub_id)
    
    def _aux_check_act(self, nm, xxxs_id, xxx_subid):
        for el_id, bus_id in xxxs_id:
            if xxx_subid[el_id] != self.sub_id:
                raise RuntimeError(f"{nm} id {el_id} is connected to sub {xxx_subid[el_id]} but this action is for sub {self.sub_id}")
            if bus_id >= len(self.ptr_bus_subid):
                raise RuntimeError(f"For {nm} {el_id}: you ask to connect it to bus {bus_id}, but its substation (id {self.sub_id}) "
                                   f"counts only {len(self.ptr_bus_subid)} busbars.")
                
    def _check_act(self, lines_id=None, loads_id=None, gens_id=None, sotrages_id=None) -> None:
        if loads_id is not None:
            self._aux_check_act("load", loads_id, self.ptr_gridprop.load_subid)
        if gens_id is not None:
            self._aux_check_act("gen", gens_id, self.ptr_gridprop.gen_subid)
        if sotrages_id is not None:
            self._aux_check_act("storage", sotrages_id, self.ptr_gridprop.storage_subid)
        if lines_id is not None:
            lines_or_id, lines_ex_id = self._aux_split_lines_id(lines_id)
            self._aux_check_act("line (or)", lines_or_id, self.ptr_gridprop.line_or_subid)
            self._aux_check_act("line (ex)", lines_ex_id, self.ptr_gridprop.line_ex_subid)
    
    def _aux_split_lines_id(self, lines_id):
        cls = type(self)
        lines_or_id = [el for el in lines_id if self._line_or_ex[el[0]] == cls.OR_ID]
        lines_ex_id = [el for el in lines_id if self._line_or_ex[el[0]] == cls.EX_ID]
        return lines_or_id, lines_ex_id
        
    def set_bus(self, *,
                lines_id : Optional[List[Tuple[int, int]]] = None,
                loads_id : Optional[List[Tuple[int, int]]]  = None,
                gens_id : Optional[List[Tuple[int, int]]]  = None,
                storages_id : Optional[List[Tuple[int, int]]]  = None):
        if self.sub_id is None:
            raise RuntimeError("You need to set the sub id on which you want to act with `act.set_subid(...)`")
        self._check_act()
        lines_or_id = None
        lines_ex_id = None
        if lines_id is not None:
            lines_or_id, lines_ex_id = self._aux_split_lines_id(lines_id)
            
        self.bus_after_action.set_bus(sub_id=self.sub_id,
                                      lines_or_id=lines_or_id,
                                      lines_ex_id=lines_ex_id,
                                      loads_id=loads_id,
                                      gens_id=gens_id,
                                      storages_id=storages_id)
            
    def get_elem_sub(self) -> Dict[Literal["line", "load", "gen", "storage"], Set]:
        res = {}
        if (self._line_or_ex != 0).any():
            res["line"] = np.where(self._line_or_ex != 0)[0]
        if self._load_id.size:
            res["load"] = 1 * self._load_id
        if self._gen_id.size:
            res["gen"] = 1 * self._gen_id
        if self._storage_id.size:
            res["storage"] = 1* self._storage_id
        return res
    
    @classmethod
    def from_grid2op(self, state: "State", act: "grid2op.Action.BaseAction"):
        # TODO
        raise NotImplementedError()
    
    def to_grid2op(self, act_space):
        dict_act = self.bus_after_action.get_grid2op_dict()
        return act_space(dict_act)
    
    def __hash__(self) -> int:
        """convert the representation of a substation reconfiguration into a hashable"""
        res = hash((self._sub_id,
                    tuple(self._line_or_ex.tolist()),
                    tuple(self._load_id.tolist()),
                    tuple(self._gen_id.tolist()),
                    tuple(self._storage_id.tolist()),
                    self.bus_after_action)
                  )
        return res