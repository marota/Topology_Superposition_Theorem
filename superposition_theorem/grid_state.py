# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

from typing import List
import numpy as np


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
    
    def from_grid2op_obs(self, obs : "grid2op.Observation.BaseObservation") -> None:
        self.line_or_subid = 1 * type(obs).line_ex_to_subid
        self.line_ex_subid = 1 * type(obs).line_ex_to_subid
        self.load_subid = 1 * type(obs).load_to_subid
        self.gen_subid = 1 * type(obs).gen_to_subid
        self.storage_subid = 1 * type(obs).storage_to_subid
        
        
class CurrentState(object):
    """This class represents the continuous part of the """   
      
    def __init__(self) -> None:
        # actual state (flows, etc.)
        self.p_or = []
        self.p_ex = []
        self.theta_or = []
        self.theta_ex = []
        self.delta_theta = []
        
        # actual state (topo)
        self.line_or_bus = []
        self.line_ex_bus = []
        self.load_bus = []
        self.gen_bus = []
        self.storage_bus = []
        
    def from_grid2op_obs(self, obs : "grid2op.Observation.BaseObservation") -> None:
        self.p_or = 1.0 * obs.p_or
        self.p_ex = 1.0 * obs.p_ex
        self.theta_or = 1.0 * obs.theta_or
        self.theta_ex = 1.0 * obs.theta_ex
        self.delta_theta = self.theta_or - self.theta_ex
        
        # actual state (topo)
        self.line_or_bus = 1 * obs.line_or_bus
        self.line_ex_bus = 1 * obs.line_ex_bus
        self.load_bus = 1 * obs.load_bus
        self.gen_bus = 1 * obs.gen_bus
        self.storage_bus = 1 * obs.storage_bus
        
        
class State(object):
    DISCO_BUS_ID = -1
    def __init__(self) -> None:
        # grid description
        self._grid : StaticGridProp = StaticGridProp()
        
        # initial state
        self._init_state : CurrentState = CurrentState()
        
        # states after unary actions
        self._unary_states = {}
    
    @classmethod
    def from_grid2op_obs(cls,
                         obs : "grid2op.Observation.BaseObservation",
                         line_ids_disc_unary = (),
                         line_ids_reco_unary = ()) -> "State":
        res = cls()
        
        # grid description
        res._grid.from_grid2op_obs(obs)
        
        # actual state (flows, etc.)
        res._init_state.from_grid2op_obs(obs)
        
        # unary states
        res._unary_states = {}
        if line_ids_disc_unary:
            dict_disco = {}
            if obs._obs_env is None:
                raise RuntimeError("You cannot build a proper Sate if the grid2op observation cannot use obs.simulate")
            
            for l_id in line_ids_disc_unary:
                # simulate the unary disconnection of powerline l_id
                obs_tmp, reward, done, info = obs.simulate(obs._obs_env.action_space({"set_line_status": [(l_id, -1)]}),
                                                           time_step=0)
                if not done:
                    state_tmp = CurrentState()
                    state_tmp.from_grid2op_obs(obs_tmp)
                    dict_disco[l_id] = state_tmp
                else:
                    raise RuntimeError(f"Impossible to disconnect powerline {l_id}: no feasible solution found.")
            res._unary_states["line_disco"] = dict_disco
        
        if line_ids_reco_unary:
            dict_reco = {}
            if obs._obs_env is None:
                raise RuntimeError("You cannot build a proper Sate if the grid2op observation cannot use obs.simulate")
            
            for l_id in line_ids_reco_unary:
                # simulate the unary disconnection of powerline l_id
                obs_tmp, reward, done, info = obs.simulate(obs._obs_env.action_space({"set_line_status": [(l_id, +1)]}),
                                                           time_step=0)
                if not done:
                    state_tmp = CurrentState()
                    state_tmp.from_grid2op_obs(obs_tmp)
                    dict_reco[l_id] = state_tmp
                else:
                    raise RuntimeError(f"Impossible to disconnect powerline {l_id}: no feasible solution found.")
            res._unary_states["line_reco"] = dict_reco
            
        return res

    def compute_betas_disco_lines(self, l_ids):
        # works only for reconnected powerline, but does not requires knowledge of theta
        por_unary = np.array([self._unary_states["line_disco"][l_id].p_or for l_id in l_ids])
        l_ids = np.array(l_ids).astype(int)
        A = por_unary[:, l_ids]
        A /= self._init_state.p_or[l_ids]
        A *= -1.
        A = A.T
        A += 1
        B = np.ones(l_ids.size)
        return por_unary, np.linalg.solve(A, B)
        
    def compute_flows_disco_lines(self, l_ids: List) -> np.ndarray:
        # por_unary, betas = self.compute_betas_disco_lines(l_ids)
        por_unary, betas = self.compute_betas_reco_lines(l_ids, key="line_disco")
        res = (1 - betas.sum()) * self._init_state.p_or
        res += np.matmul(betas, por_unary)    
        return res
    
    def compute_betas_reco_lines(self, l_ids, key="line_reco"):
        # works when flow on powerline is O, but requires theta
        por_unary = np.array([self._unary_states[key][l_id].p_or for l_id in l_ids])
        theta_unary = np.array([self._unary_states[key][l_id].delta_theta for l_id in l_ids])
        l_ids = np.array(l_ids).astype(int)
        A = theta_unary[:, l_ids]
        A /= self._init_state.delta_theta[l_ids]
        A *= -1.
        A = A.T
        A += 1
        np.fill_diagonal(A, 1)
        B = np.ones(l_ids.size)
        return por_unary, np.linalg.solve(A, B)
    
    def compute_flows_reco_lines(self, l_ids: List) -> np.ndarray:
        por_unary, betas = self.compute_betas_reco_lines(l_ids)
        res = (1 - betas.sum()) * self._init_state.p_or
        res += np.matmul(betas, por_unary)    
        return res
    
    @classmethod
    def from_pandapower_net(cls, net) -> "State":
        raise NotImplementedError("TODO")

    @classmethod
    def from_pypowsybl_grid(cls, grid) -> "State":
        raise NotImplementedError("TODO")
    