# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

from typing import List
import numpy as np

from .staticgridprop import StaticGridProp
from .state import CurrentState, CurrentTopo
from .subaction import SubAction     
            
            
class State(object):
    def __init__(self) -> None:
        # grid description
        self._grid : StaticGridProp = StaticGridProp()
        
        # initial state
        self._init_state : CurrentState = None
        self._init_topo : CurrentTopo = None
        
        # states after unary actions
        self._unary_states = {}
    
    def get_emptyact(self) -> SubAction:
        return SubAction(self._grid, self._init_state, self._init_topo)
    
    def _handle_unary_disc(self,
                           line_ids_disc_unary : List[int],
                           obs : "grid2op.Observation.BaseObservation") -> None:
        dict_disco = {}
        if obs._obs_env is None:
            raise RuntimeError("You cannot build a proper Sate if the grid2op observation cannot use obs.simulate")
        
        for l_id in line_ids_disc_unary:
            # simulate the unary disconnection of powerline l_id
            obs_tmp, reward, done, info = obs.simulate(obs._obs_env.action_space({"set_line_status": [(l_id, -1)]}),
                                                        time_step=0)
            if not done:
                state_tmp = CurrentState(self._grid)
                state_tmp.from_grid2op_obs(obs_tmp)
                dict_disco[l_id] = state_tmp
            else:
                raise RuntimeError(f"Impossible to disconnect powerline {l_id}: no feasible solution found.")
        self._unary_states["line_disco"] = dict_disco
        
    def _handle_unary_reco(self,
                           line_ids_reco_unary : List[int],
                           obs : "grid2op.Observation.BaseObservation") -> None:
        dict_reco = {}
        if obs._obs_env is None:
            raise RuntimeError("You cannot build a proper Sate if the grid2op observation cannot use obs.simulate")
        
        for l_id in line_ids_reco_unary:
            # simulate the unary reconnection of powerline l_id
            obs_tmp, reward, done, info = obs.simulate(obs._obs_env.action_space({"set_line_status": [(l_id, +1)]}),
                                                       time_step=0)
            if not done:
                state_tmp = CurrentState(self._grid)
                state_tmp.from_grid2op_obs(obs_tmp)
                dict_reco[l_id] = state_tmp
            else:
                raise RuntimeError(f"Impossible to reconnect powerline {l_id}: no feasible solution found.")
        self._unary_states["line_reco"] = dict_reco
    
    def _handle_unary_subs(self,
                           subs_actions_unary: List[SubAction],
                           obs : "grid2op.Observation.BaseObservation") -> None:
        dict_sub = {}
        if obs._obs_env is None:
            raise RuntimeError("You cannot build a proper Sate if the grid2op observation cannot use obs.simulate")
        for sub_repr in subs_actions_unary:
            g2op_act = sub_repr.to_grid2op(obs._obs_env.action_space)
            sub_hashed = hash(sub_repr)
            obs_tmp, reward, done, info = obs.simulate(g2op_act,
                                                       time_step=0)
            if not done:
                state_tmp = CurrentState(self._grid)
                state_tmp.from_grid2op_obs(obs_tmp)
                dict_sub[sub_hashed] = state_tmp
            else:
                import pdb
                pdb.set_trace()
                raise RuntimeError(f"Impossible modify the substation {sub_repr}: no feasible solution found.")
            
        self._unary_states["sub_modif"] = dict_sub
    
    @classmethod
    def from_grid2op_obs(cls,
                         obs : "grid2op.Observation.BaseObservation",
                         line_ids_disc_unary : List[int] = (),
                         line_ids_reco_unary : List[int]  = (),
                         subs_actions_unary : List[SubAction]  = ()) -> "State":
        res = cls()
        
        # grid description
        res._grid.from_grid2op_obs(obs)
        
        # actual state (flows, etc.)
        res._init_state = CurrentState(res._grid)
        res._init_state.from_grid2op_obs(obs)
        res._init_topo = CurrentTopo(res._grid)
        res._init_topo.from_grid2op_obs(obs)
        
        # unary states
        res._unary_states = {}
        if line_ids_disc_unary:
            res._handle_unary_disc(line_ids_disc_unary, obs)
        if line_ids_reco_unary:
            res._handle_unary_reco(line_ids_reco_unary, obs)
        if subs_actions_unary:
            res._handle_unary_subs(subs_actions_unary, obs)
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
    