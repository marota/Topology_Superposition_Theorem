# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

from typing import List, Literal, Dict, Any
import numpy as np
import warnings
import time
from scipy.linalg import LinAlgError

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
        self._unary_states : Dict[Literal["line_disco", "line_reco", "node_split"], Any]= {}
    
    def get_emptyact(self) -> SubAction:
        return SubAction(self._grid, self._init_state, self._init_topo)
    
    def _handle_unary_disc(self,
                           line_ids_disc_unary : List[int],
                           obs : "grid2op.Observation.BaseObservation",
                           when_error=Literal["raise", "warn"]) -> None:
        if "line_disco" in self._unary_states:
            dict_disco = self._unary_states["line_disco"]
        else:
            dict_disco = {}
        if obs._obs_env is None:
            raise RuntimeError("You cannot build a proper Sate if the grid2op observation cannot use obs.simulate")
        
        for l_id in line_ids_disc_unary:
            if l_id in dict_disco:
                # line already added previously
                continue
            
            # simulate the unary disconnection of powerline l_id
            obs_tmp, reward, done, info = obs.simulate(obs._obs_env.action_space({"set_line_status": [(l_id, -1)]}),
                                                        time_step=0)
            if not done:
                state_tmp = CurrentState(self._grid)
                state_tmp.from_grid2op_obs(obs_tmp)
                dict_disco[l_id] = state_tmp
            else:
                if when_error == "raise":
                    raise RuntimeError(f"Impossible to disconnect powerline {l_id}: no feasible solution found.")
                else:
                    warnings.warn(f"Impossible to disconnect powerline {l_id}: no feasible solution found.")
                    dict_disco[l_id] = None
        self._unary_states["line_disco"] = dict_disco
        
    def _handle_unary_reco(self,
                           line_ids_reco_unary : List[int],
                           obs : "grid2op.Observation.BaseObservation",
                           when_error=Literal["raise", "warn"]) -> None:
        if "line_reco" in self._unary_states:
            dict_reco = self._unary_states["line_reco"]
        else:
            dict_reco = {}
        if obs._obs_env is None:
            raise RuntimeError("You cannot build a proper Sate if the grid2op observation cannot use obs.simulate")
        
        for l_id in line_ids_reco_unary:
            if l_id in dict_reco:
                # line already added previously
                continue
            
            # simulate the unary reconnection of powerline l_id
            obs_tmp, reward, done, info = obs.simulate(obs._obs_env.action_space({"set_line_status": [(l_id, +1)]}),
                                                       time_step=0)
            if not done:
                state_tmp = CurrentState(self._grid)
                state_tmp.from_grid2op_obs(obs_tmp)
                dict_reco[l_id] = state_tmp
            else:
                if when_error == "raise":
                    raise RuntimeError(f"Impossible to reconnect powerline {l_id}: no feasible solution found.")
                else:
                    warnings.warn(f"Impossible to reconnect powerline {l_id}: no feasible solution found.")
                    dict_reco[l_id] = None
        self._unary_states["line_reco"] = dict_reco
    
    def _handle_unary_subs(self,
                           subs_actions_unary: List[SubAction],
                           obs : "grid2op.Observation.BaseObservation",
                           when_error=Literal["raise", "warn"]) -> None:
        if "node_split" in self._unary_states:
            dict_sub = self._unary_states["node_split"]
        else:
            dict_sub = {}
            
        if obs._obs_env is None:
            raise RuntimeError("You cannot build a proper Sate if the grid2op observation cannot use obs.simulate")
        for sub_repr in subs_actions_unary:
            g2op_act = sub_repr.to_grid2op(obs._obs_env.action_space)
            sub_hashed = hash(sub_repr)
            if sub_hashed in dict_sub:
                # this action has already been added previously
                continue
            # TODO for now: only bus splitting, check that in the original observation.
            obs_tmp, reward, done, info = obs.simulate(g2op_act,
                                                       time_step=0)
            if not done:
                virtual_flow = sub_repr.get_virtual_flow()  # pstart_subXXX in the notebook
                state_tmp = CurrentState(self._grid)
                state_tmp.from_grid2op_obs(obs_tmp)
                dict_sub[sub_hashed] = (sub_repr, state_tmp, virtual_flow)
            else:
                if when_error == "raise":
                    raise RuntimeError(f"Impossible modify the substation {sub_repr}: no feasible solution found.")
                else:
                    warnings.warn(f"Impossible modify the substation {sub_repr}: no feasible solution found.")
                    dict_sub[sub_hashed] = (None, None, None)
            
        self._unary_states["node_split"] = dict_sub
    
    @classmethod
    def from_grid2op_obs(cls,
                         obs : "grid2op.Observation.BaseObservation",
                         line_ids_disc_unary : List[int] = (),
                         line_ids_reco_unary : List[int]  = (),
                         subs_actions_unary : List[SubAction]  = (),
                         when_error=Literal["raise", "warn"]) -> "State":
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
            res._handle_unary_disc(line_ids_disc_unary, obs, when_error=when_error)
        if line_ids_reco_unary:
            res._handle_unary_reco(line_ids_reco_unary, obs, when_error=when_error)
        if subs_actions_unary:
            res._handle_unary_subs(subs_actions_unary, obs, when_error=when_error)
        return res

    def add_unary_actions_grid2op(self,
                                  obs : "grid2op.Observation.BaseObservation",
                                  line_ids_disc_unary : List[int] = (),
                                  line_ids_reco_unary : List[int]  = (),
                                  subs_actions_unary : List[SubAction]  = (),
                                  when_error=Literal["raise", "warn"]) -> None:
        if line_ids_disc_unary:
            self._handle_unary_disc(line_ids_disc_unary, obs, when_error=when_error)
        if line_ids_reco_unary:
            self._handle_unary_reco(line_ids_reco_unary, obs, when_error=when_error)
        if subs_actions_unary:
            self._handle_unary_subs(subs_actions_unary, obs, when_error=when_error)
    
    def _fill_A_disco_lines(self, A, l_ids, start_id, key="line_disco"):
        por_unary = np.array([self._unary_states[key][l_id].p_or for l_id in l_ids])
        end_id = start_id + len(l_ids)
        A[start_id:end_id, start_id:end_id] = por_unary[:, l_ids]
        A[start_id:end_id, start_id:end_id] /= self._init_state.p_or[l_ids]
        A[start_id:end_id, start_id:end_id] *= -1.
        A[start_id:end_id, start_id:end_id] = A[start_id:end_id, start_id:end_id].T
        A[start_id:end_id, start_id:end_id] += 1
        return por_unary
        
    def compute_betas_disco_lines(self, l_ids):
        # works only for reconnected powerline, but does not requires knowledge of theta
        l_ids = np.array(l_ids).astype(int)
        A = np.zeros((len(l_ids), len(l_ids)))
        por_unary = self._fill_A_disco_lines(A, l_ids, 0)
        B = np.ones(l_ids.size)
        return por_unary, np.linalg.solve(A, B)
        
    def compute_flows_disco_lines(self, l_ids: List) -> np.ndarray:
        # por_unary, betas = self.compute_betas_disco_lines(l_ids)
        por_unary, betas = self.compute_betas_reco_lines(l_ids, key="line_disco")
        res = (1 - betas.sum()) * self._init_state.p_or
        res += np.matmul(betas, por_unary)    
        return res
    
    def _fill_A_reco_lines(self, A, l_ids, start_id, key="line_reco"):
        d_theta_unary = np.array([self._unary_states[key][l_id].delta_theta for l_id in l_ids])
        end_id = start_id + len(l_ids)
        A[start_id:end_id, start_id:end_id] = d_theta_unary[:, l_ids]
        A[start_id:end_id, start_id:end_id] /= self._init_state.delta_theta[l_ids]
        A[start_id:end_id, start_id:end_id] *= -1.
        A[start_id:end_id, start_id:end_id] = A[start_id:end_id, start_id:end_id].T
        A[start_id:end_id, start_id:end_id] += 1
        
    def compute_betas_reco_lines(self, l_ids, key="line_reco"):
        # works when flow on powerline is 0, but requires theta
        l_ids = np.array(l_ids).astype(int)
        por_unary = np.array([self._unary_states[key][l_id].p_or for l_id in l_ids])
        A = np.zeros((len(l_ids), len(l_ids)))
        self._fill_A_reco_lines(A, l_ids, 0, key=key)
        np.fill_diagonal(A, 1)
        B = np.ones(l_ids.size)
        return por_unary, np.linalg.solve(A, B)
    
    def compute_flows_reco_lines(self, l_ids: List) -> np.ndarray:
        por_unary, betas = self.compute_betas_reco_lines(l_ids)
        res = (1 - betas.sum()) * self._init_state.p_or
        res += np.matmul(betas, por_unary)    
        return res    
    
    def _fill_A_node_split(self, A, li_hash, start_id, key="node_split"):
        for act_id, act_hashed in enumerate(li_hash):
            sub_repr, state_tmp, virtual_flow = self._unary_states[key][act_hashed]
            for act2_id, act2_hashed in enumerate(li_hash):
                if act2_id >= act_id:
                    continue
                sub_repr2, state_tmp2, virtual_flow2 = self._unary_states[key][act2_hashed]
                vflow1 = sub_repr.get_virtual_flow(state_tmp2)
                vflow2 = sub_repr2.get_virtual_flow(state_tmp)
                A[act_id + start_id, act2_id + start_id] = 1. - vflow1 / virtual_flow
                A[act2_id + start_id, act_id + start_id] = 1. - vflow2 / virtual_flow2
        
    def compute_betas_node_split(self, li_sub_act, key="node_split"):
        # works when flow on powerline is 0, but requires theta
        li_hash = [hash(act) for act in li_sub_act]
        A = np.zeros((len(li_sub_act), len(li_sub_act)))
        self._fill_A_node_split(A, li_hash, start_id=0, key=key)
        np.fill_diagonal(A, 1)
        B = np.ones(len(li_sub_act))
        por_unary = np.array([self._unary_states[key][act_hashed][1].p_or for act_hashed in li_hash])
        return por_unary, np.linalg.solve(A, B)
    
    def compute_flows_node_split(self, li_sub_act: List[SubAction]) -> np.ndarray:
        por_unary, betas = self.compute_betas_node_split(li_sub_act)
        res = (1 - betas.sum()) * self._init_state.p_or
        res += np.matmul(betas, por_unary)    
        return res
    
    def compute_flows(self,
                      *,  # force kwargs
                      line_ids_disc : List[int] = (),
                      line_ids_reco : List[int]  = (),
                      subs_actions : List[SubAction]  = ()
                      ) -> np.ndarray:
        """compute the flows when combining action of different type"""
        raise NotImplementedError("TODO")
        n = 0
        if subs_actions:
            li_hash = [hash(act) for act in subs_actions]
            n += len(li_hash)
        if line_ids_disc:
            n += len(line_ids_disc)
        if line_ids_reco:
            n += len(line_ids_reco)
            
        A = np.zeros((n, n))
        por_unary = np.zeros((n, self._grid.line_or_subid.shape[0]))
        start_id = 0
        if subs_actions:
            key = "node_split"
            self._fill_A_node_split(A, li_hash, start_id=start_id, key=key)
            por_unary[:len(li_hash),:] = np.array([self._unary_states[key][act_hashed][1].p_or for act_hashed in li_hash])
            start_id += len(li_hash)
            
        if line_ids_disc:
            l_ids = np.array(line_ids_disc).astype(int)
            key = "line_disco"
            tmp_por = self._fill_A_disco_lines(A, l_ids, start_id, key=key)
            por_unary[start_id:(len(line_ids_disc) + start_id),:] = tmp_por
            start_id += len(l_ids)
        
        if line_ids_reco:
            l_ids = np.array(line_ids_reco).astype(int)
            key = "line_reco"
            self._fill_A_reco_lines(A, l_ids, start_id, key=key)
            por_unary[start_id:(len(line_ids_reco) + start_id),:] = np.array([self._unary_states[key][l_id].p_or for l_id in l_ids])
            start_id += len(l_ids)
            
        np.fill_diagonal(A, 1)
        B = np.ones(n)
        betas = np.linalg.solve(A, B)
        res = (1 - betas.sum()) * self._init_state.p_or
        res += np.matmul(betas, por_unary)    
        return res
    
    def compute_flows_n1(self,
                         *,  # force kwargs
                         subs_actions : List[SubAction]  = (),
                         line_ids : List[int] = (),
                         ) -> np.ndarray:
        
        n_line = self._grid.line_or_subid.shape[0]
        n = len(subs_actions) + 1  # we do first the topo action then 1 line disconnection at a time
        A = np.zeros((n, n))
        por_unary = np.zeros((n, n_line))
        B = np.ones(n)
        res = np.full((len(line_ids), n_line), fill_value=np.NaN)
        total_time = 0.
        
        # first fill the data for the node splitting (once for all contingencies)
        beg = time.perf_counter()
        key = "node_split"
        li_hash = [hash(act) for act in subs_actions]
        self._fill_A_node_split(A, li_hash, 0, key=key)
        np.fill_diagonal(A, 1)
        por_unary[:len(li_hash),:] = np.array([self._unary_states[key][act_hashed][1].p_or for act_hashed in li_hash])
        total_time += time.perf_counter() - beg
        
        # now perform the security analysis using the extend superposition theorem
        nb_cont = 0
        for cont_id, line_id in enumerate(line_ids):
            all_line_disc = self._unary_states["line_disco"]
            if all_line_disc[line_id] is None:
                # graph is not connex
                continue
            beg = time.perf_counter()
            # update the A matrix
            for act_id, act_hashed in enumerate(li_hash):
                sub_repr, state_tmp, virtual_flow = self._unary_states[key][act_hashed]
                vflow_disc = sub_repr.get_virtual_flow(all_line_disc[line_id])
                A[act_id, -1] = 1 - vflow_disc / virtual_flow
                A[-1, act_id] = 1. - state_tmp.p_or[line_id] / self._init_state.p_or[line_id]
            # solve the linear system
            try:
                betas = np.linalg.solve(A, B)
            except LinAlgError:
                # we do not count the diverging powerflows
                continue
            
            nb_cont += 1
            por_unary[-1, :] = all_line_disc[line_id].p_or
            # save the result
            res[cont_id, :] = (1 - betas.sum()) * self._init_state.p_or
            res[cont_id, :] += np.matmul(betas, por_unary)     
            total_time += time.perf_counter() - beg
                  
        return res, total_time, nb_cont  
        
        
    @classmethod
    def from_pandapower_net(cls, net) -> "State":
        raise NotImplementedError("TODO")

    @classmethod
    def from_pypowsybl_grid(cls, grid) -> "State":
        raise NotImplementedError("TODO")
    