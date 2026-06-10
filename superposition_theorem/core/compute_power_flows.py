# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

import grid2op
import numpy as np
from superposition_theorem.core.compute_beta_coefficients import get_betas_coeff_N_unit_acts_ultimate
from superposition_theorem.core.compute_beta_coefficients import get_betas_coeff_GST


#so pVirtual_l1=por_Lconnected_l1+(DeltaPVirtual_l2*LODF2->1+DeltaPVirtual_l1*LODF3->1)
#por_l2=LODF1->2*pVirtual1=LODF1->2*(por_Lconnected_l1+(DeltaPVirtual_l2*LODF2->1+DeltaPVirtual_l3*LODF3->1))
#por_l3=LODF1->3*pVirtual1=LODF1->3*(por_Lconnected_l1+(DeltaPVirtual_l2*LODF2->1+DeltaPVirtual_l1*LODF3->1))

#(por_l2-LODF3->2*DeltaPVirtual_l3)-DeltaPVirtual_l2=0
#(LODF1->2*(por_Lconnected_l1+(DeltaPVirtual_l2*LODF2->1+DeltaPVirtual_l3*LODF3->1))-LODF3->2*DeltaPVirtual_l3)-DeltaPVirtual_l2=0
#LODF1->2*por_Lconnected_l1+(LODF1->2*LODF2->1-1)*DeltaPVirtual_l2+(LODF1->2*LODF3->1-LODF3->2)*DeltaPVirtual_l3=0

#LODF1->3*por_Lconnected_l1+(LODF1->3*LODF2->1-LODF2->3)*DeltaPVirtual_l2+(LODF1->3*LODF3->1-1)*DeltaPVirtual_l3=0

#a generic version for n-K

#def get_betas_coeff_reconnect(id_l1,id_l2,env):

def compute_flows_superposition_theorem_from_actions(idls_lines,
                                                     idls_subs,
                                                     obs_start,
                                                     unitary_actions,
                                                     check_obs_target=False,
                                                     decimal_digit_precision=4,
                                                     verbose=False):
    """
    Compute the power flows of the combined unitary actions using the superposition theorem of unitary action observations only

    Parameters
    ----------
    idls_lines: `list`:int
        List of grid lines ids corresponding to the elements on which each unitary action intervenesif relevant. Can be empty

    idls_subs: `list`:int
        List of substation ids corresponding to the elements on which each unitary action intervenes if relevant. Can be empty

    obs_start: grid2op.Observation
        base observation to start from

    unitary_actions: `list`:grid2op.Action
        List of considered unitary actions to apply from base observation

    check_obs_target: `Boolean`
        To assert if the superposition theorem computation was accurate

    decimal_digit_precision: expected number of decimal precision for superposition flow computation compared to load flow

    Returns
    -------
    p_or_combined_action: `np.array` dim n_lines
        The active power flows of the target grid state resulting from the combination of all unitary actions

    """

    # 1) first compute unit_act_observations, p_or_lines and delta_theta_lines
    unit_act_observations = [obs_start.simulate(action, time_step=0)[0] for action in unitary_actions]

    n_actions = len(unit_act_observations)
    target_obs = None
    
    if check_obs_target:
        combined_action = unitary_actions[0]
        for i in range(1, n_actions):
            combined_action += unitary_actions[i]
        target_obs, *_ = obs_start.simulate(combined_action, time_step=0)

    p_or_combined_action = compute_flows_superposition_theorem_from_unit_act_obs(idls_lines,
                                                                                 idls_subs,
                                                                                 obs_start,
                                                                                 target_obs,
                                                                                 unit_act_observations)
    
    if check_obs_target:
        if verbose:
            print("check target flows")
            print(target_obs.p_or)
            print(p_or_combined_action)
        assert (np.all((np.round(target_obs.p_or - p_or_combined_action, decimal_digit_precision) == 0.0)))

    return p_or_combined_action


def compute_flows_superposition_theorem_from_unit_act_obs(idls_lines,
                                                          idls_subs,
                                                          obs_start,
                                                          target_obs,
                                                          unit_act_observations):
    """
    Compute the power flows of the combined unitary actions using the superposition theorem of unitary action observations only

    Parameters
    ----------
    idls_lines: `list`:int
        List of grid lines ids corresponding to the elements on which each unitary action intervenesif relevant. Can be empty

    idls_subs: `list`:int
        List of substation ids corresponding to the elements on which each unitary action intervenes if relevant. Can be empty

    obs_start: grid2op.Observation
        base observation to start from

    unit_act_observations: `list`:grid2op.Observation
        List of grid states on which each unitary actions have been applied independently from obs_start

    decimal_digit_precision: 
        expected number of decimal precision for superposition flow computation compared to load flow

    Returns
    -------
    p_or_combined_action: `np.array` dim n_lines
        The active power flows of the target grid state resulting from the combination of all unitary actions

    """
    n_actions = len(unit_act_observations)
    n_line_actions = len(idls_lines)
    n_sub_actions = len(idls_subs)

    if not n_actions == n_line_actions + n_sub_actions:
        print("There are inconsistencies in your idls: should be either a sub or a line element for each indice.")
        raise ()

    # unit_act_observations
    delta_theta_unit_act_lines = np.array(
        [[get_delta_theta_line(obs_connect_idl, id_lj) for id_lj in idls_lines] for obs_connect_idl in
         unit_act_observations])#[0:n_line_actions]])
    p_or_unit_act_lines = np.array([[obs_connect_idl.p_or[id_lj] for id_lj in idls_lines] for obs_connect_idl in
                                    unit_act_observations])#[0:n_line_actions]])

    # obs_start
    delta_theta_obs_start_lines = np.array([get_delta_theta_line(obs_start, id_lj) for id_lj in idls_lines])
    p_or_obs_start_lines = np.array([obs_start.p_or[id_lj] for id_lj in idls_lines])

    # target_obs
    delta_theta_obs_target_lines = None
    p_or_obs_target_lines = None

    ###############
    # 2) expand osb_start and unit_act_observations with virtual flows
    if n_sub_actions != 0:
        delta_theta_unit_act_lines_subs, delta_theta_obs_start_lines_subs, delta_theta_obs_target_lines_subs = expand_por_lines_with_sub_virtual_line_delta_theta(
            idls_subs,
            unit_act_observations, obs_start, target_obs,
            delta_theta_unit_act_lines, delta_theta_obs_start_lines, delta_theta_obs_target_lines)

        p_or_unit_act_lines_subs, p_or_obs_start_lines_subs, p_or_obs_target_lines_subs = expand_por_lines_with_sub_virtual_line_flow(
            idls_subs, unit_act_observations,
            obs_start, target_obs,
            p_or_unit_act_lines, p_or_obs_start_lines, p_or_obs_target_lines)
    else:
        delta_theta_unit_act_lines_subs = delta_theta_unit_act_lines
        delta_theta_obs_start_lines_subs = delta_theta_obs_start_lines
        delta_theta_obs_target_lines_subs = delta_theta_obs_target_lines

        p_or_unit_act_lines_subs = p_or_unit_act_lines
        p_or_obs_start_lines_subs = p_or_obs_start_lines
        p_or_obs_target_lines_subs = p_or_obs_target_lines

    # compute the betas
    idls = idls_lines + idls_subs
    betas = get_betas_coeff_N_unit_acts_ultimate(delta_theta_unit_act_lines_subs,
                                                 delta_theta_obs_start_lines_subs,
                                                 p_or_unit_act_lines_subs,
                                                 p_or_obs_start_lines_subs,
                                                 delta_theta_obs_target_lines_subs,
                                                 p_or_obs_target_lines_subs,
                                                 idls)

    # compute the resulting p_or
    p_or_combined_action = (1 - np.sum(betas)) * obs_start.p_or
    for i in range(n_actions):
        p_or_combined_action += betas[i] * unit_act_observations[i].p_or

    return p_or_combined_action


def _sub_virtual_sensitivities(obs, idls_subs, ind_subs, is_start_reference_sub_topo):
    """Virtual-line sensitivities of each substation action on observation ``obs``.

    For a **node split** (reference topology is meshed) the substation behaves
    like a line disconnection: its sensitivity is the virtual-line flow
    (the net injection at node 1), evaluated with the node-1 element ids
    ``ind_subs`` derived from the split unitary observation. For a **node merge**
    (reference topology is already split) it behaves like a reconnection: its
    sensitivity is the angle difference between the two nodes.

    Returns two arrays (``p_or_v``, ``delta_theta_v``) of length ``len(idls_subs)``,
    matching the convention of :func:`get_betas_coeff_GST` (flow used when the
    reference flow is non-zero, angle otherwise).
    """
    p_or_v = []
    dth_v = []
    for i, sub in enumerate(idls_subs):
        if is_start_reference_sub_topo[i]:
            # split: reference is meshed -> virtual-line flow (like a disconnection)
            ind_load, ind_prod, ind_lor, ind_lex = ind_subs[i]
            p_or_v.append(get_virtual_line_flow(obs, ind_load, ind_prod, ind_lor, ind_lex))
            dth_v.append(0.0)
        else:
            # merge: reference is split -> angle across the two nodes (like a reconnection)
            p_or_v.append(0.0)
            dth_v.append(get_delta_theta_sub_2nodes(obs, sub))
    return np.array(p_or_v, dtype=float), np.array(dth_v, dtype=float)


def compute_flows_GST_from_unit_act_obs(idls_lines,
                                        obs_start,
                                        unit_act_observations,
                                        injection_observations,
                                        idls_subs=(),
                                        return_betas=False,
                                        verbose=False):
    """
    Generalized Superposition Theorem (GST): compute the flows of a target state
    that combines **topological changes** and an **injection change**, using only
    states available at the *reference* topology.

    This is the injection-aware counterpart of
    :func:`compute_flows_superposition_theorem_from_unit_act_obs`. The injection
    unitary states are observations at the *reference* topology where only the
    injection has changed (e.g. a balanced redispatch). Their flow responses
    ``e_k = obs_inj.p_or - obs_start.p_or`` are added in **pure superposition**,
    while the injection enters the topology weights **only through the right-hand
    side** of the (unchanged) EST linear system (see
    :func:`get_betas_coeff_GST`). Formally

        PF(P_tgt, T_tgt) = alpha * obs_start.p_or
                           + sum_i beta^i * unit_act_observations[i].p_or
                           + sum_k (obs_inj_k.p_or - obs_start.p_or)

    with ``alpha = 1 - sum_i beta^i`` and the ``beta^i`` solving the EST matrix
    with right-hand side ``x_l(P_tgt, T_ref) / x_l(P_ref, T_ref)`` on the
    switched assets. Setting ``injection_observations = []`` recovers the EST.

    Parameters
    ----------
    idls_lines : `list`:int
        Grid line ids of the (disconnection / reconnection) topology actions.
    obs_start : grid2op.Observation
        Reference observation ``(P_ref, T_ref)``.
    unit_act_observations : `list`:grid2op.Observation
        One observation per topology unitary action, applied alone to obs_start.
    injection_observations : `list`:grid2op.Observation
        One observation per unitary injection change, applied alone to obs_start
        at the reference topology.
    idls_subs : `list`:int
        Substation ids for node split / merge actions. A substation action is
        handled through its *virtual line* (exactly as in the EST): node split
        behaves like a line disconnection (virtual-flow ratio), node merge like a
        line reconnection (virtual-angle ratio). The injection shifts these
        virtual sensitivities just like real-line ones. As in
        :func:`compute_flows_superposition_theorem_from_unit_act_obs`, the
        ``unit_act_observations`` must be ordered *lines first, then subs*, and
        ``idls = idls_lines + idls_subs``.

    Returns
    -------
    p_or_combined_action : `np.array`, dim n_lines
        Active power flows of the combined (injection + topology) target state.
        If ``return_betas`` is True, also returns ``(alpha, betas)``.
    """
    idls_lines = list(idls_lines)
    idls_subs = list(idls_subs)
    n_line = len(idls_lines)
    n_sub = len(idls_subs)
    n = n_line + n_sub

    # --- reference-injection sensitivities on the switched LINES (EST matrix) ----
    p_or_unit_lines = np.array([[obs.p_or[lj] for lj in idls_lines] for obs in unit_act_observations])
    dth_unit_lines = np.array([[get_delta_theta_line(obs, lj) for lj in idls_lines] for obs in unit_act_observations])
    p_or_start_lines = np.array([obs_start.p_or[lj] for lj in idls_lines], dtype=float)
    dth_start_lines = np.array([get_delta_theta_line(obs_start, lj) for lj in idls_lines], dtype=float)

    if n_sub:
        # expand the matrix data with the substation VIRTUAL LINES (reuse the EST
        # helpers; obs_target=None so only obs_start + unit_act rows are built)
        dth_unit, dth_start, _ = expand_por_lines_with_sub_virtual_line_delta_theta(
            idls_subs, unit_act_observations, obs_start, None,
            dth_unit_lines, dth_start_lines, None)
        p_or_unit, p_or_start, _ = expand_por_lines_with_sub_virtual_line_flow(
            idls_subs, unit_act_observations, obs_start, None,
            p_or_unit_lines, p_or_start_lines, None)
        # node-1 element ids (from the split/merge unitary observations) and the
        # reference-topology flag for each substation, to evaluate virtual
        # sensitivities at obs_start and at the injection observations
        is_ref_sub = [not (2 in obs_start.sub_topology(s)) for s in idls_subs]
        ind_subs = [get_sub_node1_idsflow(obs, sub_id)
                    for obs, sub_id in zip(unit_act_observations[-n_sub:], idls_subs)]
    else:
        p_or_unit, p_or_start = p_or_unit_lines, p_or_start_lines
        dth_unit, dth_start = dth_unit_lines, dth_start_lines

    # --- target-injection sensitivities on the switched assets, in T_ref ----------
    #     standard ST: x(P_tgt, T_ref) = x(P_ref, T_ref) + sum_k [x(P_ref+dP^k) - x(P_ref)]
    p_or_tgt_lines = p_or_start_lines.copy()
    dth_tgt_lines = dth_start_lines.copy()
    for obs_inj in injection_observations:
        p_or_tgt_lines = p_or_tgt_lines + (np.array([obs_inj.p_or[lj] for lj in idls_lines], dtype=float) - p_or_start_lines)
        dth_tgt_lines = dth_tgt_lines + (np.array([get_delta_theta_line(obs_inj, lj) for lj in idls_lines], dtype=float) - dth_start_lines)

    if n_sub:
        p_or_v_start, dth_v_start = _sub_virtual_sensitivities(obs_start, idls_subs, ind_subs, is_ref_sub)
        p_or_v_tgt = p_or_v_start.copy()
        dth_v_tgt = dth_v_start.copy()
        for obs_inj in injection_observations:
            p_or_v_inj, dth_v_inj = _sub_virtual_sensitivities(obs_inj, idls_subs, ind_subs, is_ref_sub)
            p_or_v_tgt = p_or_v_tgt + (p_or_v_inj - p_or_v_start)
            dth_v_tgt = dth_v_tgt + (dth_v_inj - dth_v_start)
        p_or_tgt = np.append(p_or_tgt_lines, p_or_v_tgt)
        dth_tgt = np.append(dth_tgt_lines, dth_v_tgt)
    else:
        p_or_tgt = p_or_tgt_lines
        dth_tgt = dth_tgt_lines

    # --- GST betas: identical EST matrix, injection-shifted right-hand side -------
    idls = idls_lines + idls_subs
    betas = get_betas_coeff_GST(dth_unit, dth_start, p_or_unit, p_or_start,
                                p_or_tgt, dth_tgt, idls, verbose=verbose)

    # --- reconstruction: affine topology part + pure-superposition injection part -
    p_or_combined_action = (1 - np.sum(betas)) * obs_start.p_or
    for i in range(n):
        p_or_combined_action = p_or_combined_action + betas[i] * unit_act_observations[i].p_or
    for obs_inj in injection_observations:
        p_or_combined_action = p_or_combined_action + (obs_inj.p_or - obs_start.p_or)

    if return_betas:
        return p_or_combined_action, (1 - np.sum(betas)), betas
    return p_or_combined_action


def expand_por_lines_with_sub_virtual_line_flow(idls_subs,
                                                unit_act_observations,
                                                obs_start,
                                                obs_target=None,
                                                p_or_unit_act_lines=None,
                                                p_or_obs_start_lines=None,
                                                p_or_obs_target_lines=None):
    """
    Compute the power flows of the virtual lines at substations for each unitary action observation and start observation.
    This is needed and not part of the original load flow state but can be easily computed on a given observation
    This expands each array of p_or_unit_act_lines for instance by a number of n_action_subs virtual power flow

    Parameters
    ----------
    idls_subs: `list`:int
        List of substation ids corresponding to the elements on which each unitary action intervenes if relevant. Can be empty

    obs_start: grid2op.Observation
        base observation to start from

    unit_act_observations: `list`:grid2op.Observation
        List of grid states on which each unitary actions have been applied independently from obs_start

    obs_target: grid2op.Observation
        target grid state in case we want to check accuracy

    p_or_unit_act_lines: `np.array` of dim n_act_lines * n_act_lines
        active power flow in each unit_act_observation  for each line involved in unitary actions

    p_or_obs_start_lines: `np.array` of dim n_act_lines
        active power flow in obs_start  for each line involved in unitary actions

    p_or_obs_target_lines: `np.array` of dim n_act_lines
        active power flow in obs_target  for each line involved in unitary actions

    Returns
    -------
    p_or_unit_act_lines_subs: `np.array` of dim (n_act_lines + n_act_subs) * (n_act_lines + n_act_subs)
        The active power flows of grid states on which each unitary actions have been applied independently from obs_start, including the virtual line flows

    p_or_obs_start_lines_subs: `np.array` of dim (n_act_lines + n_act_subs)
        The active power flows in obs_start, including the virtual line flows for all lines and virtual lines involved in unitary actions

    p_or_obs_target_lines_subs: `np.array` of dim (n_act_lines + n_act_subs)
        The active power flows in obs_target, including the virtual line flows for all lines and virtual lines involved in unitary actions

    """

    #check which substations are already at two nodes, and which are in reference topology at start
    is_start_reference_sub_topo = [not(2 in obs_start.sub_topology(id_sub)) for id_sub in idls_subs]

    if p_or_unit_act_lines is None or len(p_or_unit_act_lines) == 0:
        p_or_unit_act_lines = [[] for i in range(len(unit_act_observations))]

    p_or_unit_act_lines_subs = [[] for i in range(len(unit_act_observations))]

    # a) compute indices of node1 for each sub of interest

    # TO DO: we assume we go from referencee fully meshed topology to a 2 nodes topology
    # should be able to deal with the reverse case, and to deal with 2 nodes to 2 nodes
    ind_subs = []
    for obs, sub_id in zip(unit_act_observations[-len(idls_subs):], idls_subs):
        (ind_load_node1, ind_prod_node1, ind_lor_node1, ind_lex_node1) = get_sub_node1_idsflow(obs, sub_id)
        ind_subs.append((ind_load_node1, ind_prod_node1, ind_lor_node1, ind_lex_node1))

    # b) compute flow of virtual line for each obs and sub
    n_subs_action = len(idls_subs)
    n_lines_action = len(unit_act_observations) - n_subs_action

    for j, obs in enumerate(unit_act_observations):
        p_or_v_lines = []
        for i, sub in enumerate(idls_subs):
            if i + n_lines_action == j:
                if is_start_reference_sub_topo[i]: #a node splitting action at this substation was applied
                    v_flow = 0.
                else: #a node merging action at this substation was applied
                    ind_load, ind_prod, ind_lor, ind_lex = ind_subs[i]
                    v_flow = get_virtual_line_flow(obs, ind_load, ind_prod, ind_lor, ind_lex)
            else:
                if is_start_reference_sub_topo[i]:#an action was applied at a different substation and this substation is still in its reference topology
                    ind_load, ind_prod, ind_lor, ind_lex = ind_subs[i]
                    v_flow = get_virtual_line_flow(obs, ind_load, ind_prod, ind_lor, ind_lex)
                else:#an action was applied at a different substation and this substation is still splitted into two nodes
                    v_flow = 0.

            p_or_v_lines.append(v_flow)
        p_or_unit_act_lines_subs[j] = np.append(p_or_unit_act_lines[j], p_or_v_lines)

    # c) compute for obs start
    if p_or_obs_start_lines is None:
        p_or_obs_start_lines = []

    p_or_v_lines = []

    for i, sub in enumerate(idls_subs):
        ind_load, ind_prod, ind_lor, ind_lex = ind_subs[i]
        if is_start_reference_sub_topo[i]:
            v_flow = get_virtual_line_flow(obs_start, ind_load, ind_prod, ind_lor, ind_lex)
        else:
            v_flow=0.
        p_or_v_lines.append(v_flow)

    p_or_obs_start_lines_subs = np.append(p_or_obs_start_lines, p_or_v_lines)

    # c) compute for obs target if not None
    p_or_obs_target_lines_subs = None
    if obs_target:
        if p_or_obs_target_lines is None:
            p_or_obs_target_lines = []

        p_or_v_lines = []

        for i, sub in enumerate(idls_subs):
            if is_start_reference_sub_topo[i]:
                v_flow=0.
            else:
                ind_load, ind_prod, ind_lor, ind_lex = ind_subs[i]
                v_flow = get_virtual_line_flow(obs_target, ind_load, ind_prod, ind_lor, ind_lex)
                p_or_v_lines.append(v_flow)

        p_or_obs_target_lines_subs = np.append(p_or_obs_target_lines, p_or_v_lines)

    return p_or_unit_act_lines_subs, p_or_obs_start_lines_subs, p_or_obs_target_lines_subs


def expand_por_lines_with_sub_virtual_line_delta_theta(idls_subs,
                                                       unit_act_observations,
                                                       obs_start,
                                                       obs_target=None,
                                                       delta_theta_unit_act_lines=None,
                                                       delta_theta_obs_start_lines=None,
                                                       delta_theta_obs_target_lines=None):

    """
    Compute the delta theta of the virtual lines at substations for each unitary action observation and start observation.
    This is needed and not part of the original load flow state but can be easily computed on a given observation.
    This expands each array of delta_theta_unit_act_lines for instance by a number of n_action_subs delta theta

    Parameters
    ----------
    idls_subs: `list`:int
        List of substation ids corresponding to the elements on which each unitary action intervenes if relevant. Can be empty

    obs_start: grid2op.Observation
        base observation to start from

    unit_act_observations: `list`:grid2op.Observation
        List of grid states on which each unitary actions have been applied independently from obs_start

    obs_target: grid2op.Observation
        target grid state in case we want to check accuracy

    delta_theta_unit_act_lines: `np.array` of dim n_act_lines * n_act_lines
        delta_theta in each unit_act_observation  for each line involved in unitary actions

    delta_theta_obs_start_lines: `np.array` of dim n_act_lines
        delta_theta in obs_start  for each line involved in unitary actions

    delta_theta_obs_target_lines: `np.array` of dim n_act_lines
        delta_theta in obs_target  for each line involved in unitary actions

    Returns
    -------
    delta_theta_unit_act_lines_subs: `np.array` of dim (n_act_lines + n_act_subs) * (n_act_lines + n_act_subs)
        The delta_thetas of grid states on which each unitary actions have been applied independently from obs_start, including the delta theta of virtual lines

    delta_theta_obs_start_lines_subs: `np.array` of dim (n_act_lines + n_act_subs)
        The delta thetas in obs_start, including the delta thetas for all lines and virtual lines involved in unitary actions

    delta_theta_obs_target_lines_subs: `np.array` of dim (n_act_lines + n_act_subs)
        The delta thetas in obs_target, including the delta thetas for all lines and virtual lines involved in unitary actions

    """

    #check which substations are already at two nodes, and which are in reference topology at start
    is_start_reference_sub_topo = [not(2 in obs_start.sub_topology(id_sub)) for id_sub in idls_subs]

    if delta_theta_unit_act_lines is None or len(delta_theta_unit_act_lines) == 0:
        delta_theta_unit_act_lines = [[] for i in range(len(unit_act_observations))]

    delta_theta_unit_act_lines_subs = [[] for i in range(len(unit_act_observations))]#delta_theta_unit_act_lines

    # TO DO: we assume we go from reference fully meshed topology to a 2 nodes topology
    # should be able to deal with the reverse case, and to deal with 2 nodes to 2 nodes

    n_subs_action = len(idls_subs)
    n_lines_action = len(unit_act_observations) - n_subs_action
    # compute delta theta of virtual line for each obs and sub. Only non null when the split node action is done at sub
    for j, obs in enumerate(unit_act_observations):
        delta_theta_v_lines = []
        for i, sub in enumerate(idls_subs):
            if i + n_lines_action == j:  # this is when the unitary action is done at this sub
                if is_start_reference_sub_topo[i]:
                    delta_theta = get_delta_theta_sub_2nodes(obs,sub)#a node splitting action at this substation was applied
                else:
                    delta_theta = 0. #a node merging action at this substation was applied
            else:
                if is_start_reference_sub_topo[i]:
                    delta_theta = 0 #an action was applied at a different substation and this substation is still in its reference topology
                else:
                    delta_theta = get_delta_theta_sub_2nodes(obs, sub) #an action was applied at a different substation and this substation is still splitted into two nodes
            delta_theta_v_lines.append(delta_theta)

        delta_theta_unit_act_lines_subs[j] = np.append(delta_theta_unit_act_lines[j],delta_theta_v_lines)

    # obs_start
    delta_theta_start=np.array([0. if is_start_reference_sub_topo[i] else get_delta_theta_sub_2nodes(obs_start,sub) for i,sub in enumerate(idls_subs) ])
    if delta_theta_obs_start_lines is None:
        delta_theta_obs_start_lines_subs = delta_theta_start
    else:
        delta_theta_obs_start_lines_subs = np.append(delta_theta_obs_start_lines,
                                                     delta_theta_start)  # assuming fully meshed initial topology at those subs

    # obs_target
    delta_theta_obs_target_lines_subs = None

    if obs_target:
        if delta_theta_obs_target_lines is None:
            delta_theta_obs_target_lines = []

        delta_theta_target = np.array(
            [get_delta_theta_sub_2nodes(obs_target, sub) if is_start_reference_sub_topo[i] else 0.  for i, sub in
             enumerate(idls_subs)])

        delta_theta_obs_target_lines_subs = np.append(delta_theta_obs_target_lines, delta_theta_target)

    return delta_theta_unit_act_lines_subs, delta_theta_obs_start_lines_subs, delta_theta_obs_target_lines_subs


def get_sub_node1_idsflow(obs, sub_id):
    # flow_mat, (ind_load, ind_prod, stor, ind_lor, ind_lex) = obs.flow_bus_matrix()

    ind_prod, prod_conn = obs._get_bus_id(
        obs.gen_pos_topo_vect, obs.gen_to_subid
    )
    ind_load, load_conn = obs._get_bus_id(
        obs.load_pos_topo_vect, obs.load_to_subid
    )
    ind_stor, stor_conn = obs._get_bus_id(
        obs.storage_pos_topo_vect, obs.storage_to_subid
    )
    ind_lor, lor_conn = obs._get_bus_id(
        obs.line_or_pos_topo_vect, obs.line_or_to_subid
    )
    ind_lex, lex_conn = obs._get_bus_id(
        obs.line_ex_pos_topo_vect, obs.line_ex_to_subid
    )

    ind_lor_node1 = [i for i in range(obs.n_line) if ind_lor[i] == sub_id]
    ind_lex_node1 = [i for i in range(obs.n_line) if ind_lex[i] == sub_id]
    ind_load_node1 = [i for i in range(obs.n_load) if ind_load[i] == sub_id]
    ind_prod_node1 = [i for i in range(obs.n_gen) if ind_prod[i] == sub_id]

    return (ind_load_node1, ind_prod_node1, ind_lor_node1, ind_lex_node1)


def get_theta_node(obs, sub_id, bus):
    obj_to_sub = obs.get_obj_connect_to(substation_id=sub_id)

    lines_or_to_sub_bus = [i for i in obj_to_sub['lines_or_id'] if obs.line_or_bus[i] == bus]
    lines_ex_to_sub_bus = [i for i in obj_to_sub['lines_ex_id'] if obs.line_ex_bus[i] == bus]

    thetas_node = np.append(obs.theta_or[lines_or_to_sub_bus], obs.theta_ex[lines_ex_to_sub_bus])
    thetas_node = thetas_node[thetas_node != 0]

    theta_node = 0.
    if len(thetas_node) != 0:
        theta_node = np.median(thetas_node)

    return theta_node


def get_delta_theta_sub_2nodes(obs, sub_id):
    theta_node_bus1 = get_theta_node(obs, sub_id=sub_id, bus=1)
    theta_node_bus2 = get_theta_node(obs, sub_id=sub_id, bus=2)
    delta_theta = theta_node_bus2 - theta_node_bus1

    return delta_theta


def get_delta_theta_line(obs, id_line):
    sub_l_ex = obs.line_ex_to_subid[id_line]
    sub_l_or = obs.line_or_to_subid[id_line]
    bus_ex = obs.line_ex_bus[id_line]
    bus_or = obs.line_or_bus[id_line]

    if bus_ex == -1:
        bus_ex = 1  # the bus on which the line will get reconnected
    if bus_or == -1:
        bus_or = 1  # the bus on which the line will get reconnected

    theta_or_l = get_theta_node(obs, sub_l_or, bus_or)
    theta_ex_l = get_theta_node(obs, sub_l_ex, bus_ex)

    delta_theta_l = theta_or_l - theta_ex_l

    return delta_theta_l  # /360*2*3.14159 #use PI number with enough significant digits!!


def get_virtual_line_flow(obs, ind_load, ind_prod, ind_lor, ind_lex):
    injectionsNode1 = np.array([-obs.p_or[i] for i in ind_lor]).sum()
    injectionsNode1 += np.array([obs.p_or[i] for i in ind_lex]).sum()
    injectionsNode1 += np.array([-obs.load_p[i] for i in ind_load]).sum()
    injectionsNode1 += np.array([obs.gen_p[i] for i in ind_prod]).sum()
    return injectionsNode1
