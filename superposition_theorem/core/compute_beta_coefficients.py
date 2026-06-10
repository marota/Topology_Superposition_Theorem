# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

import numpy as np
from typing import List, Tuple

def get_betas_coeff_N_unit_acts_ultimate(delta_theta_connect_idls : List[Tuple[int, int]],
                                         delta_theta_obs_start,
                                         p_or_connect_idls,
                                         p_or_obs_start,
                                         delta_theta_obs_target=None,
                                         p_or_obs_target=None,
                                         idls=None,
                                         verbose=False):

    """
        Compute the coefficients to apply the superposition theorem with each unitary action of any kind,
        using p_or for disconnect or split actions, or delta_thetas for reconnect or merge action

        Parameters
        ----------
        idls: :class:`list`, dtype:pair(int,int)
            List of grid lines or "virtual" lines corresponding to the elements on which each unitary action intervenes.
            The pair is the ids of each extremity node
        delta_theta_connect_idls: :class:`numpy.ndarray`, dtype:float64
            array of dim n_unitary_actions*n_unitary_actions. For each observation of unitary action intervention,
            it records the delta theta elemnts of every grid lines or "virtual lines" on which any considered unitary action could intervene
        delta_theta_obs_start: :class:`numpy.ndarray`, dtype:float64
             array of dim n_unitary_actions. For the base observation without any action,
             it records the delta theta elements of every grid lines or "virtual lines" on which any considered unitary action could intervene
        p_or_connect_idls: :class:`numpy.ndarray`, dtype:float64
            array of dim n_unitary_actions*n_unitary_actions. For each observation of unitary action intervention,
            it records the p_or elements of every grid lines or "virtual lines" on which any considered unitary action could intervene
        p_or_obs_start: :class:`numpy.ndarray`, dtype:float64
            array of dim n_unitary_actions. For the base observation without any action,
            it records the p_or elements of every grid lines or "virtual lines" on which any considered unitary action could intervene
        delta_theta_obs_target: :class:`numpy.ndarray`, dtype:float64
            array of dim n_unitary_actions. If computed for live assert of proper computation, for the target observation with all combined actions,
            it records the delta theta elements of every grid lines or "virtual lines" on which any considered unitary action could intervene
        p_or_obs_target: :class:`numpy.ndarray`, dtype:float64`
            array of dim n_unitary_actions. If computed for live assert of proper computation, for the target observation with all combined actions,
            it records the p_or elements of every grid lines or "virtual lines" on which any considered unitary action could intervene
        verbose: ``bool``
            Whether to print some informations or not
            
        Returns
        -------
        betas: :class:`numpy.ndarray`, dtype:float64
            the superposition theorem coefficient value to assign to each observation of unitary action intervention

        """

    if (p_or_obs_target is not None) and (delta_theta_obs_target is not None):
        expected_betas =[p_or_obs_target[i ] /p_or_connect_idls[i][i] if
                    ((p_or_connect_idls[i][i] != 0) and (p_or_obs_target[i])) else
                          delta_theta_obs_target[i] / delta_theta_connect_idls[i][i] for i, id_l in enumerate(idls)]
        if verbose:
            print("expected_betas :" + str(expected_betas))

    n_lines_connect = len(idls)

    a = np.array([[1 - p_or_connect_idls[i][j] / p_or_obs_start[j] if (
                (p_or_connect_idls[i][j] != 0) and (p_or_obs_start[j])) else
                   1 - delta_theta_connect_idls[i][j] / delta_theta_obs_start[j] \
                   for i in range(len(idls))] for j in range(len(idls))])
    np.fill_diagonal(a, 1)
    b = np.ones(n_lines_connect)
    if verbose:
        print(a)
        print(b)

    betas = np.linalg.solve(a, b)
    if verbose:
        print("computed betas: " + str(betas))
    return betas


def get_betas_coeff_GST(delta_theta_connect_idls,
                        delta_theta_obs_start,
                        p_or_connect_idls,
                        p_or_obs_start,
                        p_or_obs_target,
                        delta_theta_obs_target,
                        idls,
                        verbose=False):
    """
    Generalized Superposition Theorem (GST) coefficients for combined
    *injection + topology* changes.

    The system **matrix is identical** to the one of
    :func:`get_betas_coeff_N_unit_acts_ultimate` (the Extended Superposition
    Theorem matrix, evaluated at the *reference* injection). The injection
    change enters **only the right-hand side**: instead of ``b = 1``, each entry
    is the ratio of the target-injection sensitivity to the reference one on the
    corresponding switched asset, evaluated *in the reference topology*

        b_j = x_j(P_tgt, T_ref) / x_j(P_ref, T_ref)

    with ``x = p_or`` for disconnection / split assets (non-zero reference flow)
    and ``x = delta_theta`` for reconnection / merge assets (zero reference
    flow). This is the linear system of the companion paper's Theorem 1: the
    matrix is injection-independent (so it can be factored once and reused across
    injection scenarios) and ``b = 1`` recovers the EST exactly.

    The target-injection sensitivities are obtained by the standard
    superposition theorem at the reference topology, i.e.
    ``x_j(P_tgt, T_ref) = x_j(P_ref, T_ref) + sum_k x_j(dP^k, T_ref)``.

    Parameters
    ----------
    delta_theta_connect_idls, delta_theta_obs_start, p_or_connect_idls, p_or_obs_start
        Same arrays as :func:`get_betas_coeff_N_unit_acts_ultimate`; they build
        the (reference-injection) system matrix.
    p_or_obs_target, delta_theta_obs_target : :class:`numpy.ndarray`
        Per-asset flow / angle at the **target injection** in the reference
        topology, on each switched asset (dim ``n_unitary_actions``).
    idls : list
        Switched assets (grid lines or virtual lines).

    Returns
    -------
    betas : :class:`numpy.ndarray`
        The GST topology coefficients ``beta^i`` (the reference state then takes
        ``alpha = 1 - sum_i beta^i``).
    """
    n = len(idls)
    a = np.array([[1 - p_or_connect_idls[i][j] / p_or_obs_start[j] if (
                (p_or_connect_idls[i][j] != 0) and (p_or_obs_start[j])) else
                   1 - delta_theta_connect_idls[i][j] / delta_theta_obs_start[j]
                   for i in range(n)] for j in range(n)])
    np.fill_diagonal(a, 1)
    # injection-shifted right-hand side: flow ratio on disconnected/split assets,
    # angle ratio on reconnected/merged assets (those have zero reference flow)
    b = np.array([p_or_obs_target[j] / p_or_obs_start[j] if (
                (p_or_obs_start[j] is not None) and (p_or_obs_start[j] != 0)) else
                  delta_theta_obs_target[j] / delta_theta_obs_start[j]
                  for j in range(n)])
    if verbose:
        print("GST system matrix a (reference injection):\n" + str(a))
        print("GST right-hand side b (target/reference flow ratio): " + str(b))
    betas = np.linalg.solve(a, b)
    if verbose:
        print("computed GST betas: " + str(betas))
    return betas


# old functions using LODF matrix, still useful for demonstrations
def get_betas_coeff_N2_reconnect(id_l1, id_l2, delta_theta_line_start, delta_theta_line_unit_acts, env, verbose=False):
    init_obs, *_ = env.simulate(env.action_space())
    obs_disconnected_l1_l2, *_ = env.simulate(env.action_space({"set_line_status": [(id_l1, -1), (id_l2, -1)]}))
    obs_disconnect_l1, *_ = env.simulate(env.action_space({"set_line_status": [(id_l1, -1)]}))
    obs_disconnect_l2, *_ = env.simulate(env.action_space({"set_line_status": [(id_l2, -1)]}))

    expected_betas = [init_obs.p_or[id_l1] / obs_disconnect_l2.p_or[id_l1],
                      init_obs.p_or[id_l2] / obs_disconnect_l1.p_or[id_l2]]
    if verbose:
        print("expected_betas :" + str(expected_betas))

    a = [[1, 1 - delta_theta_line_unit_acts[1][id_l1] / delta_theta_line_start[id_l1]],
         [1 - delta_theta_line_unit_acts[0][id_l2] / delta_theta_line_start[id_l2], 1]
         ]
    b = [1, 1]

    if verbose:
        print(a)
        print(b)

    betas = np.linalg.solve(a, b)
    if verbose:
        print("computed betas: " + str(betas))
    return betas


def get_DeltaVirtual_Flows_NK(il_connect, p_il_connect, A, ilds, verbose=False):
    a = []
    for idl in ilds:
        a_row = np.array([A[il_connect][idl] * A[idlj][il_connect] + A[idlj][idl] for idlj in ilds])
        a.append(a_row)

    b = np.array([-p_il_connect * A[il_connect][idl] for idl in ilds])
    pls_virtual = np.linalg.solve(a, b)
    if verbose:
        print(pls_virtual)

    por_virtual_il_connect = p_il_connect
    for i in range(len(ilds)):
        por_virtual_il_connect += A[ilds[i]][il_connect] * pls_virtual[i]

    return por_virtual_il_connect


def get_Flows_NPlusK(p_init, A, ilds, p_ilds_connect, verbose=False):
    nl_connect = len(p_ilds_connect)
    por_virtual_ilds = []
    for idx, il_connect in enumerate(ilds):
        p_il_connect = p_ilds_connect[idx]
        ilds_il_connect = [idl for idl in ilds if idl != il_connect]
        por_virtual_ilds.append(get_DeltaVirtual_Flows_NK(il_connect, p_il_connect, A, ilds_il_connect))

    if verbose:
        print(por_virtual_ilds)

    por_connected = p_init
    for i in range(len(ilds)):
        por_connected -= A[ilds[i]] * por_virtual_ilds[i]

    return por_connected


def get_Virtual_Flows_N2(por_init, A, idl1, idl2, verbose=False):
    a = np.array([[A[idl1][idl1], A[idl2][idl1]], [A[idl1][idl2], A[idl2][idl2]]])
    b = np.array([-por_init[idl1], -por_init[idl2]])
    [pl1_virtual, pl2_virtual] = np.linalg.solve(a, b)
    if verbose:
        print(pl1_virtual)
        print(pl2_virtual)

    por_virtual = por_init + A[idl1] * pl1_virtual + A[idl2] * pl2_virtual

    return por_virtual


# a generic version for n-K
def get_Virtual_Flows_NK(por_init, A, ilds, verbose=False):
    a = []
    for idl in ilds:
        a_row = np.array([A[idlj][idl] for idlj in ilds])
        a.append(a_row)

    b = np.array([-por_init[idl] for idl in ilds])
    pls_virtual = np.linalg.solve(a, b)
    if verbose:
        print(pls_virtual)

    por_virtual = por_init
    for i in range(len(ilds)):
        por_virtual += A[ilds[i]] * pls_virtual[i]

    return por_virtual
