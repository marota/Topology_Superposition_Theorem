import grid2op
import numpy as np
from lightsim2grid import LightSimBackend


#so pVirtual_l1=por_Lconnected_l1+(DeltaPVirtual_l2*LODF2->1+DeltaPVirtual_l1*LODF3->1)
#por_l2=LODF1->2*pVirtual1=LODF1->2*(por_Lconnected_l1+(DeltaPVirtual_l2*LODF2->1+DeltaPVirtual_l3*LODF3->1))
#por_l3=LODF1->3*pVirtual1=LODF1->3*(por_Lconnected_l1+(DeltaPVirtual_l2*LODF2->1+DeltaPVirtual_l1*LODF3->1))

#(por_l2-LODF3->2*DeltaPVirtual_l3)-DeltaPVirtual_l2=0
#(LODF1->2*(por_Lconnected_l1+(DeltaPVirtual_l2*LODF2->1+DeltaPVirtual_l3*LODF3->1))-LODF3->2*DeltaPVirtual_l3)-DeltaPVirtual_l2=0
#LODF1->2*por_Lconnected_l1+(LODF1->2*LODF2->1-1)*DeltaPVirtual_l2+(LODF1->2*LODF3->1-LODF3->2)*DeltaPVirtual_l3=0

#LODF1->3*por_Lconnected_l1+(LODF1->3*LODF2->1-LODF2->3)*DeltaPVirtual_l2+(LODF1->3*LODF3->1-1)*DeltaPVirtual_l3=0

#a generic version for n-K

#def get_betas_coeff_reconnect(id_l1,id_l2,env):

def compute_flows_superposition_theorem(idls_lines, idls_subs, obs_start, unitary_actions, check_obs_target=False,
                                        decimal_digit_precision=4):
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
    n_actions = len(unitary_actions)
    n_line_actions = len(idls_lines)
    n_sub_actions = len(idls_subs)

    if not n_actions == n_line_actions + n_sub_actions:
        print("There are inconsistencies in your idls: should be either a sub or a line element for each indice.")
        raise ()

    # 1) first compute unit_act_observations, p_or_lines and delta_theta_lines
    unit_act_observations = [obs_start.simulate(action, time_step=0)[0] for action in unitary_actions]

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
    target_obs = None
    delta_theta_obs_target_lines = None
    p_or_obs_target_lines = None

    if (check_obs_target):
        combined_action = unitary_actions[0]
        for i in range(1, n_actions):
            combined_action += unitary_actions[i]
        target_obs, *_ = obs_start.simulate(combined_action, time_step=0)

        delta_theta_obs_target_lines = np.array([get_delta_theta_line(target_obs, id_lj) for id_lj in idls_lines])
        p_or_obs_target_lines = np.array([target_obs.p_or[id_lj] for id_lj in idls_lines])

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
    betas = get_betas_coeff_N_reconnect_disconnect_ultimate(delta_theta_unit_act_lines_subs,
                                                            delta_theta_obs_start_lines_subs,
                                                            p_or_unit_act_lines_subs, p_or_obs_start_lines_subs,
                                                            delta_theta_obs_target_lines_subs,
                                                            p_or_obs_target_lines_subs, idls)

    # compute the resulting p_or
    p_or_combined_action = (1 - np.sum(betas)) * obs_start.p_or
    for i in range(n_actions):
        p_or_combined_action += betas[i] * unit_act_observations[i].p_or

    # print(p_or_combined_action)
    # print(target_obs.p_or)
    if (check_obs_target):
        print("check target flows")
        print(target_obs.p_or)
        print(p_or_combined_action)
        assert (np.all((np.round(target_obs.p_or - p_or_combined_action, decimal_digit_precision) == 0.0)))

    return p_or_combined_action


def expand_por_lines_with_sub_virtual_line_flow(idls_subs, unit_act_observations, obs_start, obs_target=None,
                                                p_or_unit_act_lines=None, p_or_obs_start_lines=None,
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
    if (obs_target):
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


def expand_por_lines_with_sub_virtual_line_delta_theta(idls_subs, unit_act_observations, obs_start, obs_target=None,
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

    if (obs_target):
        if delta_theta_obs_target_lines is None:
            delta_theta_obs_target_lines = []

        delta_theta_target = np.array(
            [get_delta_theta_sub_2nodes(obs_target, sub) if is_start_reference_sub_topo[i] else 0.  for i, sub in
             enumerate(idls_subs)])

        delta_theta_obs_target_lines_subs = np.append(delta_theta_obs_target_lines, delta_theta_target)

    return delta_theta_unit_act_lines_subs, delta_theta_obs_start_lines_subs, delta_theta_obs_target_lines_subs


def get_betas_coeff_N_reconnect_ultimate(delta_theta_connect_idls,delta_theta_obs_start,p_or_connect_idls,p_or_obs_start,delta_theta_obs_target=None,p_or_obs_target=None,idls=None):

    """
        Compute the coefficients to apply the superposition theorem with each unitary action of kind "reconnect"

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

        Returns
        -------
        betas: :class:`numpy.ndarray`, dtype:float64
            the superposition theorem coefficient value to assign to each observation of unitary action intervention

        """



    if (p_or_obs_target is not None) and (delta_theta_obs_target is not None):
        #expected_betas=[obs_target.p_or[id_l]/obs_connect_idls[i].p_or[id_l] for i,id_l in enumerate(idls)]
        #expected_betas=[get_delta_theta_line(obs_target,id_l)/get_delta_theta_line(obs_connect_idls[i],id_l) for i,id_l in enumerate(idls)]
        expected_betas=[p_or_obs_target[i]/p_or_connect_idls[i][i] if ((p_or_connect_idls[i][i]!=0) and (p_or_obs_target[i])) else
                        delta_theta_obs_target[i]/delta_theta_connect_idls[i][i] for i,id_l in enumerate(idls)]
        print("expected_betas :"+str(expected_betas))


    n_lines_connect=len(idls)
    a=np.zeros((n_lines_connect,n_lines_connect))
    b=np.zeros(n_lines_connect)

    #    a=[[1,1-get_delta_theta_line(obs_disconnect_l1,id_l1)/get_delta_theta_line(obs_disconnected_l1_l2,id_l1)],
    #    [1-get_delta_theta_line(obs_disconnect_l2,id_l2)/get_delta_theta_line(obs_disconnected_l1_l2,id_l2),1]
    #]
    a=np.array([[1-p_or_connect_idls[i][j]/p_or_obs_start[j] if ((p_or_connect_idls[i][j]!=0) and (p_or_obs_start[j])) else
        1-delta_theta_connect_idls[i][j]/delta_theta_obs_start[j]\
        for i in range(len(idls))] for j in range(len(idls))])

    np.fill_diagonal(a, 1)

    b=np.ones(n_lines_connect)


    print(a)
    print(b)

    betas=np.linalg.solve(a,b)

    print("computed betas: "+str(betas))

    return betas
    
def get_betas_coeff_N_reconnect_disconnect_ultimate(delta_theta_connect_idls,delta_theta_obs_start,p_or_connect_idls,p_or_obs_start,delta_theta_obs_target=None,p_or_obs_target=None,idls=None):
        """
        Compute the coefficients to apply the superposition theorem with each considered unitary action of kind either "reconnect" or "disconnect"

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
        
        Returns
        -------
        betas: :class:`numpy.ndarray`, dtype:float64
            the superposition theorem coefficient value to assign to each observation of unitary action intervention 
            
        """
        return get_betas_coeff_N_reconnect_ultimate(delta_theta_connect_idls,delta_theta_obs_start,p_or_connect_idls,p_or_obs_start,delta_theta_obs_target,p_or_obs_target,idls)

def get_sub_node1_idsflow(obs, sub_id):
    # flow_mat, (ind_load, ind_prod, stor, ind_lor, ind_lex)=obs.flow_bus_matrix()

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
    InjectionsNode1 = np.array([-obs.p_or[i] for i in ind_lor]).sum()
    InjectionsNode1 += np.array([obs.p_or[i] for i in ind_lex]).sum()
    InjectionsNode1 += np.array([-obs.load_p[i] for i in ind_load]).sum()
    InjectionsNode1 += np.array([obs.gen_p[i] for i in ind_prod]).sum()
    return InjectionsNode1

def get_DeltaVirtual_Flows_NK(il_connect,p_il_connect,A,ilds):
    
    
    a=[]
    for idl in ilds:
        a_row=np.array([A[il_connect][idl]*A[idlj][il_connect]+A[idlj][idl] for idlj in ilds])
        a.append(a_row)

    b=np.array([-p_il_connect*A[il_connect][idl] for idl in ilds])
    pls_virtual=np.linalg.solve(a,b)
    print(pls_virtual)
    
    por_virtual_il_connect=p_il_connect
    for i in range(len(ilds)):
        por_virtual_il_connect+=A[ilds[i]][il_connect]*pls_virtual[i]
    
    return por_virtual_il_connect


def get_Flows_NPlusK(p_init,A,ilds,p_ilds_connect):
    
    nl_connect=len(p_ilds_connect)
    por_virtual_ilds=[]
    for idx,il_connect in enumerate(ilds):
        p_il_connect=p_ilds_connect[idx]
        ilds_il_connect=[idl for idl in ilds if idl!=il_connect]
        por_virtual_ilds.append(get_DeltaVirtual_Flows_NK(il_connect,p_il_connect,A,ilds_il_connect))
    
    print(por_virtual_ilds)
    
    por_connected=p_init
    for i in range(len(ilds)):
        por_connected-=A[ilds[i]]*por_virtual_ilds[i]
        
    return por_connected

def get_Virtual_Flows_N2(por_init,A,idl1,idl2):
    a=np.array([[A[idl1][idl1],A[idl2][idl1]],[A[idl1][idl2],A[idl2][idl2]]])
    b=np.array([-por_init[idl1],-por_init[idl2]])
    [pl1_virtual,pl2_virtual]=np.linalg.solve(a,b)
    print(pl1_virtual)
    print(pl2_virtual)
    
    por_virtual=por_init+A[idl1]*pl1_virtual+A[idl2]*pl2_virtual
    
    return por_virtual

#a generic version for n-K
def get_Virtual_Flows_NK(por_init,A,ilds):
    a=[]
    for idl in ilds:
        a_row=np.array([A[idlj][idl] for idlj in ilds])
        a.append(a_row)

    b=np.array([-por_init[idl] for idl in ilds])
    pls_virtual=np.linalg.solve(a,b)
    print(pls_virtual)
    
    por_virtual=por_init
    for i in range(len(ilds)):
        por_virtual+=A[ilds[i]]*pls_virtual[i]
    
    return por_virtual

def get_betas_coeff_N2_reconnect(id_l1,id_l2,env):
    init_obs, *_ = env.simulate(env.action_space())
    obs_disconnected_l1_l2, *_ = env.simulate(env.action_space({"set_line_status": [(id_l1, -1),(id_l2, -1)]}))
    obs_disconnect_l1, *_ = env.simulate(env.action_space({"set_line_status": [(id_l1, -1)]}))
    obs_disconnect_l2, *_ = env.simulate(env.action_space({"set_line_status": [(id_l2, -1)]}))
    
    expected_betas=[init_obs.p_or[id_l1]/obs_disconnect_l2.p_or[id_l1],init_obs.p_or[id_l2]/obs_disconnect_l1.p_or[id_l2]]
    print("expected_betas :"+str(expected_betas))
    
    sub_l1_ex=init_obs.line_ex_to_subid[id_l1]
    sub_l1_or=init_obs.line_or_to_subid[id_l1]

    sub_l2_ex=init_obs.line_ex_to_subid[id_l2]
    sub_l2_or=init_obs.line_or_to_subid[id_l2]
    
    a=[[1,1-get_delta_theta_line(obs_disconnect_l1,id_l1)/get_delta_theta_line(obs_disconnected_l1_l2,id_l1)],
        [1-get_delta_theta_line(obs_disconnect_l2,id_l2)/get_delta_theta_line(obs_disconnected_l1_l2,id_l2),1]
    ]
    b=[1,1]

    
    print(a)
    print(b)

    betas=np.linalg.solve(a,b)
    
    print("computed betas: "+str(betas))
    return betas
    
    

def get_betas_coeff_N_reconnect(idls,obs_connect_idls,obs_start,obs_target=None):
    
    if obs_target is not None:
        #expected_betas=[obs_target.p_or[id_l]/obs_connect_idls[i].p_or[id_l] for i,id_l in enumerate(idls)]
        expected_betas=[get_delta_theta_line(obs_target,id_l)/get_delta_theta_line(obs_connect_idls[i],id_l) for i,id_l in enumerate(idls)]
        print("expected_betas :"+str(expected_betas))
    

    n_lines_connect=len(idls)
    a=np.zeros((n_lines_connect,n_lines_connect))
    b=np.zeros(n_lines_connect)
    
    #    a=[[1,1-get_delta_theta_line(obs_disconnect_l1,id_l1)/get_delta_theta_line(obs_disconnected_l1_l2,id_l1)],
    #    [1-get_delta_theta_line(obs_disconnect_l2,id_l2)/get_delta_theta_line(obs_disconnected_l1_l2,id_l2),1]
    #]
    a=np.array([[1-get_delta_theta_line(obs_connect_idls[i],idls[j])/get_delta_theta_line(obs_start,idls[j])\
        for i in range(len(idls))] for j in range(len(idls))])
    
    np.fill_diagonal(a, 1)
       
    b=np.ones(n_lines_connect)
    
        
    print(a)
    print(b)

    betas=np.linalg.solve(a,b)
    
    print("computed betas: "+str(betas))
    return betas
    
#def get_betas_coeff_N_reconnect_disconnect(idls,obs_connect_idls,obs_start,obs_target=None):
#    return get_betas_coeff_N_reconnect(idls,obs_connect_idls,obs_start,obs_target)




