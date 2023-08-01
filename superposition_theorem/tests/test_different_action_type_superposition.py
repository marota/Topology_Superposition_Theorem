import warnings
import numpy as np
import unittest

import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend


from superposition_theorem.core.compute_power_flows import get_virtual_line_flow, get_sub_node1_idsflow, compute_flows_superposition_theorem_from_actions, get_delta_theta_sub_2nodes, get_delta_theta_line

class TestDiverseActionCombinationSup(unittest.TestCase):

    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")


            params = Parameters()
            params.NO_OVERFLOW_DISCONNECTION = True
            params.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
            params.ENV_DC = True
            params.MAX_LINE_STATUS_CHANGED = 99999
            params.MAX_SUB_CHANGED = 99999

            self.env = grid2op.make(env_name,backend=LightSimBackend(),param=params)

            self.env.set_max_iter(20)
        self.chronic_id = 0
        self.max_iter = 10
        self.decimal_accuracy = 4 #until how many decimals we check the perfect accuracy

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_line_disconection_line_reconnection_combination_by_hand(self):
        """Testing the combination of one line reconnection and one line disconnection.
        Computing by hand the superposition of their unitary action state to reach the combined action state"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3  # 1#2#3
        id_l2 = 7  # 2#4#7

        #we reconnect line 1 and disconnect line 2
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {"set_line_status": [(id_l2, -1)]},  # sub4
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {"set_line_status": [(id_l2, +1)]}]

        combined_opposite_action = self.env.action_space(opposite_action_list[0])+self.env.action_space(opposite_action_list[1])
        obs_start, *_  = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # computing intermediate states in which we applied one unitary action

        obs1 = obs_start.simulate(unitary_actions[0], time_step=0)[0]
        obs2 = obs_start.simulate(unitary_actions[1], time_step=0)[0]

        # solving the linear system by "hand" and computing the superposition
        a = np.array([[1, 1 - get_delta_theta_line(obs2, id_l1) / get_delta_theta_line(obs_start, id_l1)],
                      [1 -  obs1.p_or[id_l2] / obs_start.p_or[id_l2], 1]])

        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_target_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or

        assert (np.all((np.round(obs_target.p_or - p_target_computed,self.decimal_accuracy ) == 0.0)))

    def test_line_disconection_line_reconnection_combination_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
        in the case of a combination of one line disconnection and one line reconnection"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3  # 1#2#3
        id_l2 = 7  # 2#4#7

        idls_lines=[id_l1,id_l2]
        idls_subs=[]

        #we reconnect line 1 and disconnect line 2
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {"set_line_status": [(id_l2, -1)]},  # sub4
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {"set_line_status": [(id_l2, +1)]}]

        combined_opposite_action = self.env.action_space(opposite_action_list[0])+self.env.action_space(opposite_action_list[1])
        obs_start, *_  = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions, check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_node_merging_splitting_combination_by_hand(self):
        """Testing the combination of one node merging and one node splitting actions at different substations.
        Computing by hand the superposition of their unitary action state
        to reach the combined action state"""

        self.env.set_id(self.chronic_id)
        self.env.reset()

        unitary_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}},  # merging
                               {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},  # splitting
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},
                                {'set_bus': {'substations_id': [(4, (1, 1, 1, 1, 1))]}},
                                ]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        id_sub1 = 5  # 3#2#3
        id_sub2 = 4  # 7#4#7

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to
        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # computing intermediate states in which we applied one unitary action

        obs1 = obs_start.simulate(unitary_actions[0], time_step=0)[0]
        obs2 = obs_start.simulate(unitary_actions[1], time_step=0)[0]

        # computing delta theetas at a substation between nodes split

        delta_theta_sub1_obs2 = get_delta_theta_sub_2nodes(obs2, id_sub1)
        delta_theta_sub1_obs_start = get_delta_theta_sub_2nodes(obs_start, id_sub1)

        assert (delta_theta_sub1_obs2 != 0)
        assert (delta_theta_sub1_obs_start != 0)

        # virtual line flows: this is not directly computed by the load flow solver.
        # you need to recover the ids of the element of one of the that will appear after splitting
        # and compute the imbalance of flows for these elements before the substation gets split:
        # this gives the equivalent of virtual line flows between the two nodes that will appear
        (ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2, ind_lex_node1_sub2) = get_sub_node1_idsflow(obs2,
                                                                                                                   id_sub2)
        p_start_sub2 = get_virtual_line_flow(obs_start, ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2,
                                             ind_lex_node1_sub2)

        p_obs1_sub2 = get_virtual_line_flow(obs1, ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2,
                                            ind_lex_node1_sub2)

        # solving the linear system by "hand" and computing the superposition
        a = np.array([[1, 1 - delta_theta_sub1_obs2 / delta_theta_sub1_obs_start],
                      [1 - p_obs1_sub2 / p_start_sub2, 1]])

        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_target_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_node_merging_splitting_combination_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
        in the case of a combination of one node merging and one node splitting"""

        self.env.set_id(self.chronic_id)
        self.env.reset()

        unitary_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}},  # merging
                               {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},  # splitting
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},
                                {'set_bus': {'substations_id': [(4, (1, 1, 1, 1, 1))]}},
                                ]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        id_sub1 = 5  # 3#2#3
        id_sub2 = 4  # 7#4#7
        idls_lines = []
        idls_subs = [id_sub1, id_sub2]

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions,
                                                                check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_line_disconection_node_splitting_combination_by_hand(self):
        """Testing the combination of one node splitting and one line disconnection actions.
        Computing by hand the superposition of their unitary action state to reach the combined action state"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub1 = 5

        #we reconnect line 1 and disconnect line 2
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub4
                               ]

        obs_start, *_  = self.env.step(self.env.action_space())

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # computing intermediate states in which we applied one unitary action

        obs1, *_  = obs_start.simulate(unitary_actions[0], time_step=0) # line disconnection
        obs2, *_ = obs_start.simulate(unitary_actions[1], time_step=0) # node splitting

        (ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1, ind_lex_node1_sub1) = get_sub_node1_idsflow(obs2,
                                                                                                                   id_sub1)

        # virtual line flows: this is not directly computed by the load flow solver.
        # you need to recover the ids of the element of one of the that will appear after splitting
        # and compute the imbalance of flows for these elements before the substation gets split:
        # this gives the equivalent of virtual line flows between the two nodes that will appear
        p_start_sub1 = get_virtual_line_flow(obs_start, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1,
                                             ind_lex_node1_sub1)

        p_obs1_sub1 = get_virtual_line_flow(obs1, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1,
                                            ind_lex_node1_sub1)

        # solving the linear system by "hand" and computing the superposition
        a = np.array([[1, 1 - obs2.p_or[id_l1] / obs_start.p_or[id_l1]],
                      [1 -  p_obs1_sub1 / p_start_sub1, 1]])

        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_target_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or

        assert (np.all((np.round(obs_target.p_or - p_target_computed,self.decimal_accuracy ) == 0.0)))

    def test_node_splitting_line_disconnection_combination_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
        in the case of a combination of one node splitting and one line disconnection"""

        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub1 = 5
        idls_lines = [id_l1]
        idls_subs = [id_sub1]

        # we reconnect line 1 and disconnect line 2
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub4
                               ]

        obs_start, *_ = self.env.step(self.env.action_space())

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions,
                                                                check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_line_reconnection_node_merging_combination_by_hand(self):
        """Testing the combination of one node splitting and one line disconnection actions.
        Computing by hand the superposition of their unitary action state to reach the combined action state"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        #we reconnect line 1 and disconnect line 2
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}},  # sub4
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}}
                                ]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # computing intermediate states in which we applied one unitary action

        obs1, *_  = obs_start.simulate(unitary_actions[0], time_step=0) # line disconnection
        obs2, *_ = obs_start.simulate(unitary_actions[1], time_step=0) # node splitting

        # computing delta theetas at a substation between nodes split

        delta_theta_sub_obs1 = get_delta_theta_sub_2nodes(obs1, id_sub)
        delta_theta_sub_obs_start = get_delta_theta_sub_2nodes(obs_start, id_sub)

        assert (delta_theta_sub_obs1 != 0)
        assert (delta_theta_sub_obs_start != 0)

        # solving the linear system by "hand" and computing the superposition
        a = np.array([[1, 1 - get_delta_theta_line(obs2, id_l1) / get_delta_theta_line(obs_start, id_l1)],
                      [1 -  delta_theta_sub_obs1 / delta_theta_sub_obs_start, 1]])

        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_target_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or

        assert (np.all((np.round(obs_target.p_or - p_target_computed,self.decimal_accuracy ) == 0.0)))

    def test_line_reconnection_node_merging_combination_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
        in the case of a combination of one node merging and one line reconnection"""

        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        idls_lines = [id_l1]
        idls_subs = [id_sub]

        # we reconnect line 1 and merge substation
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},
                               {'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}},
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}}
                                ]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions,
                                                                check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_line_reconnection_node_splitting_combination_by_hand(self):
        """Testing the combination of one node splitting and one line reconnection actions.
        Computing by hand the superposition of their unitary action state to reach the combined action state"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        #we reconnect line 1 and split substation
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub4
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}}
                                ]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # computing intermediate states in which we applied one unitary action

        obs1, *_  = obs_start.simulate(unitary_actions[0], time_step=0) # line disconnection
        obs2, *_ = obs_start.simulate(unitary_actions[1], time_step=0) # node splitting

        (ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1, ind_lex_node1_sub1) = get_sub_node1_idsflow(obs2,
                                                                                                                   id_sub)

        # virtual line flows: this is not directly computed by the load flow solver.
        # you need to recover the ids of the element of one of the that will appear after splitting
        # and compute the imbalance of flows for these elements before the substation gets split:
        # this gives the equivalent of virtual line flows between the two nodes that will appear
        p_start_sub1 = get_virtual_line_flow(obs_start, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1,
                                             ind_lex_node1_sub1)

        p_obs1_sub1 = get_virtual_line_flow(obs1, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1,
                                            ind_lex_node1_sub1)

        # solving the linear system by "hand" and computing the superposition
        a = np.array([[1, 1 - get_delta_theta_line(obs2, id_l1) / get_delta_theta_line(obs_start, id_l1)],
                      [1 -  p_obs1_sub1 / p_start_sub1, 1]])

        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_target_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or

        assert (np.all((np.round(obs_target.p_or - p_target_computed,self.decimal_accuracy ) == 0.0)))

    def test_node_splitting_line_reconnection_combination_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
        in the case of a combination of one node splitting and one line reconnection"""

        id_l1 = 3
        id_sub = 5

        idls_lines = [id_l1]
        idls_subs = [id_sub]

        #we reconnect line 1 and split substation
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub4
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}}
                                ]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions,
                                                                check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_line_disconnection_node_merging_combination_by_hand(self):
        """Testing the combination of one node merging and one line disconnection actions.
        Computing by hand the superposition of their unitary action state to reach the combined action state"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        #we disconnect line 1 and merge substation
        unitary_action_list = [{"set_line_status": [(id_l1, -1)]},  # sub5
                               {'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}},  # sub4
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, +1)]},
                                {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}}
                                ]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # computing intermediate states in which we applied one unitary action

        obs1, *_  = obs_start.simulate(unitary_actions[0], time_step=0) # line disconnection
        obs2, *_ = obs_start.simulate(unitary_actions[1], time_step=0) # node splitting

        # computing delta theetas at a substation between nodes split

        delta_theta_sub_obs1 = get_delta_theta_sub_2nodes(obs1, id_sub)
        delta_theta_sub_obs_start = get_delta_theta_sub_2nodes(obs_start, id_sub)

        assert (delta_theta_sub_obs1 != 0)
        assert (delta_theta_sub_obs_start != 0)

        # solving the linear system by "hand" and computing the superposition
        a = np.array([[1, 1 - obs2.p_or[id_l1] / obs_start.p_or[id_l1]],
                      [1 -  delta_theta_sub_obs1 / delta_theta_sub_obs_start, 1]])

        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_target_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or

        assert (np.all((np.round(obs_target.p_or - p_target_computed,self.decimal_accuracy ) == 0.0)))

    def test_line_disconnection_node_merging_combination_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
        in the case of a combination of one node merging and one line disconnection"""

        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5
        idls_lines = [id_l1]
        idls_subs = [id_sub]

        #we disconnect line 1 and merge substation
        unitary_action_list = [{"set_line_status": [(id_l1, -1)]},  # sub5
                               {'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}},  # sub4
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, +1)]},
                                {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}}
                                ]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = self.env.action_space({})
        for unit_act in unitary_actions:
            combined_action+=unit_act

        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions,
                                                                check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_all_four_action_type_combination_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
        in the case of a combination of one node merging, one node splitting, one line disconnection, one line reconnection"""

        self.env.set_id(self.chronic_id)
        self.env.reset()

        #be careful with ambiguous actions when combining actions on lines and substations such as
        #Grid2OpException AmbiguousAction InvalidLineStatus InvalidLineStatus('You ask to disconnect a powerline but also to connect it to a certain bus.')

        id_l1 = 3  # 1#2#3
        id_l2 = 2  # 2#4#7
        id_sub1 = 5
        id_sub2 = 4
        idls_lines = [id_l1,id_l2]
        idls_subs = [id_sub1, id_sub2]

        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # reconnection
                               {"set_line_status": [(id_l2, -1)]}, # disconnection
                               {'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}},  # merging
                               {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},  # splitting
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {"set_line_status": [(id_l2, +1)]},
                                {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},
                                {'set_bus': {'substations_id': [(4, (1, 1, 1, 1, 1))]}},
                                ]

        combined_opposite_action=self.env.action_space({})
        for unit_act in opposite_action_list:
            combined_opposite_action+=self.env.action_space(unit_act)
        obs_start, *_ = self.env.step(combined_opposite_action)

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

        combined_action = self.env.action_space({})
        for unit_act in unitary_actions:
            combined_action+=unit_act
        obs_target, *_ = obs_start.simulate(combined_action, time_step=0)

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions,
                                                                check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

