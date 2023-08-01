import warnings
import numpy as np
import unittest

import grid2op
from grid2op.Parameters import Parameters

from superposition_theorem.core.compute_power_flows import get_virtual_line_flow, get_sub_node1_idsflow, compute_flows_superposition_theorem_from_actions

class TestNodeSplittingSup(unittest.TestCase):

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

            self.env = grid2op.make(env_name,param=params)

            self.env.set_max_iter(20)
        self.chronic_id = 0
        self.max_iter = 10
        self.decimal_accuracy = 4 #until how many decimals we check the perfect accuracy

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_2_combined_actions_node_splitting_by_hand(self):
        """Testing the combination of two nodes splitting actions at different substations. Computing by hand the superposition of their unitary action state
        to reach the combined action state"""

        self.env.set_id(self.chronic_id)
        self.env.reset()

        unitary_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub5
                               {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},  # sub4
                               ]

        obs_start, *_ = self.env.step(self.env.action_space({}))

        id_sub1 = 5  # 3#2#3
        id_sub2 = 4  # 7#4#7

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to
        combined_action = unitary_actions[0] + unitary_actions[1]
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # computing intermediate states in which we applied one unitary action

        obs1 = obs_start.simulate(unitary_actions[0], time_step=0)[0]
        obs2 = obs_start.simulate(unitary_actions[1], time_step=0)[0]

        (ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1, ind_lex_node1_sub1) = get_sub_node1_idsflow(obs1,
                                                                                                                   id_sub1)
        (ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2, ind_lex_node1_sub2) = get_sub_node1_idsflow(obs2,
                                                                                                                   id_sub2)

        # virtual line flows: this is not directly computed by the load flow solver.
        # you need to recover the ids of the element of one of the that will appear after splitting
        # and compute the imbalance of flows for these elements before the substation gets split:
        # this gives the equivalent of virtual line flows between the two nodes that will appear
        p_start_sub1 = get_virtual_line_flow(obs_start, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1,
                                             ind_lex_node1_sub1)
        p_start_sub2 = get_virtual_line_flow(obs_start, ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2,
                                             ind_lex_node1_sub2)

        p_obs1_sub2 = get_virtual_line_flow(obs1, ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2,
                                            ind_lex_node1_sub2)

        p_obs2_sub1 = get_virtual_line_flow(obs2, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1,
                                            ind_lex_node1_sub1)

        # solving the linear system by "hand" and computing the superposition
        a = np.array([[1, 1 - p_obs2_sub1 / p_start_sub1],
                      [1 - p_obs1_sub2 / p_start_sub2, 1]])

        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_target_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or

        assert (np.all((np.round(obs_target.p_or - p_target_computed,self.decimal_accuracy ) == 0.0)))

    def test_2_combined_actions_node_splitting_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
        in the case of a combination of two nodes splitting actions at different substations"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        unitary_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub5
                               {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},  # sub4
                               ]

        obs_start, *_ = self.env.step(self.env.action_space({}))

        id_sub1 = 5  # 3#2#3
        id_sub2 = 4  # 7#4#7
        idls_lines = []
        idls_subs = [id_sub1, id_sub2]

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing intermediate states in which we applied one unitary action
        combined_action = self.env.action_space({})
        for unit_act in unitary_actions:
            combined_action+=unit_act
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions, check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_4_combined_actions_node_splitting_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem_from_actions function
             in the case of a combination of 4 node splitting actions at 4 different substations"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        unitary_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub5
                               {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},  # sub4
                               {'set_bus': {'substations_id': [(3, (2, 1, 1, 2, 1, 2))]}},  # sub3
                               {'set_bus': {'substations_id': [(8, (2, 1, 1, 2, 2))]}},  # sub8
                               ]

        obs_start, *_ = self.env.step(self.env.action_space({}))

        id_sub1 = 5  # 3#2#3
        id_sub2 = 4  # 7#4#7
        id_sub3 = 3
        id_sub4 = 8

        idls_lines = []
        idls_subs = [id_sub1, id_sub2,id_sub3, id_sub4]

        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]

        # computing intermediate states in which we applied one unitary action
        combined_action = self.env.action_space({})
        for unit_act in unitary_actions:
            combined_action+=unit_act
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        # running superposition theorem function
        check_obs_target = False
        p_target_computed = compute_flows_superposition_theorem_from_actions(idls_lines, idls_subs, obs_start, unitary_actions, check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))