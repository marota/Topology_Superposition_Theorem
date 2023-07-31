import warnings
import numpy as np
import unittest

import grid2op
from grid2op.Parameters import Parameters

from superposition_theorem import get_virtual_line_flow, get_sub_node1_idsflow, compute_flows_superposition_theorem, get_delta_theta_line

class TestLineReconnectionSup(unittest.TestCase):

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

    def test_2_combined_actions_line_reconnection_by_hand(self):
        """Testing the combination of two line reconnection actions at different substations.
        Computing by hand the superposition of their unitary action state to reach the combined action state"""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3  # 1#2#3
        id_l2 = 7  # 2#4#7
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {"set_line_status": [(id_l2, +1)]},  # sub4
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {"set_line_status": [(id_l2, -1)]}]

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
                      [1 -  get_delta_theta_line(obs1, id_l2) / get_delta_theta_line(obs_start, id_l2), 1]])

        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_target_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or

        assert (np.all((np.round(obs_target.p_or - p_target_computed,self.decimal_accuracy ) == 0.0)))

    def test_2_combined_actions_line_reconnection_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem function
        in the case of a combination of two line reconnections"""
        id_l1 = 3  # 1#2#3
        id_l2 = 7  # 2#4#7
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
                               {"set_line_status": [(id_l2, +1)]},  # sub4
                               ]

        #need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {"set_line_status": [(id_l2, -1)]}]

        combined_opposite_action = self.env.action_space(opposite_action_list[0]) + self.env.action_space(
            opposite_action_list[1])
        obs_start, *_ = self.env.step(combined_opposite_action)

        idls_lines = [id_l1,id_l2]
        idls_subs = []

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to
        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]
        combined_action = self.env.action_space({})
        for unit_act in unitary_actions:
            combined_action+=unit_act
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        #running superposition theorem function
        check_obs_target = False
        p_target_computed=compute_flows_superposition_theorem(idls_lines, idls_subs, obs_start, unitary_actions, check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))

    def test_4_combined_actions_line_reconnection_sup_theorem(self):
        """Testing the compute_flows_superposition_theorem function
        in the case of a combination of four line reconnections"""
        id_l1 = 1  # 3#2#3
        id_l2 = 2  # 7#4#7
        id_l3 = 7  # 7#4#7
        id_l4 = 4
        unitary_action_list = [{"set_line_status": [(id_l1, +1)]},
                               {"set_line_status": [(id_l2, +1)]},
                               {"set_line_status": [(id_l3, +1)]},
                               {"set_line_status": [(id_l4, +1)]},
                               ]

        # need ooposite list of actions to start with an observation where those lines are disconnected
        opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
                                {"set_line_status": [(id_l2, -1)]},
                                {"set_line_status": [(id_l3, -1)]},
                                {"set_line_status": [(id_l4, -1)]}
                                ]
        combined_opposite_action=self.env.action_space({})
        for unit_act in opposite_action_list:
            combined_opposite_action+=self.env.action_space(unit_act)

        obs_start, *_ = self.env.step(combined_opposite_action)

        idls_lines = [id_l1,id_l2,id_l3,id_l4]
        idls_subs = []

        # computing obs_target load flow with the effect of combined action: this is the ground truth to compare to
        unitary_actions = [self.env.action_space(unitary_act) for unitary_act in unitary_action_list]
        combined_action = self.env.action_space({})
        for unit_act in unitary_actions:
            combined_action+=unit_act
        obs_target = obs_start.simulate(combined_action, time_step=0)[0]

        #running superposition theorem function
        check_obs_target = False
        p_target_computed=compute_flows_superposition_theorem(idls_lines, idls_subs, obs_start, unitary_actions, check_obs_target)

        assert (np.all((np.round(obs_target.p_or - p_target_computed, self.decimal_accuracy) == 0.0)))