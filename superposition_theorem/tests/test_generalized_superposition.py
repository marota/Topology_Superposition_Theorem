# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of superposition_theorem

import warnings
import numpy as np
import unittest

import grid2op
from grid2op.Parameters import Parameters

from superposition_theorem import State
from superposition_theorem.core.compute_power_flows import (
    compute_flows_GST_from_unit_act_obs,
    compute_flows_superposition_theorem_from_unit_act_obs,
)


class TestGeneralizedSuperposition(unittest.TestCase):
    """Generalized Superposition Theorem (GST): combined injection + topology changes.

    The injection change (a balanced redispatch) is applied alone to obs_start at
    the reference topology; its flow response is added in pure superposition while
    the injection enters the topology weights only through the right-hand side.
    """

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
            self.env = grid2op.make(env_name, param=params, test=True)
            self.env.set_max_iter(20)
        self.chronic_id = 0
        self.decimal = 4  # same accuracy as the getting-started notebook
        # a balanced redispatch (sums to zero) -> applied verbatim, within ramps
        self.redisp = [(0, +5.0), (5, -5.0)]

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_2_line_disconnection_plus_injection(self):
        """Two line disconnections combined with an injection change."""
        self.env.set_id(self.chronic_id)
        self.env.reset()
        id_l1, id_l2 = 3, 12
        obs_start, *_ = self.env.step(self.env.action_space({}))

        obs1 = obs_start.simulate(self.env.action_space({"set_line_status": [(id_l1, -1)]}), 0)[0]
        obs2 = obs_start.simulate(self.env.action_space({"set_line_status": [(id_l2, -1)]}), 0)[0]
        obs_inj = obs_start.simulate(self.env.action_space({"redispatch": self.redisp}), 0)[0]

        combined = self.env.action_space({"set_line_status": [(id_l1, -1), (id_l2, -1)],
                                          "redispatch": self.redisp})
        obs_target = obs_start.simulate(combined, 0)[0]

        # functional API
        p_gst = compute_flows_GST_from_unit_act_obs([id_l1, id_l2], obs_start,
                                                    [obs1, obs2], [obs_inj])
        assert np.all(np.round(obs_target.p_or - p_gst, self.decimal) == 0.0)

        # object-oriented API
        st = State.from_grid2op_obs(obs_start, line_ids_disc_unary=[id_l1, id_l2])
        st.add_injection_states_grid2op(obs_start, [self.env.action_space({"redispatch": self.redisp})])
        p_gst_oo = st.compute_flows_gst(line_ids_disc=[id_l1, id_l2])
        assert np.all(np.round(obs_target.p_or - p_gst_oo, self.decimal) == 0.0)

    def test_2_line_reconnection_plus_injection(self):
        """Two line reconnections combined with an injection change."""
        self.env.set_id(self.chronic_id)
        self.env.reset()
        id_l1, id_l2 = 3, 12
        # reference topology has the two lines disconnected
        obs_start, *_ = self.env.step(self.env.action_space(
            {"set_line_status": [(id_l1, -1), (id_l2, -1)]}))

        obs1 = obs_start.simulate(self.env.action_space({"set_line_status": [(id_l1, +1)]}), 0)[0]
        obs2 = obs_start.simulate(self.env.action_space({"set_line_status": [(id_l2, +1)]}), 0)[0]
        obs_inj = obs_start.simulate(self.env.action_space({"redispatch": self.redisp}), 0)[0]

        combined = self.env.action_space({"set_line_status": [(id_l1, +1), (id_l2, +1)],
                                          "redispatch": self.redisp})
        obs_target = obs_start.simulate(combined, 0)[0]

        p_gst = compute_flows_GST_from_unit_act_obs([id_l1, id_l2], obs_start,
                                                    [obs1, obs2], [obs_inj])
        assert np.all(np.round(obs_target.p_or - p_gst, self.decimal) == 0.0)

    def test_two_injections_plus_two_disconnections(self):
        """Two injection levers (n_P=2) combined with two line disconnections."""
        self.env.set_id(self.chronic_id)
        self.env.reset()
        id_l1, id_l2 = 3, 12
        obs_start, *_ = self.env.step(self.env.action_space({}))
        dP1 = [(0, +5.0), (5, -5.0)]
        dP2 = [(1, +5.0), (5, -5.0)]

        obs1 = obs_start.simulate(self.env.action_space({"set_line_status": [(id_l1, -1)]}), 0)[0]
        obs2 = obs_start.simulate(self.env.action_space({"set_line_status": [(id_l2, -1)]}), 0)[0]
        obs_i1 = obs_start.simulate(self.env.action_space({"redispatch": dP1}), 0)[0]
        obs_i2 = obs_start.simulate(self.env.action_space({"redispatch": dP2}), 0)[0]

        # combined injection equals dP1 + dP2 nodally (applied verbatim within ramps)
        combined = self.env.action_space({"set_line_status": [(id_l1, -1), (id_l2, -1)],
                                          "redispatch": [(0, +5.0), (1, +5.0), (5, -10.0)]})
        obs_target = obs_start.simulate(combined, 0)[0]

        p_gst = compute_flows_GST_from_unit_act_obs([id_l1, id_l2], obs_start,
                                                    [obs1, obs2], [obs_i1, obs_i2])
        assert np.all(np.round(obs_target.p_or - p_gst, self.decimal) == 0.0)

    def test_gst_recovers_est_without_injection(self):
        """With no injection state, the GST must reduce exactly to the EST."""
        self.env.set_id(self.chronic_id)
        self.env.reset()
        id_l1, id_l2 = 3, 12
        obs_start, *_ = self.env.step(self.env.action_space({}))
        obs1 = obs_start.simulate(self.env.action_space({"set_line_status": [(id_l1, -1)]}), 0)[0]
        obs2 = obs_start.simulate(self.env.action_space({"set_line_status": [(id_l2, -1)]}), 0)[0]

        combined = self.env.action_space({"set_line_status": [(id_l1, -1), (id_l2, -1)]})
        obs_target = obs_start.simulate(combined, 0)[0]

        p_gst = compute_flows_GST_from_unit_act_obs([id_l1, id_l2], obs_start,
                                                    [obs1, obs2], [])  # no injection
        p_est = compute_flows_superposition_theorem_from_unit_act_obs(
            [id_l1, id_l2], [], obs_start, obs_target, [obs1, obs2])
        # with no injection the GST is the EST (agreement at solver precision)
        assert np.all(np.round(p_gst - p_est, self.decimal) == 0.0)
        assert np.all(np.round(obs_target.p_or - p_gst, self.decimal) == 0.0)


    def test_node_split_plus_injection(self):
        """Two node-splitting actions combined with an injection change."""
        self.env.set_id(self.chronic_id)
        self.env.reset()
        id_sub1, id_sub2 = 5, 4
        act_s1 = {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}}
        act_s2 = {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}}
        obs_start, *_ = self.env.step(self.env.action_space({}))

        obs1 = obs_start.simulate(self.env.action_space(act_s1), 0)[0]   # sub actions
        obs2 = obs_start.simulate(self.env.action_space(act_s2), 0)[0]
        obs_inj = obs_start.simulate(self.env.action_space({"redispatch": self.redisp}), 0)[0]

        combined = (self.env.action_space(act_s1) + self.env.action_space(act_s2)
                    + self.env.action_space({"redispatch": self.redisp}))
        obs_target = obs_start.simulate(combined, 0)[0]

        # no lines: unit_act_observations are the sub actions, idls_subs in same order
        p_gst = compute_flows_GST_from_unit_act_obs(
            [], obs_start, [obs1, obs2], [obs_inj], idls_subs=[id_sub1, id_sub2])
        assert np.all(np.round(obs_target.p_or - p_gst, self.decimal) == 0.0)

    def test_node_merge_plus_injection(self):
        """Two node-merging actions combined with an injection change."""
        self.env.set_id(self.chronic_id)
        self.env.reset()
        id_sub1, id_sub2 = 5, 4
        # reference topology: the two substations are split
        split = (self.env.action_space({'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}})
                 + self.env.action_space({'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}}))
        obs_start, *_ = self.env.step(split)
        merge_s1 = {'set_bus': {'substations_id': [(5, (1, 1, 1, 1, 1, 1, 1))]}}
        merge_s2 = {'set_bus': {'substations_id': [(4, (1, 1, 1, 1, 1))]}}

        obs1 = obs_start.simulate(self.env.action_space(merge_s1), 0)[0]
        obs2 = obs_start.simulate(self.env.action_space(merge_s2), 0)[0]
        obs_inj = obs_start.simulate(self.env.action_space({"redispatch": self.redisp}), 0)[0]

        combined = (self.env.action_space(merge_s1) + self.env.action_space(merge_s2)
                    + self.env.action_space({"redispatch": self.redisp}))
        obs_target = obs_start.simulate(combined, 0)[0]

        p_gst = compute_flows_GST_from_unit_act_obs(
            [], obs_start, [obs1, obs2], [obs_inj], idls_subs=[id_sub1, id_sub2])
        assert np.all(np.round(obs_target.p_or - p_gst, self.decimal) == 0.0)

    def test_mixed_line_and_sub_plus_injection(self):
        """A line disconnection AND a node split, combined with an injection change.

        unit_act_observations must be ordered lines-first, subs-last (as in the
        EST functional API), and idls = idls_lines + idls_subs.
        """
        self.env.set_id(self.chronic_id)
        self.env.reset()
        id_l, id_sub = 3, 5
        act_line = {"set_line_status": [(id_l, -1)]}
        act_sub = {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}}
        obs_start, *_ = self.env.step(self.env.action_space({}))

        obs_l = obs_start.simulate(self.env.action_space(act_line), 0)[0]   # line first
        obs_s = obs_start.simulate(self.env.action_space(act_sub), 0)[0]    # sub last
        obs_inj = obs_start.simulate(self.env.action_space({"redispatch": self.redisp}), 0)[0]

        combined = (self.env.action_space(act_line) + self.env.action_space(act_sub)
                    + self.env.action_space({"redispatch": self.redisp}))
        obs_target = obs_start.simulate(combined, 0)[0]

        p_gst = compute_flows_GST_from_unit_act_obs(
            [id_l], obs_start, [obs_l, obs_s], [obs_inj], idls_subs=[id_sub])
        assert np.all(np.round(obs_target.p_or - p_gst, self.decimal) == 0.0)

        # with no injection, the GST equals the EST result (sanity, avoids the
        # EST target-check path); both match the topology-only ground truth
        obs_target_topo = obs_start.simulate(
            self.env.action_space(act_line) + self.env.action_space(act_sub), 0)[0]
        p_est = compute_flows_GST_from_unit_act_obs(
            [id_l], obs_start, [obs_l, obs_s], [], idls_subs=[id_sub])
        assert np.all(np.round(obs_target_topo.p_or - p_est, self.decimal) == 0.0)


if __name__ == "__main__":
    unittest.main()
