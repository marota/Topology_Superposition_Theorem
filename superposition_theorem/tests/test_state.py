import grid2op
import numpy as np
from superposition_theorem import State
import unittest
import itertools


class TestStateGrid2op(unittest.TestCase):
    def setUp(self) -> None:
        self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
        param = self.env.parameters
        param.ENV_DC = True  # force the computation of the powerflow in DC mode
        param.MAX_LINE_STATUS_CHANGED = 99999
        param.MAX_SUB_CHANGED = 99999
        param.NO_OVERFLOW_DISCONNECTION = True
        self.env.change_parameters(param)
        self.env.change_forecast_parameters(param)
        time_series_id = 0
        self.env.set_id(time_series_id)
        self.obs = self.env.reset()
        self.obs_start, *_  = self.env.step(self.env.action_space({}))
        self.tol = 3e-5
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_init_disco(self):
        l_discs = (1, 2, 3, 4, 7)
        state = State.from_grid2op_obs(self.obs_start, l_discs)
        assert len(state._unary_states) == 1
        assert len(state._unary_states["line_disco"]) == 5
        for l_disc in l_discs:
            assert np.abs(state._unary_states["line_disco"][l_disc].p_or[l_disc]).max() == 0.
        assert np.abs(state._init_state.p_or - self.obs_start.p_or).max() == 0.
    
    def test_2disco(self):
        l_discs = (1, 2, 3, 4, 7)
        state = State.from_grid2op_obs(self.obs_start, l_discs)
        for i, j in itertools.product(l_discs, l_discs):
            if j == i:
                continue
            res_p = state.compute_flows_disco_lines((i, j))
            real_p, *_ = self.obs_start.simulate(self.env.action_space({"set_line_status": [(i, -1), (j, -1)]}), time_step=0)
            real_p = real_p.p_or
            assert np.abs(real_p - res_p).max() <= self.tol, f"error for {i, j}: {np.abs(real_p - res_p).max()}"
            
    def test_3disco(self):
        l_discs = (3, 4, 7)
        state = State.from_grid2op_obs(self.obs_start, l_discs)
        res_p = state.compute_flows_disco_lines(l_discs)
        real_p, *_ = self.obs_start.simulate(self.env.action_space({"set_line_status": [(3, -1), (4, -1), (7, -1)]}), time_step=0)
        real_p = real_p.p_or
        assert np.abs(real_p - res_p).max() <= self.tol, f"error for {np.abs(real_p - res_p).max()}"
    
    def _aux_test_reco(self):
        l_reco = (3, 4, 7)
        start_obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(el, -1) for el in l_reco]}))
        state = State.from_grid2op_obs(start_obs, line_ids_reco_unary=l_reco)
        return l_reco, start_obs, state
    
    def test_init_reco(self):
        l_reco, start_obs, state = self._aux_test_reco()
        assert len(state._unary_states) == 1
        assert len(state._unary_states["line_reco"]) == 3
        for l_rec in l_reco:
            assert np.abs(state._unary_states["line_reco"][l_rec].p_or[l_rec]).max() > self.tol
        assert np.abs(state._init_state.p_or - start_obs.p_or).max() == 0.
    
    def test_2reco(self):
        l_reco, start_obs, state = self._aux_test_reco()
        for i, j in itertools.product(l_reco, l_reco):
            if j == i:
                continue
            res_p = state.compute_flows_reco_lines((i, j))
            real_p, *_ = start_obs.simulate(self.env.action_space({"set_line_status": [(i, +1), (j, +1)]}), time_step=0)
            real_p = real_p.p_or
            assert np.abs(real_p - res_p).max() <= self.tol, f"error for {i, j}: {np.abs(real_p - res_p).max()}"
 
    def test_3reco(self):
        l_reco, start_obs, state = self._aux_test_reco()
        assert len(l_reco) == 3 
        res_p = state.compute_flows_reco_lines(l_reco)
        real_p, *_ = start_obs.simulate(self.env.action_space({"set_line_status": [(el, +1) for el in l_reco]}), time_step=0)
        real_p = real_p.p_or
        assert np.abs(real_p - res_p).max() <= self.tol, f"error for {np.abs(real_p - res_p).max()}"   
            
            
if __name__ == "__main__":
    unittest.main()
    
