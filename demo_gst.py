"""GST demonstration on l2rpn_case14_sandbox (DC), mirroring getting_started."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
from superposition_theorem import State
from superposition_theorem.core.compute_power_flows import (
    compute_flows_GST_from_unit_act_obs,
    compute_flows_superposition_theorem_from_unit_act_obs)

param = Parameters()
param.ENV_DC = True
param.MAX_LINE_STATUS_CHANGED = 99999
param.MAX_SUB_CHANGED = 99999
param.NO_OVERFLOW_DISCONNECTION = True
env = grid2op.make("l2rpn_case14_sandbox", param=param, backend=LightSimBackend(), test=True)
A = env.action_space
DEC = 4

def report(name, target, computed):
    d = float(np.max(np.abs(target - computed)))
    print(f"  {name:<42s} max|Δ| = {d:.2e} MW   {'OK' if d < 1e-3 else 'XX'}")
    return d

print("="*78)
print("Example A — two line disconnections + an injection change (redispatch)")
print("="*78)
env.set_id(0); env.reset(); obs_start, *_ = env.step(A({}))
id_l1, id_l2 = 3, 12
redisp = [(0, +5.0), (5, -5.0)]                      # balanced injection change dP
obs1, *_ = obs_start.simulate(A({"set_line_status": [(id_l1, -1)]}), 0)   # topo unit states
obs2, *_ = obs_start.simulate(A({"set_line_status": [(id_l2, -1)]}), 0)
obs_inj, *_ = obs_start.simulate(A({"redispatch": redisp}), 0)            # injection unit state
# ground truth: topology + injection together
combined = A({"set_line_status": [(id_l1, -1), (id_l2, -1)], "redispatch": redisp})
obs_target, *_ = obs_start.simulate(combined, 0)

p_gst, alpha, betas = compute_flows_GST_from_unit_act_obs(
    [id_l1, id_l2], obs_start, [obs1, obs2], [obs_inj], return_betas=True)
print(f"  alpha={alpha:.4f}, betas={np.round(betas,4)}")
report("GST (functional API)", obs_target.p_or, p_gst)

# OO API
st = State.from_grid2op_obs(obs_start, line_ids_disc_unary=[id_l1, id_l2])
st.add_injection_states_grid2op(obs_start, [A({"redispatch": redisp})])
p_gst_oo = st.compute_flows_gst(line_ids_disc=[id_l1, id_l2])
report("GST (object-oriented State API)", obs_target.p_or, p_gst_oo)

# baselines
p_est_only = compute_flows_superposition_theorem_from_unit_act_obs(
    [id_l1, id_l2], [], obs_start, obs_target, [obs1, obs2])   # topology only, no injection
report("EST only (ignores injection) — expected wrong", obs_target.p_or, p_est_only)
p_naive = p_est_only + (obs_inj.p_or - obs_start.p_or)         # naive separable
report("naive separable (EST_ref + injection) — wrong", obs_target.p_or, p_naive)

print("\n" + "="*78)
print("Example B — two line reconnections + an injection change (redispatch)")
print("="*78)
env.set_id(0); env.reset()
# start from a state where l1,l2 are disconnected (reference topology for reco)
obs_start, *_ = env.step(A({"set_line_status": [(id_l1, -1), (id_l2, -1)]}))
obs1, *_ = obs_start.simulate(A({"set_line_status": [(id_l1, +1)]}), 0)
obs2, *_ = obs_start.simulate(A({"set_line_status": [(id_l2, +1)]}), 0)
obs_inj, *_ = obs_start.simulate(A({"redispatch": redisp}), 0)
combined = A({"set_line_status": [(id_l1, +1), (id_l2, +1)], "redispatch": redisp})
obs_target, *_ = obs_start.simulate(combined, 0)
p_gst, alpha, betas = compute_flows_GST_from_unit_act_obs(
    [id_l1, id_l2], obs_start, [obs1, obs2], [obs_inj], return_betas=True)
print(f"  alpha={alpha:.4f}, betas={np.round(betas,4)}")
report("GST (functional API)", obs_target.p_or, p_gst)
p_est_only = compute_flows_superposition_theorem_from_unit_act_obs(
    [id_l1, id_l2], [], obs_start, obs_target, [obs1, obs2])
report("EST only (ignores injection) — expected wrong", obs_target.p_or, p_est_only)

print("\n" + "="*78)
print("Example C — two injection changes (n_P=2) + two line disconnections")
print("="*78)
env.set_id(0); env.reset(); obs_start, *_ = env.step(A({}))
dP1 = [(0, +5.0), (5, -5.0)]
dP2 = [(1, +5.0), (5, -5.0)]
obs1, *_ = obs_start.simulate(A({"set_line_status": [(id_l1, -1)]}), 0)
obs2, *_ = obs_start.simulate(A({"set_line_status": [(id_l2, -1)]}), 0)
obs_i1, *_ = obs_start.simulate(A({"redispatch": dP1}), 0)
obs_i2, *_ = obs_start.simulate(A({"redispatch": dP2}), 0)
# combined injection applied verbatim = dP1 + dP2 nodally
combined = A({"set_line_status": [(id_l1, -1), (id_l2, -1)],
              "redispatch": [(0, +5.0), (1, +5.0), (5, -10.0)]})
obs_target, *_ = obs_start.simulate(combined, 0)
p_gst = compute_flows_GST_from_unit_act_obs(
    [id_l1, id_l2], obs_start, [obs1, obs2], [obs_i1, obs_i2])
report("GST with 2 injection levers + 2 disconnections", obs_target.p_or, p_gst)

print("\n" + "="*78)
print("Example D — two node-splitting actions + an injection change")
print("="*78)
env.set_id(0); env.reset(); obs_start, *_ = env.step(A({}))
id_sub1, id_sub2 = 5, 4
act_s1 = {"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}}
act_s2 = {"set_bus": {"substations_id": [(4, (2, 1, 2, 1, 2))]}}
obs1, *_ = obs_start.simulate(A(act_s1), 0)                 # sub unitary states
obs2, *_ = obs_start.simulate(A(act_s2), 0)
obs_inj, *_ = obs_start.simulate(A({"redispatch": redisp}), 0)
combined = A(act_s1) + A(act_s2) + A({"redispatch": redisp})
obs_target, *_ = obs_start.simulate(combined, 0)
# no lines: unit_act_observations are the sub actions, idls_subs in the same order
p_gst, alpha, betas = compute_flows_GST_from_unit_act_obs(
    [], obs_start, [obs1, obs2], [obs_inj], idls_subs=[id_sub1, id_sub2], return_betas=True)
print(f"  alpha={alpha:.4f}, betas={np.round(betas,4)}")
report("GST (2 node splits + injection)", obs_target.p_or, p_gst)
p_est_only = compute_flows_GST_from_unit_act_obs(
    [], obs_start, [obs1, obs2], [], idls_subs=[id_sub1, id_sub2])  # EST (no injection)
report("EST only (ignores injection) — expected wrong", obs_target.p_or, p_est_only)

print("\n" + "="*78)
print("Example E — mixed: line disconnection + node split + injection")
print("="*78)
env.set_id(0); env.reset(); obs_start, *_ = env.step(A({}))
id_l, id_sub = 3, 5
act_line = {"set_line_status": [(id_l, -1)]}
act_sub = {"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}}
obs_l, *_ = obs_start.simulate(A(act_line), 0)     # line action first
obs_s, *_ = obs_start.simulate(A(act_sub), 0)      # sub action last
obs_inj, *_ = obs_start.simulate(A({"redispatch": redisp}), 0)
combined = A(act_line) + A(act_sub) + A({"redispatch": redisp})
obs_target, *_ = obs_start.simulate(combined, 0)
# ordering: idls_lines first, idls_subs last
p_gst, alpha, betas = compute_flows_GST_from_unit_act_obs(
    [id_l], obs_start, [obs_l, obs_s], [obs_inj], idls_subs=[id_sub], return_betas=True)
print(f"  alpha={alpha:.4f}, betas={np.round(betas,4)}")
report("GST (line disco + node split + injection)", obs_target.p_or, p_gst)

print("\nAll examples done.")
