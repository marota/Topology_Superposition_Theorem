import grid2op
import numpy as np

from superposition_theorem import State

env = grid2op.make("l2rpn_case14_sandbox")
param = env.parameters
param.ENV_DC = True  # force the computation of the powerflow in DC mode
param.MAX_LINE_STATUS_CHANGED = 99999
param.MAX_SUB_CHANGED = 99999
param.NO_OVERFLOW_DISCONNECTION = True
env.change_parameters(param)
env.change_forecast_parameters(param)
env.set_id(0)
obs = env.reset()
start_obs, *_  = env.step(env.action_space({}))

id_sub1 = 5  
id_sub2 = 4  
unitary_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub5
                       {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},  # sub4
                       ]

unitary_actions = [env.action_space(unitary_act) for unitary_act in unitary_action_list]

init_state = State.from_grid2op_obs(start_obs)

# first unary action
id_sub1 = 5
act1 = init_state.get_emptyact()
act1.set_subid(id_sub1)
# act1.get_elem_sub()  # to know which elements are connected to this substation
act1.set_bus(lines_id=[(7, 1), (8, 1), (9, 2), (17, 2)],
             loads_id=[(4, 2)],
             gens_id=[(2, 1), (3, 2)])

# second unary action
id_sub2 = 4
act2 = init_state.get_emptyact()
act2.set_subid(id_sub2)
# act2.get_elem_sub()  # to know which elements are connected to this substation
act2.set_bus(lines_id=[(1, 2), (4, 1), (6, 2), (17, 1)],
             loads_id=[(3, 2)])
# state = State.from_grid2op_obs(start_obs, subs_actions_unary=[act1, act2])
# computed_por = state.compute_flows_node_split([act1, act2])
# real_obs, *_ = start_obs.simulate(unitary_actions[0] + unitary_actions[1], time_step=0)
# print(np.abs(computed_por - real_obs.p_or).max())
#############""


state = State.from_grid2op_obs(start_obs,
                               line_ids_disc_unary=(2, 3),
                               subs_actions_unary=[act1, act2])



# p_or_combined = state.compute_flows(line_ids_disc=(2, 3), subs_actions=[act1, act2])

# combined_act = act1.to_grid2op(env.action_space) + act2.to_grid2op(env.action_space)
# combined_act += env.action_space({"set_line_status": [(2, -1), (3, -1)]})
# obs_true, r, done, info = start_obs.simulate(combined_act, time_step=0)
# assert not info["exception"], f'Error while performing the powerflow check: {info["exception"]}'
# por_true = obs_true.p_or
# print(f"Max difference: {np.abs(p_or_combined - por_true).max():.2e} MW")

##########
print("Normal A")
state.compute_flows_disco_lines((2, 3))

print("A I got")
res, total_time, nb_cont  = state.compute_flows_n1(subs_actions=[act1, act2], line_ids=(2, 3))

combined_act = act1.to_grid2op(env.action_space) + act2.to_grid2op(env.action_space)
f_0 , *_ = start_obs.simulate(combined_act + env.action_space({"set_line_status": [(2, -1)]}), time_step=0)
f_1 , *_ = start_obs.simulate(combined_act + env.action_space({"set_line_status": [(3, -1)]}), time_step=0)
print(f"Max difference: {np.abs(res[0] - f_0.p_or).max():.2e} MW")
print(f"Max difference: {np.abs(res[1] - f_1.p_or).max():.2e} MW")
import pdb
pdb.set_trace()
import sys
sys.exit(0)
##############


import grid2op
from superposition_theorem import State

env = grid2op.make("l2rpn_case14_sandbox")
param = env.parameters
param.ENV_DC = True  # force the computation of the powerflow in DC mode
param.MAX_LINE_STATUS_CHANGED = 99999
param.MAX_SUB_CHANGED = 99999
param.NO_OVERFLOW_DISCONNECTION = True
env.change_parameters(param)
env.change_forecast_parameters(param)
time_series_id = 2
env.set_id(time_series_id)
obs_start = env.reset()
# obs_start, *_  = env.step(env.action_space({}))
tol = 3e-5

# env.set_id(time_series_id)
# env.reset()

# id_l1 = 3  # 1#2#3
# id_l2 = 7  # 2#4#7
# unitary_action_list = [{"set_line_status": [(id_l1, +1)]},  # sub5
#                        {"set_line_status": [(id_l2, +1)]},  # sub4
#                        ]

# # need opposite list of actions to start with an observation where those lines are disconnected
# opposite_action_list = [{"set_line_status": [(id_l1, -1)]},
#                         {"set_line_status": [(id_l2, -1)]}]

# combined_opposite_action = env.action_space(opposite_action_list[0]) + env.action_space(opposite_action_list[1])
# obs_start, reward, done, info  = env.step(combined_opposite_action)
# state = State.from_grid2op_obs(obs_start, line_ids_reco_unary=(id_l1, id_l2))
# res_p = state.compute_flows_reco_lines((id_l1, id_l2))
# import pdb
# pdb.set_trace()

env.set_id(time_series_id)
env.reset()

id_sub1 = 5  
id_sub2 = 4  
unitary_action_list = [{'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},  # sub5
                       {'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},  # sub4
                       ]

unitary_actions = [env.action_space(unitary_act) for unitary_act in unitary_action_list]

obs_start, *_  = env.step(env.action_space({}))
init_state = State.from_grid2op_obs(obs_start)

# first unary action
act1 = init_state.get_emptyact()
act1.set_subid(id_sub1)
act1.get_elem_sub()
act1.set_bus(lines_id=[(7, 1), (8, 1), (9, 2), (17, 2)],
             loads_id=[(4, 2)],
             gens_id=[(2, 2), (3, 2)])
# act1.bus_after_action.get_grid2op_dict()

# second unary action
act2 = init_state.get_emptyact()
act2.set_subid(id_sub2)
act2.get_elem_sub()
act2.set_bus(lines_id=[(1, 2), (4, 1), (6, 2), (17, 1)],
             loads_id=[(3, 2)])
# act2.bus_after_action.get_grid2op_dict()
# print(unitary_actions[1])

# computing obs_target load flow with the effect of combined action: this is the ground truth to compare to

# combined_action = unitary_actions[0] + unitary_actions[1]
# obs_target = obs_start.simulate(combined_action, time_step=0)[0]

state = State.from_grid2op_obs(obs_start, subs_actions_unary=[act1, act2])
# TODO in a generic way for any change between 2 buses configuration