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