

import warnings
warnings.filterwarnings("ignore")

import grid2op
import numpy as np
from lightsim2grid import LightSimBackend
from grid2op.Parameters import Parameters
from core.superposition_theorem import *

if __name__ == "__main__":
    param = Parameters()
    param.ENV_DC = True
    param.MAX_LINE_STATUS_CHANGED = 99999
    param.MAX_SUB_CHANGED = 99999
    param.NO_OVERFLOW_DISCONNECTION=True
    #backend=LightSimBackend()
    env = grid2op.make("l2rpn_case14_sandbox", param=param)#,backend=backend)


    chronic_id=0

    #if you want to reset the environment on your chronic, always do env.set_id first
    env.set_id(chronic_id)
    env.reset()

    unitary_action_list=[{'set_bus':{'substations_id':[(5,(1,1,2,2,1,2,2))]}},#sub5
    {'set_bus':{'substations_id':[(4,(2,1,2,1,2))]}},#sub4
    {'set_bus':{'substations_id':[(3,(2,1,1,2,1,2))]}},#sub3
    {'set_bus':{'substations_id':[(8,(2,1,1,2,2))]}},#sub8
    {'set_bus':{'substations_id':[(1,(1,2,1,2,2,2))]}}#sub1
    ]

    obs_start, *_ = env.step(env.action_space({}))

    from grid2op.PlotGrid import PlotMatplot
    plot_helper = PlotMatplot(env.observation_space)
    fig = plot_helper.plot_obs(obs_start,line_info="p")
    fig.show()

    id_sub1=5#3#2#3
    id_sub2=4#7#4#7
    idls_lines=[]
    idls_subs=[id_sub1,id_sub2]
    unitary_actions=[env.action_space(unitary_action_list[0]),env.action_space(unitary_action_list[1])]

    combined_action=unitary_actions[0]+unitary_actions[1]
    print(combined_action)
    obs_target=obs_start.simulate(combined_action,time_step=0)[0]
    fig = plot_helper.plot_obs(obs_target,line_info="p")
    fig.show()

    #try to recover obs_target from unitary obs

    obs1=obs_start.simulate(unitary_actions[0],time_step=0)[0]
    obs2=obs_start.simulate(unitary_actions[1],time_step=0)[0]

    (ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1, ind_lex_node1_sub1)=get_sub_node1_idsflow(obs1, id_sub1)
    (ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2, ind_lex_node1_sub2)=get_sub_node1_idsflow(obs2, id_sub2)

    #virtual line flows
    p_start_sub1=get_virtual_line_flow(obs_start, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1, ind_lex_node1_sub1)
    p_start_sub2=get_virtual_line_flow(obs_start, ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2, ind_lex_node1_sub2)

    p_obs1_sub1=get_virtual_line_flow(obs1, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1, ind_lex_node1_sub1)
    p_obs1_sub2=get_virtual_line_flow(obs1, ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2, ind_lex_node1_sub2)

    p_obs2_sub1=get_virtual_line_flow(obs2, ind_load_node1_sub1, ind_prod_node1_sub1, ind_lor_node1_sub1, ind_lex_node1_sub1)
    p_obs2_sub2=get_virtual_line_flow(obs2, ind_load_node1_sub2, ind_prod_node1_sub2, ind_lor_node1_sub2, ind_lex_node1_sub2)

    #beta1=0.9879600151073893
    #beta2=1.0838056920761117
    #obs_start.p_or*(1-beta1-beta2)+beta1*obs1.p_or+beta2*obs2.p_or

    #[1-p_or_connect_idls[i][j]/p_or_obs_start[j]
    a=np.array([[1,1-p_obs2_sub1/p_start_sub1],
                [1-p_obs1_sub2/p_start_sub2,1]])

    b=np.ones(2)
    betas=np.linalg.solve(a,b)

    p_target_computed=betas[0]*obs1.p_or+betas[1]*obs2.p_or+(1-betas.sum())*obs_start.p_or

    assert(np.all((np.round(obs_target.p_or-p_target_computed,4)==0.0)))

    check_obs_target=True
    compute_flows_superposition_theorem(idls_lines,idls_subs,obs_start,unitary_actions,check_obs_target)



