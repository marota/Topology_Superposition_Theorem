# %%
import warnings
import numpy as np
from tqdm import tqdm
from lightsim2grid import LightSimBackend
from lightsim2grid.solver import SolverType
from lightsim2grid.securityAnalysis import SecurityAnalysisCPP  # lightsim2grid
from lightsim2grid.gridmodel import init_from_pandapower
import time
import pandapower as pp  # pandapower
import pandapower.networks as pn  # grid cases
import plotly.graph_objects as go  # plotting
from superposition_theorem import State  # this study
import grid2op
from grid2op.Parameters import Parameters 
from grid2op.Chronics import ChangeNothing
import tempfile
import os
import pypowsybl
import pypowsybl.network
from pypowsybl.network import convert_from_pandapower
import pickle

# ordered per number of branches
case_names = ["case14", 
              "case118", "case_illinois200", 
              "case300", 
              "case1354pegase",
              "case1888rte", "GBnetwork", "case3120sp", "case2848rte", "case2869pegase", 
              "case6495rte", 
              "case6515rte",
              "case9241pegase"
             ]

# %%
def compute_extended_ST(case, nb_unit_acts=1):
    # technical details for creating a state
    with tempfile.TemporaryDirectory() as dir:
        grid_path = os.path.join(dir, "grid.json")
        pp.to_json(case, grid_path)
        param = Parameters()
        param.ENV_DC = True  # force the computation of the powerflow in DC mode
        param.MAX_LINE_STATUS_CHANGED = 99999
        param.MAX_SUB_CHANGED = 99999
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("blank",
                               chronics_class=ChangeNothing,
                               grid_path=grid_path,
                               test=True,
                               backend=LightSimBackend(),  # for speed to compute the initial information
                               param=param,
                               _add_to_name=f"{case.bus.shape[0]}_bus",
                               )
            env.change_parameters(param)
            env.change_forecast_parameters(param)

    obs = env.reset()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        state = State.from_grid2op_obs(obs,
                                       line_ids_disc_unary=tuple(range(env.n_line)),
                                       when_error="warns")
    
    nb_line_per_sub = np.zeros(env.n_sub)
    for sub_id in range(env.n_sub):
        nb_line_per_sub[sub_id] += (type(env).line_or_to_subid == sub_id).sum()
        nb_line_per_sub[sub_id] += (type(env).line_ex_to_subid == sub_id).sum()
        
    # takes action on the first two substation where it is possible todo make it "more random"
    sub_ids = np.where((env.sub_info >= 4) & (nb_line_per_sub >= 2))[0][:nb_unit_acts]  

    unit_acts = []
    for sub_id in sub_ids:
        un_act = state.get_emptyact()
        un_act.set_subid(sub_id)
        elems = un_act.get_elem_sub()
        # assign a powerline per nodes at least (todo add more "randomness")
        topo = {"lines_id" : [(l_id, lnum % 2 + 1) for lnum, l_id in enumerate(elems["lines_id"])]}
        # randomnly assign a bus to anything else
        for k in ["loads_id", "gens_id", "storages_id"]:
            if k not in elems:
                continue
            tmp_ = np.random.choice([1, 2], len(elems[k]))
            topo[k] = [(el, tmp) for el, tmp in zip(elems[k], tmp_)]
        un_act.set_bus(**topo)
        unit_acts.append(un_act)
    state.add_unary_actions_grid2op(obs, subs_actions_unary=unit_acts)
    
    _, total_time, nb_cont  = state.compute_flows_n1(subs_actions=unit_acts, line_ids=tuple(range(env.n_line)))
    return total_time, nb_cont

# %%
def compute_lightsim2grid(case, dc=True, lodf=False):
    """compute the full security analysis using lightsim2grid"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        gridmodel = init_from_pandapower(case)
    # perform the action
    # .XXX(blablabla)
    if not dc and lodf:
        raise RuntimeError("Impossible to compute AC powerflow using LODF")
    
    # start the initial computation
    if dc:
        V = gridmodel.dc_pf(1.04 * np.ones(case.bus.shape[0], dtype=np.complex128), 10, 1e-7)
    else:
        V = gridmodel.ac_pf(1.04 * np.ones(case.bus.shape[0], dtype=np.complex128), 10, 1e-7)
    if V.shape[0] == 0:
        # ac pf has diverged
        warnings.warn(f"Impossible to compute the security analysis for {case.bus.shape[0]}: divergence")
        return None, 0
    
    # initial the model
    if not lodf:
        sec_analysis = SecurityAnalysisCPP(gridmodel)
        if dc:
            sec_analysis.change_solver(SolverType.KLUDC)
        for branch_id in range(len(gridmodel.get_lines()) + len(gridmodel.get_trafos())):
            sec_analysis.add_n1(branch_id)
        
        # now do the security analysis
        beg = time.perf_counter()
        sec_analysis.compute(V, 10, 1e-7)
        vs_sa = sec_analysis.get_voltages()
        mw_sa = sec_analysis.compute_power_flows()
        tot_time = time.perf_counter() - beg
        nb_solved = sec_analysis.nb_solved()
    else:
        res_powerflow = 1.0 * np.concatenate((gridmodel.get_lineor_res()[0], gridmodel.get_trafohv_res()[0]))
        beg = time.perf_counter()
        LODF_mat = gridmodel.get_lodf()
        mat_flow = np.tile(res_powerflow, LODF_mat.shape[0]).reshape(LODF_mat.shape)
        por_lodf = mat_flow + LODF_mat.T * mat_flow.T
        tot_time = time.perf_counter() - beg
        nb_solved = LODF_mat.shape[0]
    return tot_time, nb_solved
    

# %%
def compute_pandapower(case):
    pp.rundcpp(case)  # run initial powerflow
    
    nb_cont = 0
    tot_time = 0.
    
    in_service_col_num = (case.line.columns == "in_service").nonzero()[0][0]
    # now do the security analysis
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for branch_id in range(case.line.shape[0]):
            beg = time.perf_counter()
            case.line.iloc[branch_id, in_service_col_num] = False
            pp.rundcpp(case, check_connectivity=False)
            if case["_ppc"]["success"]:
                tot_time += time.perf_counter() - beg
                nb_cont += 1
            case.line.iloc[branch_id, in_service_col_num] = True
        
    in_service_col_num = (case.trafo.columns == "in_service").nonzero()[0][0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for branch_id in range(case.trafo.shape[0]):
            beg = time.perf_counter()
            case.trafo.iloc[branch_id, in_service_col_num] = False
            pp.rundcpp(case, check_connectivity=False)
            if case["_ppc"]["success"]:
                tot_time += time.perf_counter() - beg
                nb_cont += 1
            case.trafo.iloc[branch_id, in_service_col_num] = True
    return tot_time, nb_cont

# %%
def get_pypowsybl_grid(case):
    n_vl = case.bus.shape[0]
    if n_vl == 9:
        return pypowsybl.network.create_ieee9()
    elif n_vl == 14:
        return pypowsybl.network.create_ieee14()
    elif n_vl == 57:
        return pypowsybl.network.create_ieee57()
    elif n_vl == 118:
        return pypowsybl.network.create_ieee118()
    elif n_vl == 300:
        return pypowsybl.network.create_ieee300()
    if n_vl < 3000:
        # otherwise OutOfMemory:
        # pypowsybl._pypowsybl.PyPowsyblError: java.lang.OutOfMemoryError: Garbage-collected heap size exceeded. 
        # Consider increasing the maximum Java heap size, for example with '-Xmx'
        return convert_from_pandapower(case)
    return None


def compute_pypowsybl(case, dc=True):
    grid = get_pypowsybl_grid(case)
    if grid is None:
        return None
    
    analysis = pypowsybl.security.create_analysis()
    analysis.add_single_element_contingencies(grid.get_lines().index)
    analysis.add_single_element_contingencies(grid.get_2_windings_transformers().index)
    analysis.add_monitored_elements(branch_ids=grid.get_lines().index)
    analysis.add_monitored_elements(branch_ids=grid.get_2_windings_transformers().index)
    beg = time.perf_counter()
    if dc:
        res = analysis.run_dc(grid)
    else:
        res = analysis.run_ac(grid)
    tot_time = time.perf_counter() - beg
    nb_cont = grid.get_lines().shape[0] + grid.get_2_windings_transformers().shape[0]
    return tot_time, nb_cont

# %%
res_table = []
res_per_cont = []
nb_branch = []
for case_nm in tqdm(case_names):
    this_row = [case_nm, None, None, None, None, None, None, None]  # total time
    this_row_per_cont = [case_nm, None, None, None, None, None, None, None]  # time for a single contingency
    
    # retrieve the case file from pandapower
    case = getattr(pn, case_nm)()
    nb_branch.append(case.line.shape[0] + case.trafo.shape[0])
    
    # use extended ST
    total_time, nb_cont = compute_extended_ST(case)
    this_row[1] = total_time
    if total_time is not None:
        this_row_per_cont[1] = total_time / nb_cont
    ##### end extended ST
    
    # use lightsim2grid (DC)
    total_time, nb_cont = compute_lightsim2grid(case)
    this_row[2] = total_time
    if total_time is not None:
        this_row_per_cont[2] = total_time / nb_cont
    ##### end lightsim2grid
    
    # use lightsim2grid (DC based on LODF)
    total_time, nb_cont = compute_lightsim2grid(case, lodf=True)
    this_row[3] = total_time
    if total_time is not None:
        this_row_per_cont[3] = total_time / nb_cont
    ##### end lightsim2grid
    
    # use pandapower
    total_time, nb_cont = compute_pandapower(case)
    this_row[4] = total_time
    if total_time is not None:
        this_row_per_cont[4] = total_time / nb_cont
    ##### end 
    
    # use lightsim2grid (AC)
    total_time, nb_cont = compute_lightsim2grid(case, dc=False)
    this_row[5] = total_time
    if total_time is not None:
        this_row_per_cont[5] = total_time / nb_cont
    ##### end lightsim2grid
    
    # use pypowsybl (DC)
    res = compute_pypowsybl(case, dc=True)
    if res is not None:
        total_time, nb_cont = res
        this_row[6] = total_time
        if total_time is not None:
            this_row_per_cont[6] = total_time / nb_cont
    ##### end pypowsybl
    
    # use pypowsybl (AC)
    res = compute_pypowsybl(case, dc=False)
    if res is not None:
        total_time, nb_cont = res
        this_row[7] = total_time
        if total_time is not None:
            this_row_per_cont[7] = total_time / nb_cont
    ##### end pypowsybl
    
    res_table.append(this_row)
    res_per_cont.append(this_row_per_cont)

headers = ["Grid",
           "Ext ST",
           "lightsim2grid (DC)",
           "lightsim2grid (LODF)",
           "pandapower (DC)",
           "lightsim2grid (AC)",
           "pypowsybl (DC)",
           "pypowsybl (AC)",
           ]

with open("computation_times.pickle", "wb") as f:
    pickle.dump(obj=[headers, res_table, res_per_cont, nb_branch], file=f)


