import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import math
import pickle
import random
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from time import perf_counter as pc

def network_training(city_name, city_network_filename, distance_and_times_filename, locations_and_windows_filename, 
                     num_to_generate_FM, num_to_generate_LM):
    # Get complete network: F, D, S, P_locations, 
    city_network_df = pd.read_excel(city_network_filename, sheet_name='node_info')
    with open(distance_and_times_filename, 'rb') as handle:
        distance_and_times = pickle.load(handle)
    with open(locations_and_windows_filename, 'rb') as handle:
        locations_and_windows = pickle.load(handle)
    
    # Maximum possible size of network
    max_num_FMs = len(city_network_df.loc[city_network_df['type'].str.contains('FM')])
    max_num_DSPs = len(city_network_df.loc[city_network_df['type'].str.contains('LM')])
    max_num_lockers = len(city_network_df.loc[city_network_df['type'].str.contains('locker')])
    max_locker_nodes_list = list(city_network_df.loc[city_network_df['type']=='locker']['node'])
    max_num_package_destinations = len(city_network_df.loc[city_network_df['type'].str.contains('package')])
    max_num_nodes = len(city_network_df)
    
    # Distances and travel times
    distances, travel_times = get_distances_and_times_single(distance_and_times) 
    distance_matrix = get_distance_matrix_single(distances, max_num_nodes)
    travel_time_matrix = get_travel_time_matrix_single(travel_times, max_num_nodes)
    earliest, latest = get_time_windows(locations_and_windows)
    bigM_matrix = get_bigM_matrix_single(travel_times, max_num_nodes)
        
    # Last-miler nodes - READ FROM FILE
    last_milers_df = pd.read_excel(city_network_filename, sheet_name='last_milers')
    all_LM_nodes = list(last_milers_df['LM_id'])
    num_vehicles_per_DSP = list(last_milers_df['num_vehicles'])
    assert len(num_vehicles_per_DSP) == max_num_DSPs, 'Not all DSPs have vehicles. Recheck network config file.'
    
    # Costs and vehicles
    num_vehicles_per_FM = [1]*max_num_FMs
    cost_per_km_for_FM = [0.9]*max_num_FMs
    num_vehicles_per_DSP = list(last_milers_df['num_vehicles'])
    cost_per_km_for_DSP = list(last_milers_df['cost_per_km_for_DSP'])
    cost_per_DSP_vehicle = 100
    
       
    # Satellite nodes
    max_locker_nodes = list(city_network_df.loc[city_network_df['type']=='locker']['node'])
    # Total satellite capacities
    max_locker_capacities = {}
    max_locker_capacities[5] = 25
    max_locker_capacities[6] = 25
    max_locker_capacities[7] = 10
    all_locker_nodes = list(max_locker_capacities.keys())
    all_locker_capacities = list(max_locker_capacities.values())
    
    
    # Electricity generation info - READ FROM FILE
    electricity_inputs = pd.read_excel(city_network_filename, sheet_name='electricity_generation_breakdwn')
    emission_factor = list(electricity_inputs['Emission Factor'])
    generation_percentage = list(electricity_inputs['Generation Percentage'])
    
    # Battery capacity of electric vehicle in kWh
    battery_capacity = 7
    
    # Time violation penalty
    time_violation_penalty = 1000
        
    # Emissions
    fm_engine_params = pd.read_excel(city_network_filename, sheet_name='first_mile_vehicle_engine_param')   
    load = 2500
    emissions_matrix_ICE = compute_emissions_ICE_single(fm_engine_params, load, distance_matrix, travel_time_matrix, max_num_nodes)
    emissions_matrix_EV = compute_emissions_EV_single(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, max_num_nodes)

    # Start generation 
    feasible_ys = []
    feasible_lamdas = []
    ps = []
    dests = []
    fmdeps = []
    fmnods = []
    fmarcs = []
    dspdeps = []
    dspnods = []
    dsparcs = []
    lockconf = []
    sellocknods=[]
    pckconf = []
       
    counter = 0    
    print('Generating training data points...')
    while counter < num_to_generate_FM:
        locker_config = generate_locker_config(max_num_lockers)
        lockconf.append(locker_config)
        selected_locker_nodes = [a*b for a,b in zip(locker_config, list(max_locker_capacities.keys()))]
        selected_locker_nodes = [s for s in selected_locker_nodes if s>0]
        sellocknods.append(selected_locker_nodes)
        total_available_capacity = sum([a*b for a,b in zip(locker_config, all_locker_capacities)])
        package_dest_config = generate_package_config(max_num_package_destinations, int(max_num_package_destinations/10), total_available_capacity)
#         package_dest_config = generate_package_config(max_num_package_destinations, int(max_num_package_destinations/5), total_available_capacity)
        pckconf.append(package_dest_config)  
        fm_config = create_fm_config(max_num_FMs)
        lm_config = create_lm_config(max_num_DSPs)
    
        package_dest_ids, Pf, destinations,fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs = get_sub_problem_info(city_network_df, fm_config, lm_config, locker_config, package_dest_config, selected_locker_nodes)
        ps.append(package_dest_ids)
        dests.append(destinations)
        fmdeps.append(fm_depots)
        fmnods.append(fm_f_nodes)
        fmarcs.append(fm_f_arcs)        
        dspdeps.append(dsp_depots)
        dspnods.append(dsp_d_nodes)
        dsparcs.append(dsp_d_arcs)
        
        # generate y, lamda and save
        y_sol = generate_feasible_y(max_num_package_destinations, max_num_DSPs, package_dest_ids, all_LM_nodes, dsp_depots)
        feasible_ys.append(y_sol)
        lamda_sol = generate_feasible_lamda(max_num_package_destinations, max_num_nodes, max_locker_capacities,selected_locker_nodes, package_dest_ids)
        feasible_lamdas.append(lamda_sol)
    
        # update counter
        counter += 1
      
    # First mile data generation
    print('FM data generation...')
    first_mile_data_generation(feasible_lamdas, max_num_package_destinations, max_locker_nodes_list,
                      ps, fmdeps, fmnods, fmarcs, sellocknods, num_vehicles_per_FM, cost_per_km_for_FM, 
                       distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest, 
                       emissions_matrix_ICE, time_violation_penalty, city_name)
    
    # Last mile data generation
    print('LM data generation...')
    feasible_ys_used = feasible_ys[:num_to_generate_LM]
    feasible_lamdas_used = feasible_lamdas[:num_to_generate_LM]
    last_mile_data_generation(feasible_ys_used, feasible_lamdas_used, max_num_package_destinations, max_num_DSPs, max_locker_nodes_list,
                      ps, dspdeps, dspnods, dsparcs, sellocknods, dests, num_vehicles_per_DSP, cost_per_km_for_DSP, distance_matrix,
                           travel_time_matrix, bigM_matrix, earliest, latest, emissions_matrix_EV, time_violation_penalty,
                          city_network_df, city_name)    
    return

def run_instance(city_name, problem_instance, fm_regr_model, lm_regr_model, features_LM, training_metrics, satellite_penalty, vehicle_penalty):
    # --- Read in complete network ---
    distance_and_times_filename = 'instances/'+city_name+'_distances_times_2_3_3_50.pickle'
    locations_and_windows_filename = 'instances/'+city_name+'_locations_windows_2_3_3_50.pickle'
    problem_info_filename = 'instances/'+city_name+'_2_3_3_50.xlsx'
    
    with open(distance_and_times_filename, 'rb') as handle:
        distance_and_times = pickle.load(handle)
    with open(locations_and_windows_filename, 'rb') as handle:
        locations_and_windows = pickle.load(handle)
        
    # Get distances and times for whole network
    distances, travel_times = get_distances_and_times_single(distance_and_times)     
    max_num_FMs = 2
    max_num_DSPs = 3
    max_num_lockers = 3
    max_num_packages = 50
    city_instance_df = pd.read_excel(problem_info_filename, sheet_name='node_info')
    max_num_nodes = len(city_instance_df)

    distance_matrix = get_distance_matrix_single(distances, max_num_nodes)
    travel_time_matrix = get_travel_time_matrix_single(travel_times, max_num_nodes)
    earliest, latest = get_time_windows(locations_and_windows)
    bigM_matrix = get_bigM_matrix_single(travel_times, max_num_nodes)

    # FM & LM vehicles and emissions
    fm_engine_params = pd.read_excel(problem_info_filename, sheet_name='first_mile_vehicle_engine_param')    
    electricity_inputs = pd.read_excel(problem_info_filename, sheet_name='electricity_generation_breakdwn')
    emission_factor = list(electricity_inputs['Emission Factor'])
    generation_percentage = list(electricity_inputs['Generation Percentage'])
    # Battery capacity of electric vehicle in kWh
    battery_capacity = 7
    load = 2500 #Assuming average load of 2500kg...
    emissions_matrix_ICE = compute_emissions_ICE_single(fm_engine_params, load, distance_matrix, travel_time_matrix, max_num_nodes)
    emissions_matrix_EV = compute_emissions_EV_single(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, max_num_nodes)
    time_violation_penalty = 1000

    # First-miler nodes
    all_FM_nodes = list(city_instance_df.loc[city_instance_df['type'].str.contains('FM')]['node'])
    num_vehicles_per_FM = [1]*max_num_FMs
    cost_per_km_for_FM = [0.9]*max_num_FMs

    # Last-miler nodes
    all_LM_nodes = list(city_instance_df.loc[city_instance_df['type'].str.contains('LM')]['node'])
    num_vehicles_per_DSP = [1]*max_num_DSPs
    cost_per_km_for_DSP = [0.9]*max_num_DSPs

    # Satellite nodes
    max_locker_nodes = list(city_instance_df.loc[city_instance_df['type']=='locker']['node'])
    # Total satellite capacities
    max_locker_capacities = {}
    max_locker_capacities[5] = 25
    max_locker_capacities[6] = 25
    max_locker_capacities[7] = 10
    all_locker_nodes = list(max_locker_capacities.keys())
    all_locker_capacities = list(max_locker_capacities.values())  
    
    # --- Instance problem config ---
    fm_config = problem_instance[0]
    lm_config = problem_instance[1]
    locker_config = problem_instance[2]
    num_packages_in_instance = problem_instance[3]    
    package_config = [0]*max_num_packages
    package_config[:num_packages_in_instance] = [1]*num_packages_in_instance
    problem_config = fm_config + lm_config + locker_config + package_config
    instance_name = city_name+'_'+str(sum(fm_config))+'_'+str(sum(lm_config))+'_'+str(sum(locker_config))+'_'+str(sum(package_config))
    print('>>> Solving', instance_name)
    selected_locker_nodes = [a*b for a,b in zip(locker_config, list(max_locker_capacities.keys()))]
    selected_locker_nodes = [s for s in selected_locker_nodes if s>0]
#     package_ids, Pf, destinations, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs = get_problem_instance_info(city_instance_df, problem_config, selected_locker_nodes)
    
    package_ids, Pf, destinations, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs = get_sub_problem_info(city_instance_df, fm_config, lm_config, locker_config, package_config, selected_locker_nodes)
       
    
    # --- Optimization ---
    # Create a dict that maps z_f to y_p_d and lamda_p_s
    feature_names = list(features_LM.columns)
    z_feature_names = []
    for f in range(len(feature_names)):
        z_feature_names.append('z_'+str(f))

    z_mapping = {z_feature_names[i]: feature_names[i] for i in range(len(z_feature_names))}

    # Create a new Gurobi model
    clopt_timer = pc()
    model_instance_name = 'CL_Leader_'+instance_name
    model = gp.Model(model_instance_name)
    model.setParam('OutputFlag', False)
    model.Params.MIPGap = 1e-2
    model.Params.threads = 4
    model.Params.timelimit = 2700

    
    # ----- Sets -----
    all_D = range(max_num_DSPs)
    instance_D = range(sum(lm_config))
    all_F = range(max_num_FMs)
    all_P = list(range(max_num_packages)) 
    delivery_P = package_ids 
    not_considered = [value for value in all_P if value not in delivery_P]
    unused_lockers = list(set(all_locker_nodes) - set(selected_locker_nodes))
    
    # ----- Variables -----
    # ----- Leader Variables -----
    # y_pd = 1 if parcel p is offered to DSP d by the leader
    y = {(p, d): model.addVar(vtype=GRB.BINARY, name='y_%d_%d' % (p, d)) for p in all_P for d in all_D}
    # lamda_ps = 1 if parcel p is placed at satellite s
    lamda = {(p, s): model.addVar(vtype=GRB.BINARY, name='lamda_%d_%d' % (p, s)) for p in all_P for s in all_locker_nodes}
    # z_f is based on value of y and lamda
    z = {f: model.addVar(vtype=GRB.BINARY, name='z_%d' % f) for f in range(len(z_feature_names))}
    # first mile emissions
    emissions_first = model.addVars(max_num_FMs, lb=0.0, name='first_emis')
    # last mile emissions
    emissions_last = model.addVars(sum(lm_config), lb=0.0, name='last_emis')
    
    
    # Number of satellites used
    phi = {s: model.addVar(vtype=GRB.BINARY, name='phi_%d' % s) for s in all_locker_nodes}    
    # Number of vehicles (DSPs) used
    zeta = {d: model.addVar(vtype=GRB.BINARY, name='zeta_%d' % d) for d in all_D}   
    
    instance_name = instance_name +'_CL_penalties'+str(satellite_penalty)+str(vehicle_penalty)
    
    # Update model to integrate new variables
    model.update()
    
    # Add constraints on y and lamda that fix the values of non-existents packages
    # to zero
    for p in not_considered:
        for d in all_D:
            model.addConstr(model.getVarByName('y_%d_%d' % (p,d)) == 0)
        for s in all_locker_nodes:
            model.addConstr(model.getVarByName('lamda_%d_%d' % (p,s)) == 0)
    # Add constraints on y and lamda that ensure that unused lockers remain unused
    for p in all_P:
        for s in unused_lockers:
            model.addConstr(model.getVarByName('lamda_%d_%d' % (p,s)) == 0)

    # Add constraints that link z to y and lamda
    for fz in range(len(z_feature_names)):
        model.addConstr(model.getVarByName(list(z_mapping.keys())[fz]) == model.getVarByName(list(z_mapping.values())[fz]))

    # Add constraints that fix the values of the DSP not in use
    if len(list(instance_D)) < len(list(all_D)):
        for p in delivery_P:
            model.addConstr(model.getVarByName('y_%d_%d' % (p, max(list(all_D)))) == 0)

    # ----- Leader Constraints -----
    # Respect satellites' capacity constraint
    for s in selected_locker_nodes:
        model.addConstr(gp.quicksum(lamda[p, s] for p in delivery_P) <= max_locker_capacities[s])
    # A parcel should only be assigned to one satellite
    for p in delivery_P:
        model.addConstr(gp.quicksum(lamda[p, s] for s in selected_locker_nodes) == 1)
    # Only one DSP should be assigned to each parcel
    for p in delivery_P:
        model.addConstr(gp.quicksum(y[p, d] for d in instance_D) == 1)

    # Add learned constraints for first mile
    for f in all_F:
        add_predictor_constr(model, fm_regr_model, lamda, emissions_first[f])

    # Add learned constraints for first mile
    for d in instance_D:
        add_predictor_constr(model, lm_regr_model, z ,emissions_last[d])
        
    # Add constraints on the number of vehicles and satellites used
    for p in all_P:
        # number of vehicles used
        for d in all_D:
            model.addConstr(model.getVarByName('zeta_%d' % d) >= model.getVarByName('y_%d_%d' % (p,d)), name='vehPen')
        # number of satellites used
        for s in all_locker_nodes:
            model.addConstr(model.getVarByName('phi_%d' % s) >= model.getVarByName('lamda_%d_%d' % (p,s)), name='satPen')
    
    

    # ----- Leader Objective Function -----
    obj_expr = gp.LinExpr()
    for f in all_F:
        obj_expr += emissions_first[f]
    for d in instance_D:
        obj_expr += emissions_last[d]
        
    obj_expr += gp.quicksum(satellite_penalty*phi[s] for s in selected_locker_nodes)
    obj_expr += gp.quicksum(vehicle_penalty*zeta[d] for d in instance_D)
    
   
    # Set objective
    model.setObjective(obj_expr, GRB.MINIMIZE)
    instance_solution = None

    # Optimize the model
    #print('Optimizing CL Leader model')
    try:
        model.optimize()

        # Take y and lamda, solve param foll probs and get x and w
        y_res_list = [var for var in model.getVars() if 'y' in var.VarName]
        y_res_list_values = [abs(item.X) for item in y_res_list]
        y_sol_cl = np.array(y_res_list_values).reshape(max_num_packages, max_num_DSPs)
        lamda_res_list = [var for var in model.getVars() if 'lamda' in var.VarName]
        lamda_res_list_values = [abs(item.X) for item in lamda_res_list]
        lamda_sol_small = np.array(lamda_res_list_values).reshape(max_num_packages, max_num_lockers)
        lamda_sol_cl = np.zeros((max_num_packages,max_num_nodes))
        lamda_sol_cl[:, all_locker_nodes] = lamda_sol_small
        CL_total_time = pc()-clopt_timer

        # --- Solve parameterized follower problems ---
        mip_time_limit = 300
        mip_emphasis = 0 # default: balance feasibility and optimality  

        fm_follower_timer = pc()
        fm_Objs, fm_BBs, fm_Gaps, fm_Times, fm_Sol_dfs = solve_param_first_mile_followers_static_single(lamda_sol_cl, Pf, fm_f_nodes, fm_f_arcs, fm_depots, cost_per_km_for_FM, 
                                          num_vehicles_per_FM, selected_locker_nodes, distance_matrix, travel_time_matrix,
                                          bigM_matrix, earliest, latest, mip_time_limit, mip_emphasis)
        total_FM_foll_time  = pc() - fm_follower_timer

        lm_follower_timer = pc()
        lm_Objs, lm_BBs, lm_Gaps, lm_Times, lm_Sol_dfs = solve_param_last_mile_followers_static_single(y_sol_cl, lamda_sol_cl, dsp_d_nodes, dsp_d_arcs, dsp_depots,
                                                      cost_per_km_for_DSP, package_ids, destinations, selected_locker_nodes,
                                                       distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, 
                                                      time_violation_penalty, mip_time_limit, mip_emphasis)
        total_LM_foll_time  = pc() - lm_follower_timer

        opt_timings = []
        opt_timings.append(CL_total_time)
        opt_timings.append(total_FM_foll_time)
        opt_timings.append(total_LM_foll_time)

        # Compute real solution
        instance_solution = compute_upper_bound(fm_Sol_dfs, lm_Sol_dfs, emissions_matrix_ICE, emissions_matrix_EV)

         # Write solution to file    
        write_instance_results_to_file(instance_name, lamda_sol_cl, fm_Sol_dfs, lm_Sol_dfs, distance_matrix, num_vehicles_per_FM, num_vehicles_per_DSP, time_violation_penalty, emissions_matrix_EV, emissions_matrix_ICE, destinations, instance_solution,training_metrics, opt_timings)

        print('Instance solved.\n')
    except:
        print('Error solving CL leader')
    return instance_solution

def get_distances_and_times_single(distance_and_times):
    distances = []
    travel_times = []

    for i in range(len(distance_and_times)):
        a = distance_and_times[i]['from']
        b = distance_and_times[i]['to']

        arc_distances_dict = {'from': int(a), 'to': int(b), 'distances': distance_and_times[i]['average_distance']}
        distances.append(arc_distances_dict)

        arc_travel_times_dict = {'from': int(a), 'to': int(b), 'travel_times': distance_and_times[i]['average_travel_time']}
        travel_times.append(arc_travel_times_dict)


    # Sort so that array goes from 0 to numNodes
    distances = sorted(distances, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    travel_times = sorted(travel_times, key=lambda k: (k.get('from', 0),k.get('to', 0)))    
    return distances, travel_times

def get_distance_matrix_single(instance_tt_distances, numNodes):   
    # Put in matrix from
    all_distances = []
    for i in range(len(instance_tt_distances)):
        all_distances.append(instance_tt_distances[i]['distances']/1000) # convert to kilometers

    distance_matrix = [all_distances[i:i + numNodes] for i in range(0, len(all_distances), numNodes)]    
    return distance_matrix

def get_travel_time_matrix_single(instance_tt_travel_times, numNodes):    
    # Put in matrix from
    all_travel_times = []
    for i in range(len(instance_tt_travel_times)):
        all_travel_times.append(instance_tt_travel_times[i]['travel_times'])

    travel_time_matrix = [all_travel_times[i:i + numNodes] for i in range(0, len(all_travel_times), numNodes)]    
    return travel_time_matrix

def get_time_windows(time_windows_data):
    earliest = []
    latest = []
    toHours = 3600000
    
    for i in range(len(time_windows_data)):
        earliest.append(time_windows_data[i]['earliest_delivery'])
        latest.append(time_windows_data[i]['latest_delivery'])
    
    earliest = list(np.array(earliest)/toHours)
    latest = list(np.array(latest)/toHours)
    
    # Prepend time window for depot
    earliest = [min(earliest)] + earliest
    latest = [max(latest)] + latest    
    return earliest, latest

def get_bigM_matrix_single(instance_tt_travel_times, numNodes):    
    # Put in matrix from
    max_travel_time_per_arc = []
    for i in range(len(instance_tt_travel_times)):
        max_travel_time_per_arc.append(2*(instance_tt_travel_times[i]['travel_times']))

    bigM_matrix = [max_travel_time_per_arc[i:i + numNodes] for i in range(0, len(max_travel_time_per_arc), numNodes)]    
    return bigM_matrix

def compute_emissions_ICE_single(fm_engine_params, load, distance_matrix, travel_time_matrix, num_nodes):     
    k_e = float(fm_engine_params['engine friction factor'])
    N_e = float(fm_engine_params['engine speed'])
    V_e = float(fm_engine_params['engine displacement'])
    alpha = float(fm_engine_params['alpha']) # alpha, beta, gamma and lambda are constants from Franceschetti et al. (2013)
    beta = float(fm_engine_params['beta'])
    gamma = float(fm_engine_params['gamma'])
    lamda = float(fm_engine_params['lambda'])
    mu_eng = float(fm_engine_params['curb weight (in kg)'])
    
    emission_ij = []    
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):                
            if travel_time_matrix[i][j]!=0:
                emission = lamda * (k_e * N_e * V_e * travel_time_matrix[i][j] \
                                    + gamma * beta * distance_matrix[i][j] * (distance_matrix[i][j] / travel_time_matrix[i][j])**2 \
                                               + gamma * alpha * (mu_eng + load) * distance_matrix[i][j])                    
                emission_ij.append(emission*2680)  #conversion to gCO2
            else:
                emission_ij.append(0)
                
    # Put in matrix form
    E_ij = [emission_ij[i:i+num_nodes] for i in range(0, len(emission_ij), num_nodes)]    
    return E_ij

def EVCO2(kwh_consumption, emission_factor, generation_percentage):
    energy_mix = sum([emission_factor[i] * generation_percentage[i] for i in range(len(emission_factor))])    
    # if you multiply gCO2eq/kWh with kWh you get gCO2eq
    emissions_gCO2 = kwh_consumption * energy_mix    
    return emissions_gCO2 # in gCO2eq
    
def compute_emissions_EV_single(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_nodes):
    emission_ij = []    
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):  
            if travel_time_matrix[i][j]!=0:
                kwh_consumption = battery_capacity * distance_matrix[i][j] / 100 # kilowatt-hours/100 km
                emission = EVCO2(kwh_consumption, emission_factor, generation_percentage)

                emission_ij.append(emission)
            else:
                emission_ij.append(0)
                                
    # Put in matrix form
    E_ij = [emission_ij[i:i+num_nodes] for i in range(0, len(emission_ij), num_nodes)]                    
    return E_ij

def generate_locker_config(max_num_lockers):    
    # Generate locker configuration
    locker_config = list(np.random.choice([0, 1], size=max_num_lockers))
    if all(elem == 0 for elem in locker_config):
        index = random.randint(0, len(locker_config)-1)
        locker_config[index] = 1
    return locker_config

def generate_package_config(max_num_destinations, max_train_destinations, total_available_capacity):
    # Generate package configuration
    package_config = [0] * max_num_destinations
    indices = random.sample(range(max_num_destinations), random.randint(int(max_train_destinations / 2), max_train_destinations))
    for ind in indices:
        package_config[ind] = 1

    # Ensure sum(package_config) <= total_available_capacity
    current_capacity = sum(package_config)
    if current_capacity > total_available_capacity:
        # Randomly set enough elements to zero to reduce the sum
        ones_indices = [i for i, val in enumerate(package_config) if val == 1]
        indices_to_zero = random.sample(ones_indices, current_capacity - total_available_capacity)
        for ind in indices_to_zero:
            package_config[ind] = 0
    return package_config

def create_fm_config(max_num_FMs):
    fm_config = [0] * max_num_FMs    
    fm_index = random.randint(0, max_num_FMs - 1)
    fm_config[fm_index] = 1    
    return fm_config

def create_lm_config(max_num_DSPs):
    lm_config = [0] * max_num_DSPs    
    dsp_index = random.randint(0, max_num_DSPs - 1)
    lm_config[dsp_index] = 1    
    return lm_config

def get_firstmiler_package_origins(data):
    groups = {}    
    for idx, origin in enumerate(data['first_miler_of_origin']):
        if origin not in groups:
            groups[origin] = []
        groups[origin].append(idx)    
    return groups

def create_bounds(num_vehicles_per_DSP):
    veh_ranges = np.cumsum(num_vehicles_per_DSP)
    bounds = []
    bounds.append(range(veh_ranges[0]))
    for i in range(len(veh_ranges)):
        if i != 0:
            bounds.append(range(veh_ranges[i-1], veh_ranges[i]))
    return bounds

def get_destinations(df):
    destinations = {}
    for index, row in df.iterrows():
        if row['type'] == 'package':
            destinations[int(row['package_id'])] = int(row['node'])
    return destinations

def get_sub_problem_info(city_instance_df, fm_config, lm_config, locker_config, package_config,
                        selected_locker_nodes):        
    problem_config = fm_config + lm_config + locker_config + package_config
    city_instance_df['problem_config'] = problem_config
    city_sub_instance = city_instance_df[city_instance_df['problem_config'] == 1]
    
    # Set half of 'first_miler_of_origin' to 0 and the other half to 1 where 'type' == 'package'
    package_count = city_sub_instance[city_sub_instance['type'] == 'package'].shape[0]
    half_count = package_count // 2
    city_sub_instance.loc[city_sub_instance['type'] == 'package', 'first_miler_of_origin'] = [0] * half_count + [1] * half_count    

    first_depots_df = city_sub_instance.loc[city_instance_df['type'].str.contains('FM')]
    first_depots_df.reset_index(inplace=True, drop=True)
    last_depots_df = city_sub_instance.loc[city_instance_df['type'].str.contains('LM')]
    last_depots_df.reset_index(inplace=True, drop=True)

    package_destination_ids = list(city_sub_instance.loc[city_sub_instance['type']=='package']['package_id'])
    package_destination_ids = [int(x) for x in package_destination_ids]    

    destinations = get_destinations(city_sub_instance)
    
    # Get which packages come from which first-miler
    filtered_city = city_sub_instance.dropna(subset=['package_id'])
    orig_dict = filtered_city[['package_id', 'first_miler_of_origin']].to_dict(orient='list')
    fm_origins = get_firstmiler_package_origins(orig_dict)
    all_origins_bounds = []
    for i in range(len(fm_origins)):
        all_origins_bounds.append(len(fm_origins[i]))
    Pf = create_bounds(all_origins_bounds)    
        
    # FM depots, nodes and arcs
    fm_depots = []
    fm_f_nodes = []
    # Assuming only one depot per FM
    for f in range(len(first_depots_df)):
        fm_depots.append(int(first_depots_df['node'][f]))
        fm_f_nodes.append([int(first_depots_df['node'][f])] + selected_locker_nodes)

    fm_f_arcs = []
    for f in range(len(fm_f_nodes)):
        arcs = [(i, j) for i in fm_f_nodes[f] for j in fm_f_nodes[f] if i!=j]
        fm_f_arcs.append(arcs)
        
    # DSP depots, nodes and arcs
    dsp_depots = []
    dsp_d_nodes = []
    # Assuming only one depot per DSP
    for d in range(len(last_depots_df)):
        dsp_depots.append(int(last_depots_df['node'][d]))
        dsp_d_nodes.append([int(last_depots_df['node'][d])] + selected_locker_nodes + list(destinations.values()))
      
    dsp_d_arcs = []
    for d in range(len(dsp_d_nodes)):
        arcs = [(i, j) for i in dsp_d_nodes[d] for j in dsp_d_nodes[d] if i!=j]
        dsp_d_arcs.append(arcs)        
    return package_destination_ids, Pf, destinations, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs

def generate_feasible_y(max_num_packages, max_num_DSPs, package_ids, all_LM_nodes, dsp_depots):
    rows = max_num_packages
    cols = max_num_DSPs
    
    # Initialize the array with zeros
    feasible_y_sol = np.zeros((rows, cols))
    # Set one value to one in each row randomly
    for i in package_ids:
        j = [all_LM_nodes.index(depot) for depot in dsp_depots][0] # use correct column index
        feasible_y_sol[i, j] = 1       
    return feasible_y_sol 

def generate_feasible_lamda(max_num_packages, max_num_nodes, max_locker_capacities, selected_locker_nodes, package_ids):      
    # Initialize the array with zeros
    feasible_lamda_sol = np.zeros((max_num_packages, max_num_nodes))
    # Set one value to one in each row randomly while maintaining column sum constraint
    for i in package_ids:
        # Generate random column index based on available locker nodes
        col_idx = random.choice(selected_locker_nodes)
        # Adjust column index if adding 1 would exceed locklist[col_idx]
        while feasible_lamda_sol[:, col_idx].sum() >= max_locker_capacities[col_idx]:
            col_idx = random.choice(selected_locker_nodes)
        feasible_lamda_sol[i, col_idx] = 1
    return feasible_lamda_sol

def last_miler_emissions_training(sol_df, emissions_matrix_EV):
    if sol_df is not None:
        x_rows = sol_df[sol_df['name'].str.startswith('x_')]
        x_rows.reset_index(inplace=True)
        # Remove possible duplicates
        if len(x_rows) > 0:
            for i in range(len(x_rows)):
                if x_rows.loc[i]['value'] < 0.1:
                    x_rows.drop([i], inplace=True)
        x_rows.reset_index(inplace=True)

        total_emissions_d = 0

        for i in range(x_rows.shape[0]):
            row = x_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            total_emissions_d += emissions_matrix_EV[row[0]][row[1]]
    else:
        total_emissions_d = None
    return total_emissions_d 

def first_miler_emissions_training(sol_df, emissions_matrix_ICE):
    if sol_df is not None:
        w_rows = sol_df[sol_df['name'].str.startswith('w_')]
        w_rows.reset_index(inplace=True)
        # Remove possible duplicates
        if len(w_rows) > 0:
            for i in range(len(w_rows)):
                if w_rows.loc[i]['value'] < 0.1:
                    w_rows.drop([i], inplace=True)
        w_rows.reset_index(inplace=True)

        total_emissions_f = 0

        for i in range(w_rows.shape[0]):
            row = w_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            total_emissions_f += emissions_matrix_ICE[row[0]][row[1]]
    else:
        total_emissions_f = None
    return total_emissions_f 

def extract_w_static_single(sol_df):
    w_rows = sol_df[sol_df['name'].str.startswith('w')]
    w_rows = w_rows.reset_index(drop=True)
    
    # Remove possible duplicates
    if len(w_rows) > 0:
        for i in range(len(w_rows)):
            if w_rows.loc[i]['value'] < 0.1:
                w_rows.drop([i], inplace=True)
    w_rows.reset_index(inplace=True)    
    
    w_df = pd.DataFrame(columns=['k','i','j'])
    for i in range(w_rows.shape[0]):
        row = w_rows['name'][i].split('_')
        row.pop(0);    
        row = [int(i) for i in row]
        w_df.loc[len(w_df)] = row
    return w_df

def extract_x_static_single(sol_df):    
    if len(sol_df) > 0:
        x_rows = sol_df[sol_df['name'].str.startswith('x')]
        x_rows = x_rows.reset_index(drop=True)

        # Remove possible duplicate
        if len(x_rows) > 0:
            for i in range(len(x_rows)):
                if x_rows.loc[i]['value'] < 0.1:
                    x_rows.drop([i], inplace=True)
        x_rows.reset_index(inplace=True)

        x_df = pd.DataFrame(columns=['i','j'])
        for i in range(x_rows.shape[0]):
            row = x_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            x_df.loc[len(x_df)] = row
    return x_df

def extract_t(lastmiler_final_sol):
    t_df = None
    
    if len(lastmiler_final_sol) > 0:
        t_rows = lastmiler_final_sol[lastmiler_final_sol['name'].str.startswith('t_')]
        t_rows = t_rows.reset_index(drop=True)
        
        # Remove possible duplicate
        if len(t_rows) > 0:
            for i in range(len(t_rows)):
                if t_rows.loc[i]['value'] < 0.01:
                    t_rows.drop([i], inplace=True)
        t_rows = t_rows.reset_index(drop=True)        

        t_df = pd.DataFrame(columns=['j','time'])
        for i in range(t_rows.shape[0]):
            row = t_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(t_rows['value'][i])
            t_df.loc[len(t_df)] = row    
      
    return t_df

def extract_alphas_follower(sol_df):
    alpha_early_df = None
    alpha_late_df = None    
    if len(sol_df) > 0:
        alpha_early_df = sol_df[sol_df['name'].str.startswith('alphaEarly')]
        alpha_early_df = alpha_early_df.reset_index(drop=True)          
        alpha_late_df = sol_df[sol_df['name'].str.startswith('alphaLate')]
        alpha_late_df = alpha_late_df.reset_index(drop=True)        
    return alpha_early_df,alpha_late_df 

def compute_upper_bound(first_m_sol_dfs, last_m_sol_dfs, emissions_matrix_ICE, emissions_matrix_EV):
    # Compute first mile emissions
    fm_emissions = []
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = extract_w_static_single(first_m_sol_dfs[i])
            for t in range(len(w_sol)):
                arc_emission = emissions_matrix_ICE[int(w_sol.loc[t]['i'])][int(w_sol.loc[t]['j'])]
                fm_emissions.append(arc_emission)
    
    # Compute last mile emissions
    lm_emissions = []
    for i in range(len(last_m_sol_dfs)):
        if len(last_m_sol_dfs[i]) > 0:
            x_sol = extract_x_static_single(last_m_sol_dfs[i])
            for t in range(len(x_sol)):
                arc_emission = emissions_matrix_EV[int(x_sol.loc[t]['i'])][int(x_sol.loc[t]['j'])]
                lm_emissions.append(arc_emission)                
    # Sum up and output
    return sum(fm_emissions) + sum(lm_emissions)

def get_FM_distance_and_emissions(first_m_sol_dfs, num_vehicles_per_FM, distance_matrix, emissions_matrix_ICE):
    dist_res = []
    emm_res = []
    
    # Compute first mile emissions
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = extract_w_static_single(first_m_sol_dfs[i]) 
            distances = []
            for j in range(len(w_sol)):
                i_index = w_sol.loc[j]['i']
                j_index = w_sol.loc[j]['j']
                distances.append(distance_matrix[i_index][j_index]) 
            dist_res.append(sum(distances))  
            # Compute total emissions by last-miler
            total_emissions_f = 0            
            for t in range(len(w_sol)):
                i_index = int(w_sol.loc[t]['i'])
                j_index = int(w_sol.loc[t]['j'])
                total_emissions_f += emissions_matrix_ICE[i_index][j_index]
            emm_res.append(total_emissions_f)
        else:
            dist_res.append(0) 
            emm_res.append(0)
    # Create columns based on number of first-milers
    cols = []
    for m in range(len(first_m_sol_dfs)):
        cols.append('First Miler ' + str(m))

    # create dataframe and write to file
    fm_dist_emm_df = pd.DataFrame([dist_res, emm_res], columns = cols, index = (['Total distance (km)', 'Total CO2 emissions (gCO2eq)'])) 
    return fm_dist_emm_df

def get_LM_distance_and_emissions(lm_Sol_dfs, distance_matrix, emissions_matrix_EV):
    dist_res = []
    emm_res = []
    for d in range(len(lm_Sol_dfs)):    
        # If solution exists:
        if len(lm_Sol_dfs[d]) > 0:
            # Compute total distance and emissions by last-miler
            x_sol_final_d = extract_x_static_single(lm_Sol_dfs[d])  
            total_distance_d = 0
            total_emissions_d = 0
            for i in range(len(x_sol_final_d)):
                i_index = int(x_sol_final_d.loc[i]['i'])
                j_index = int(x_sol_final_d.loc[i]['j'])
                total_distance_d += distance_matrix[i_index][j_index]
                total_emissions_d += emissions_matrix_EV[i_index][j_index]
            dist_res.append(total_distance_d)
            emm_res.append(total_emissions_d)
        else:
            dist_res.append(0) 
            emm_res.append(0)

    # Create columns based on number of last-milers
    cols = []
    for i in range(len(lm_Sol_dfs)):
        cols.append('Last Miler ' + str(i))

    # create dataframe and write to file
    dist_emm_df = pd.DataFrame([dist_res, emm_res], columns = cols, index = (['Total distance (km)', 'Total CO2 emissions (gCO2eq)']))    
    return dist_emm_df

def get_locker_assignments(lamda_final):
    assignments = {}
    # Iterate over columns
    for col in range(lamda_final.shape[1]):
        # Get the indices where the value is 1
        indices = np.where(lamda_final[:, col] == 1)[0]

        # If there are indices, add them to the assignments dictionary
        if len(indices) > 0:
            assignments[col] = indices.tolist()            
    return assignments

def get_times_at_destinations(last_m_sol_dfs_final, destinations):       
    all_arrive_times = []    
    for d in range(len(last_m_sol_dfs_final)):    
        # If solution for DSP d exists:
        if len(last_m_sol_dfs_final[d]) > 0:
            x_temp = extract_x_static_single(last_m_sol_dfs_final[d])
            t_temp = extract_t(last_m_sol_dfs_final[d])
            arrive_time_d = x_temp.merge(t_temp, on='j', how='left')
            # Change 'j' to 'Destination node'
            arrive_time_d.rename(columns={'j': 'Destination node'}, inplace=True)            
            # Get times at which the nodes are visited in HH:MM
            arrive_time_d.sort_values(by=['time'],inplace=True)
            converted_hours = []
            for i in range(len(arrive_time_d)):
                hours = int(arrive_time_d.iloc[i]['time'])
                minutes = int(arrive_time_d.iloc[i]['time']*60) % 60
                converted_hours.append("%02d:%02d" % (hours, minutes))
            arrive_time_d['Arrival Time at Destination (HH:MM)'] = converted_hours
            # Drop old time column and i
            arrive_time_d.drop(columns=['time', 'i'], inplace=True)
            # Identify last miler
            arrive_time_d['Carried by Last Miler'] = d
            
            # Add package id
            dests_df = pd.DataFrame(data = list(zip(list(destinations.keys()), list(destinations.values()))),
                        columns=['Package ID', 'Destination node'])
            # Merge on 'Destination node'
            arrive_time_d = arrive_time_d.merge(dests_df, on='Destination node', how='left')
            # Locker nodes show up as NaN so clearly state that they are locker nodes
            arrive_time_d.fillna('N/A - Depot or Locker node', inplace=True)
            
            all_arrive_times.append(arrive_time_d)     

    # Create dataframe and return
    times_at_dest_df = pd.concat(all_arrive_times, axis=0, ignore_index=True)
    # Rearrange columns
    times_at_dest_df = times_at_dest_df[['Package ID', 'Destination node', 'Arrival Time at Destination (HH:MM)', 'Carried by Last Miler']]    
    return times_at_dest_df

def get_FM_follower_objectives(first_m_sol_dfs, num_vehicles_per_FM, distance_matrix):
    follower_objectives = []        
    # Compute follower objectives
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = extract_w_static_single(first_m_sol_dfs[i])          
            distances = []
            for j in range(len(w_sol)):
                i_index = w_sol.loc[j]['i']
                j_index = w_sol.loc[j]['j']
                distances.append(distance_matrix[i_index][j_index])     
            follower_objectives.append(sum(distances))  
        else:
            follower_objectives.append(0)  
    return follower_objectives

def get_LM_follower_objectives(last_m_sol_dfs_final, num_vehicles_per_DSP, distance_matrix, time_violation_penalty):
    follower_objectives = []
    for d in range(len(last_m_sol_dfs_final)):    
        # If solution exists:
        if len(last_m_sol_dfs_final[d]) > 0:
            # Compute total distance by last-miler
            xm_sol_final_d = extract_x_static_single(last_m_sol_dfs_final[d])
            distances = []
            for j in range(len(xm_sol_final_d)):
                i_index = xm_sol_final_d.loc[j]['i']
                j_index = xm_sol_final_d.loc[j]['j']
                distances.append(distance_matrix[i_index][j_index])   
            total_distance = sum(distances)
            # Early and late penalties            
            alphaEarly, alphaLate = extract_alphas_follower(last_m_sol_dfs_final[d])
            total_early_penalty = alphaEarly['value'].sum()
            total_late_penalty = alphaLate['value'].sum()            
            obj_value = total_distance + time_violation_penalty*(total_early_penalty + total_late_penalty)            
            follower_objectives.append(obj_value)  
        else:
            follower_objectives.append(0)             
    return follower_objectives

def compute_follower_objs(fm_FINAL, lm_FINAL, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, time_violation_penalty):
    fm_objs_optimal = get_FM_follower_objectives(fm_FINAL, num_vehicles_per_FM, distance_matrix)    
    lm_objs_optimal = get_LM_follower_objectives(lm_FINAL, num_vehicles_per_DSP, distance_matrix, time_violation_penalty)
    
    objcols = []
    for f in range(len(fm_objs_optimal)):
        objcols.append('FM ' + str(f) + ' Obj optimal')
    for d in range(len(lm_objs_optimal)):
        objcols.append('LM ' + str(d) + ' Obj optimal')
    
    objectives = fm_objs_optimal + lm_objs_optimal
    obj_df = pd.DataFrame([objectives], columns = objcols)   
    return obj_df

def write_instance_results_to_file(instance_name, lamda_sol_cl, fm_Sol_dfs, lm_Sol_dfs, distance_matrix, num_vehicles_per_FM, num_vehicles_per_DSP, time_violation_penalty, emissions_matrix_EV, emissions_matrix_ICE, destinations, instance_solution,training_metrics, opt_timings):
    res_filename = 'output/'+instance_name + '_Methods_CL_results.xlsx'
    
    # Get distance travelled and total first-mile emissions
    FM_dist_emm_df = get_FM_distance_and_emissions(fm_Sol_dfs, num_vehicles_per_FM, distance_matrix, emissions_matrix_ICE)
    
    # Get distance travelled and total last-mile emissions
    dist_emm_df = get_LM_distance_and_emissions(lm_Sol_dfs, distance_matrix, emissions_matrix_EV)
    
    # Get package assignments to Lockers    
    locker_assignments = get_locker_assignments(lamda_sol_cl)
    locker_ass_df = pd.DataFrame([(k, v) for k, vals in locker_assignments.items() for v in vals], columns=['Locker node', 'Package ID'])
    
    # Get package arrival times
    times_at_dest_df = get_times_at_destinations(lm_Sol_dfs, destinations)
    
    # Get follower objectives
    foll_objs_df = compute_follower_objs(fm_Sol_dfs, lm_Sol_dfs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, time_violation_penalty)
    
    
    # Instance solution
    inst_sol = training_metrics + opt_timings + [instance_solution] + [sum(opt_timings)]
    inst_cols=['FM train time (s)', 'FM R-sq', 'LM train time (s)', 'LM R-sq', 
           'CL Opt time (s)', 'Total FM solve time (s)', 'Total LM solve time (s)','Instance solution', 'Total Opt time (s)']
    sol_df = pd.DataFrame([inst_sol], columns=inst_cols)
    
    print('instance solution =',instance_solution)
    print('total opt time =',sum(opt_timings))
    
    # Write all to file
    with pd.ExcelWriter(res_filename) as writer:  
        sol_df.to_excel(writer, sheet_name='CL Instance Solution', index=False)
        FM_dist_emm_df.to_excel(writer, sheet_name='FM Distance and Emissions')
        dist_emm_df.to_excel(writer, sheet_name='LM Distance and Emissions')
        locker_ass_df.to_excel(writer, sheet_name='Assignments to Lockers', index=False)
        times_at_dest_df.to_excel(writer, sheet_name='Package Arrival Times', index=False)
        foll_objs_df.to_excel(writer, sheet_name='Follower Objectives', index=False)
    return

def solve_param_first_mile_followers_static_single(lamda_sol_cl, Pf, fm_f_nodes, fm_f_arcs, fm_depots, cost_per_km_for_FM, 
                                  num_vehicles_per_FM, selected_locker_nodes, distance_matrix, travel_time_matrix,
                                  bigM_matrix, earliest, latest, mip_time_limit, mip_emphasis):
    Objs = []
    BBs = []
    Gaps = []
    Times = []
    Sol_dfs = []

    for f in range(len(fm_depots)):  
        obj_val, best_bound, mip_gap, solve_time, sol_df = first_mile_follower_static_single(lamda_sol_cl, Pf[f], fm_f_nodes[f], fm_f_arcs[f], fm_depots[f], cost_per_km_for_FM[f], 
                                  num_vehicles_per_FM[f], selected_locker_nodes, distance_matrix, travel_time_matrix,
                                  bigM_matrix, earliest, latest, mip_time_limit, mip_emphasis)
        
        Objs.append(obj_val) 
        BBs.append(best_bound)
        Gaps.append(mip_gap)
        Times.append(solve_time)
        Sol_dfs.append(sol_df)
        
    return Objs, BBs, Gaps, Times, Sol_dfs

def solve_param_last_mile_followers_static_single(y_sol_cl, lamda_sol_cl, dsp_d_nodes, dsp_d_arcs, dsp_depots,
                                              cost_per_km_for_DSP, package_ids, destinations, selected_locker_nodes,
                                               distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, 
                                              time_violation_penalty, mip_time_limit, mip_emphasis):
    Objs = []
    BBs = []
    Gaps = []
    Times = []
    Sol_dfs = []

    for d in range(len(dsp_depots)):  
        print('Solving for LM:', d)
        obj_val, best_bound, mip_gap, solve_time, sol_df = last_mile_follower_static_single(y_sol_cl[:, d], lamda_sol_cl, dsp_d_nodes[d], dsp_d_arcs[d], dsp_depots[d],
                                              cost_per_km_for_DSP[d], package_ids, destinations, selected_locker_nodes,
                                               distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, 
                                              time_violation_penalty, mip_time_limit, mip_emphasis)
        
        Objs.append(obj_val) 
        BBs.append(best_bound)
        Gaps.append(mip_gap)
        Times.append(solve_time)
        Sol_dfs.append(sol_df)
        
    return Objs, BBs, Gaps, Times, Sol_dfs

def first_mile_follower_static_single(lamda, Pf, V1f, A1f, fol_depot, cost_per_km, num_vehicles_for_follower, locker_nodes, 
                        distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest, mip_time_limit, mip_emphasis):
    model = gp.Model('First_Mile_Follower_Static_Single')
    model.setParam('OutputFlag', False)
    model.Params.timelimit = mip_time_limit
    model.Params.MIPFocus  = mip_emphasis
    model.Params.threads = 4
    
    # ----- Sets -----
    K = range(num_vehicles_for_follower)
    satellites = locker_nodes   

    # ----- Variables ----- 
    # w_kij = 1 if arc (i,j) is traversed by vehicle k  
    w = {(k,i,j): model.addVar(vtype=GRB.BINARY, name='w_%d_%d_%d' % (k,i,j)) for k in K for i in V1f for j in V1f if i!=j}
    # Arrival time of vehicle k at node i - tau_ki
    tau = {(k,i): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='tau_%d_%d' % (k,i)) for k in K for i in V1f}  
    
    # ----- Objective function: Minimize the the travel cost ----- 
    objective = gp.quicksum(cost_per_km*distance_matrix[i][j] * w[k,i,j] for k in K for (i,j) in A1f)
    model.setObjective(objective, sense=GRB.MINIMIZE)

    # ----- Constraints -----             
    # Link assignment and routing variables
    for p in Pf:
        for s in satellites:
            model.addConstr(gp.quicksum(w[k,i,s] for k in K for i in V1f if i!=s) >= lamda[p,s], name='LINKVARS')
    
    # Leave depot
    for k in K:
        model.addConstr(gp.quicksum(w[k,fol_depot,s] for s in satellites) == 1, name="LEAVEDEPOT")

    # Flow conservation
    for i in V1f:
        for k in K:
            model.addConstr(gp.quicksum(w[k,j,i] for j in V1f if i!=j) - gp.quicksum(w[k,i,j] for j in V1f if i!=j) == 0, name="FLOW") 

    # Arrival time
    for (i,j) in A1f:
        if j==fol_depot: continue
        for k in K:
            model.addConstr(tau[k,i] + travel_time_matrix[i][j] <= 
                                 tau[k,j] + (1 - w[k,i,j]) * 2*bigM_matrix[i][j], name="ARRIVAL")

    # Earliest and latest time windows
    for k in K:
        for i in V1f:
            model.addConstr(earliest[i]*gp.quicksum(w[k,i,j] for j in V1f if i!=j) <= tau[k,i], name="EARLYWINDOW")
            model.addConstr(tau[k,i] <= latest[i]*gp.quicksum(w[k,i,j] for j in V1f if i!=j), name="LATEWINDOW")

    # Solve
    model.optimize()
    obj_val = model.objVal
    bb = model.ObjBound # best bound
    gap = model.MIPGap
    stime = model.Runtime
    

    var_names = []
    var_values = []
    for v in model.getVars():
        if v.x != 0:
            var_names.append(v.varName)
            var_values.append(v.x)
    
    sol_df = pd.DataFrame({'name': var_names, 'value': var_values})        
    return obj_val, bb, gap, stime, sol_df

def last_mile_follower_static_single(y, lamda, V2d, A2d, fol_depot, cost_per_km, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty, mip_time_limit, mip_emphasis):
    
    model = gp.Model('Last_Mile_Follower_Static_Single')
    model.setParam('OutputFlag', False)
    model.Params.timelimit = mip_time_limit
    model.Params.MIPFocus  = mip_emphasis
    model.Params.threads = 4
    model.Params.MIPGap = 1e-2
    
    # --- Sets ---
    P = packages 
    satellites = locker_nodes

    # ----- Variables -----             
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(i,j): model.addVar(vtype=GRB.BINARY, name='x_%d_%d' % (i,j)) for i in V2d for j in V2d if i!=j}

    # Arrival time of vehicle k at node i - t_ki
    t = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='t_%d' % (i)) for i in V2d}  
        
    # Variables for earliest and latest times
    alpha_early = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='alphaEarly_%d' % (i)) for i in V2d}
    alpha_late = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='alphaLate_%d' % (i)) for i in V2d}
    
    # Variable z used to linearize the quadratic constraint
    z = {(p,s,j):model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='z_%d_%d_%d' % (p,s,j)) for p in P for s in satellites for j in V2d}
       
    # ----- Objective function: Minimize the travel cost ----- 
    objective = gp.quicksum(cost_per_km*distance_matrix[i][j] * x[i,j] for (i,j) in A2d)
    objective += time_violation_penalty * gp.quicksum(alpha_early[i] for i in V2d)
    objective += time_violation_penalty * gp.quicksum(alpha_late[i] for i in V2d)
    model.setObjective(objective, sense=GRB.MINIMIZE)
    
    # ----- Time Violation Constraints ----- 
    for i in V2d:
        model.addConstr(alpha_early[i] >= earliest[i] * gp.quicksum(x[i,j] for j in V2d if i!=j) - t[i], name="TIMEVIOL1")
        model.addConstr(alpha_late[i] >= t[i] - latest[i] * gp.quicksum(x[i,j] for j in V2d if i!=j), name="TIMEVIOL2")
    
    # ----- Constraints -----            
    # If a vehicle k picks up a package p, it should go to the destination of p
    for p in P:
        model.addConstr(gp.quicksum(x[j,destinations[p]] for j in V2d if j != destinations[p]) == y[p], name="GOTODEST")
    
    # A package p should be picked up from its satellite s   
    for p in P:        
        model.addConstr(gp.quicksum(z[p,s,j] for s in satellites for j in V2d if s!=j) >= y[p], name="PICKUP1")     
    for p in P:
        for s in satellites:
            for j in V2d:
                if j==s: continue        
                model.addConstr(z[p,s,j] >= lamda[p,s] + x[s,j] - 1, name="PICKUP2")
                model.addConstr(z[p,s,j] <= lamda[p,s], name="PICKUP3")
                model.addConstr(z[p,s,j] <= x[s,j], name="PICKUP4")

    # A vehicle shoud go from a depot to a satellite
    model.addConstr(gp.quicksum(x[fol_depot,s] for s in satellites) <= 1, name="DEPOTTOSAT")
    
    # A vehicle may go from a satellite to another node    
    for s in satellites:
        model.addConstr(gp.quicksum(x[s,j] for j in V2d if s!=j) <= 1, name="SATTONODE")
    
    # Flow conservation at all nodes
    for j in V2d:    
        model.addConstr(gp.quicksum(x[i,j] for i in V2d if i!=j) - gp.quicksum(x[j,i] for i in V2d if i!=j) == 0, name="FLOWCONS")
 
    
    # Satellite Time constraints
    for p in P:
        model.addConstr(gp.quicksum(lamda[p,s]*(t[s] + travel_time_matrix[s][destinations[p]]) for s in satellites) <= t[destinations[p]] + (1-y[p])*max(max(bigM_matrix)), name="SAT TIME")
                
    # Arrival time
    for (i,j) in A2d:
        if j==fol_depot: continue
        model.addConstr(t[i] + x[i,j]* travel_time_matrix[i][j] <= t[j] + (1 - x[i,j]) *3*max(max(bigM_matrix)), name="ARRIVETIME")
    
    # Solve
    model.optimize()
    obj_val = model.objVal
    bb = model.ObjBound # best bound
    gap = model.MIPGap
    stime = model.Runtime
    

    var_names = []
    var_values = []
    for v in model.getVars():
        if v.x != 0:
            var_names.append(v.varName)
            var_values.append(v.x)
    
    sol_df = pd.DataFrame({'name': var_names, 'value': var_values})
    
    return obj_val, bb, gap, stime, sol_df

def first_mile_data_generation(feasible_lamdas, max_num_package_destinations, max_locker_nodes_list,
                      ps, fmdeps, fmnods, fmarcs, sellocknods, num_vehicles_per_FM, cost_per_km_for_FM, 
                       distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest, 
                       emissions_matrix_ICE, time_violation_penalty, city_name):
    # Store results in dataframe
    # Create headers
    headers = []
    for p in range(max_num_package_destinations):
        for s in max_locker_nodes_list:
            headers.append('lamda_'+str(p)+'_'+str(s))
    headers.append('FM_objective')
    headers.append('FM_best_bound')
    headers.append('FM_gap')
    headers.append('FM_sol_time')
    headers.append('FM_emissions')
    
    # write headers to file
    output_filename = 'output/'+ city_name + '_FM_network_training_data.csv'
    with open(output_filename, 'w') as file:
        file.write(','.join(headers)+'\n')
        
    # Solve LM follower problem
    mip_time_limit = 300 # seconds
    mip_emphasis = 1     # prioritize feasible solutions
        
    training_outputs = []
    for i in range(len(feasible_lamdas)):
        print('i=',i)
        f = fmdeps[i][0]
        obj, bb, gap, stime, sol_df = first_mile_follower_static_single(feasible_lamdas[i], ps[i], fmnods[i][0], fmarcs[i][0], 
                                          fmdeps[i][0], cost_per_km_for_FM[f], num_vehicles_per_FM[f], sellocknods[i], 
                                          distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest, mip_time_limit, mip_emphasis)
               
        emissions = first_miler_emissions_training(sol_df, emissions_matrix_ICE)
        sol = [obj, bb, gap, stime, emissions]
        training_outputs.append(sol)
        
        # Write lamda_p_s to list
        flam = []
        lams = feasible_lamdas[i][:,max_locker_nodes_list]
        for p in range(max_num_package_destinations):
            for s in range(len(max_locker_nodes_list)):
                flam.append(lams[p,s])
                
        # Add solutions
        flam.append(sol[0])
        flam.append(sol[1])
        flam.append(sol[2])
        flam.append(sol[3])
        flam.append(sol[4])
        
        # Write to file
        with open(output_filename, 'a') as file:
            file.write(','.join(str(v) for v in flam)+'\n')   

    return

def last_mile_data_generation(feasible_ys, feasible_lamdas, max_num_package_destinations, max_num_DSPs, max_locker_nodes_list,
                      ps, dspdeps, dspnods, dsparcs, sellocknods, dests, num_vehicles_per_DSP, cost_per_km_for_DSP, 
                       distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest, 
                       emissions_matrix_EV, time_violation_penalty, city_network_df, city_name):
    # Store results in dataframe
    # Create headers
    headers = []
    for p in range(max_num_package_destinations):
        for e in range(max_num_DSPs):
            headers.append('y_'+str(p)+'_'+str(e))
    for p in range(max_num_package_destinations):
        for s in max_locker_nodes_list:
            headers.append('lamda_'+str(p)+'_'+str(s))
    headers.append('LM_objective')
    headers.append('LM_best_bound')
    headers.append('LM_gap')
    headers.append('LM_sol_time')
    headers.append('LM_emissions')
    
    # write headers to file
    output_filename = 'output/'+ city_name + '_LM_network_training_data.csv'
    with open(output_filename, 'w') as file:
        file.write(','.join(headers)+'\n')
        
    # Solve LM follower problem
    mip_time_limit = 200#300 # seconds
    mip_emphasis = 1     # prioritize feasible solutions
    
    offset = city_network_df[city_network_df['type'].str.contains('LM')].iloc[0]['node']
    
    training_outputs = []
    for i in range(len(feasible_ys)):
        print('i=',i)
        d_offset = dspdeps[i][0] - offset# This d needs to be offset
        obj, bb, gap, stime, sol_df = last_mile_follower_static_single(feasible_ys[i][:, d_offset], feasible_lamdas[i], dspnods[i][0],
                                                                                   dsparcs[i][0], dspdeps[i][0], cost_per_km_for_DSP[d_offset], ps[i],
                                                                                   dests[i], sellocknods[i], distance_matrix, travel_time_matrix, 
                                                                                   earliest, latest, bigM_matrix, time_violation_penalty,
                                                                                   mip_time_limit, mip_emphasis)
        
        emissions = last_miler_emissions_training(sol_df, emissions_matrix_EV)
        sol = [obj, bb, gap, stime, emissions]
        training_outputs.append(sol)
        
        # Write y_p_d to list
        fy_flam = []
        for row in feasible_ys[i]:
            fy_flam.extend(row)
        
        # Write lamda_p_s to list
        lams = feasible_lamdas[i][:,max_locker_nodes_list]
        for p in range(max_num_package_destinations):
            for s in range(len(max_locker_nodes_list)):
                fy_flam.append(lams[p,s])
                
        # Add solutions
        fy_flam.append(sol[0])
        fy_flam.append(sol[1])
        fy_flam.append(sol[2])
        fy_flam.append(sol[3])
        fy_flam.append(sol[4])
        
        # Write to file
        with open(output_filename, 'a') as file:
            file.write(','.join(str(v) for v in fy_flam)+'\n')
    
    return

def train_regression_models(fm_foll_data_aggregated, lm_foll_data_aggregated):
    print("MLP")
    # --- First Mile Agent ---
    fm_timer = pc()
    # Split into features and target
    target_FM = fm_foll_data_aggregated['FM_emissions']
    features_FM = fm_foll_data_aggregated.drop(['FM_objective','FM_best_bound','FM_gap','FM_sol_time','FM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_FM, features_test_FM, target_train_FM, target_test_FM = train_test_split(features_FM, target_FM, test_size=0.8, random_state=42)
    
    # Fit model for first mile agent
    print('Training FM regression model')
    # These parameters were already determined via a grid search
    mlp_regr_firstm = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=100000, random_state=1)
    mlp_regr_firstm.fit(features_FM.values, target_FM.values)    
    test_score_f = mlp_regr_firstm.score(features_test_FM.values, target_test_FM.values)
    print("FM test score:", round(test_score_f,2))
    fm_total_train_time = pc()-fm_timer
                
    # --- Last Mile Agent ---
    lm_timer = pc()
    # Split into features and target
    target_LM = lm_foll_data_aggregated['LM_emissions']
    features_LM = lm_foll_data_aggregated.drop(['LM_objective','LM_best_bound','LM_gap','LM_sol_time','LM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_LM, features_test_LM, target_train_LM, target_test_LM = train_test_split(features_LM, target_LM, test_size=0.8, random_state=42)
    
    # Fit model for last mile agent
    print('Training LM regression models')    
    # These parameters were already determined via a grid search    
    mlp_regr_lastm = MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, random_state=1)
    mlp_regr_lastm.fit(features_LM.values, target_LM.values)
    test_score_l = mlp_regr_lastm.score(features_test_LM.values, target_test_LM.values)
    print("LM test score:", round(test_score_l,2))
    lm_total_train_time = pc()-lm_timer
 
    return mlp_regr_firstm, mlp_regr_lastm, test_score_f, test_score_l, fm_total_train_time, lm_total_train_time

def train_regression_models_grid_search_mlp(fm_foll_data_aggregated, lm_foll_data_aggregated):
    print("MLP")
    # Define the parameter grid to search
    param_grid = {
        'hidden_layer_sizes': [(10,),(20,),(50,),(10,10),(50, 50)]
    }
    
    # --- First Mile Agent ---
    fm_timer = pc()
    # Split into features and target
    target_FM = fm_foll_data_aggregated['FM_emissions']
    features_FM = fm_foll_data_aggregated.drop(['FM_objective','FM_best_bound','FM_gap','FM_sol_time','FM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_FM, features_test_FM, target_train_FM, target_test_FM = train_test_split(features_FM, target_FM, test_size=0.8, random_state=42)
    
    # Fit model for first mile agent
    print('Training FM regression model')
    mlp_fm = MLPRegressor(max_iter=10000, random_state=1)
    # Initialize GridSearchCV
    grid_search_fm = GridSearchCV(mlp_fm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_fm.fit(features_train_FM, target_train_FM)
    # Get the best model
    mlp_regr_firstm = grid_search_fm.best_estimator_
    test_score_f = mlp_regr_firstm.score(features_test_FM, target_test_FM)
    print("FM test score:", round(test_score_f,2))
    fm_total_train_time = pc()-fm_timer
                
    # --- Last Mile Agent ---
    lm_timer = pc()
    # Split into features and target
    target_LM = lm_foll_data_aggregated['LM_emissions']
    features_LM = lm_foll_data_aggregated.drop(['LM_objective','LM_best_bound','LM_gap','LM_sol_time','LM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_LM, features_test_LM, target_train_LM, target_test_LM = train_test_split(features_LM, target_LM, test_size=0.8, random_state=42)
    
    # Fit model for last mile agent
    print('Training LM regression models')    
    mlp_lm = MLPRegressor(max_iter=10000, random_state=1)
    # Initialize GridSearchCV
    grid_search_lm = GridSearchCV(mlp_lm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_lm.fit(features_train_LM, target_train_LM)
    # Get the best model
    mlp_regr_lastm = grid_search_lm.best_estimator_
    test_score_l = mlp_regr_lastm.score(features_test_LM, target_test_LM)
    print("LM test score:", round(test_score_l,2))
    lm_total_train_time = pc()-lm_timer
 
    return mlp_regr_firstm, mlp_regr_lastm, test_score_f, test_score_l, fm_total_train_time, lm_total_train_time


def train_regression_models_grid_search_cart(fm_foll_data_aggregated, lm_foll_data_aggregated):
    print("CART")
    # Define the parameter grid to search
    param_grid = {"max_depth": [3,4,5,6,7,8,9,10],
            'min_samples_leaf': [0.02, 0.04, 0.06],
            "max_features": [0.4, 0.6, 0.8, 1.0]}
    
    # --- First Mile Agent ---
    fm_timer = pc()
    # Split into features and target
    target_FM = fm_foll_data_aggregated['FM_emissions']
    features_FM = fm_foll_data_aggregated.drop(['FM_objective','FM_best_bound','FM_gap','FM_sol_time','FM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_FM, features_test_FM, target_train_FM, target_test_FM = train_test_split(features_FM, target_FM, test_size=0.8, random_state=42)
    
    # Fit model for first mile agent
    print('Training FM regression model')
    mlp_fm = DecisionTreeRegressor(random_state=1)
    # Initialize GridSearchCV
    grid_search_fm = GridSearchCV(mlp_fm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_fm.fit(features_train_FM, target_train_FM)
    # Get the best model
    mlp_regr_firstm = grid_search_fm.best_estimator_
    test_score_f = mlp_regr_firstm.score(features_test_FM, target_test_FM)
    print("FM test score:", round(test_score_f,2))
    fm_total_train_time = pc()-fm_timer
                
    # --- Last Mile Agent ---
    lm_timer = pc()
    # Split into features and target
    target_LM = lm_foll_data_aggregated['LM_emissions']
    features_LM = lm_foll_data_aggregated.drop(['LM_objective','LM_best_bound','LM_gap','LM_sol_time','LM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_LM, features_test_LM, target_train_LM, target_test_LM = train_test_split(features_LM, target_LM, test_size=0.8, random_state=42)
    
    # Fit model for last mile agent
    print('Training LM regression models')    
    mlp_lm = DecisionTreeRegressor(random_state=1)
    # Initialize GridSearchCV
    grid_search_lm = GridSearchCV(mlp_lm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_lm.fit(features_train_LM, target_train_LM)
    # Get the best model
    mlp_regr_lastm = grid_search_lm.best_estimator_
    test_score_l = mlp_regr_lastm.score(features_test_LM, target_test_LM)
    print("LM test score:", round(test_score_l,2))
    lm_total_train_time = pc()-lm_timer
 
    return mlp_regr_firstm, mlp_regr_lastm, test_score_f, test_score_l, fm_total_train_time, lm_total_train_time

def train_regression_models_grid_search_gbm(fm_foll_data_aggregated, lm_foll_data_aggregated):
    print("GBM")
    # Define the parameter grid to search
    param_grid = {
            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "max_depth": [2,3,4,5],
            "n_estimators": [5]
        }
    
    # --- First Mile Agent ---
    fm_timer = pc()
    # Split into features and target
    target_FM = fm_foll_data_aggregated['FM_emissions']
    features_FM = fm_foll_data_aggregated.drop(['FM_objective','FM_best_bound','FM_gap','FM_sol_time','FM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_FM, features_test_FM, target_train_FM, target_test_FM = train_test_split(features_FM, target_FM, test_size=0.8, random_state=42)
    
    # Fit model for first mile agent
    print('Training FM regression model')
    mlp_fm = GradientBoostingRegressor(random_state=1)
    # Initialize GridSearchCV
    grid_search_fm = GridSearchCV(mlp_fm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_fm.fit(features_train_FM, target_train_FM)
    # Get the best model
    mlp_regr_firstm = grid_search_fm.best_estimator_
    test_score_f = mlp_regr_firstm.score(features_test_FM, target_test_FM)
    print("FM test score:", round(test_score_f,2))
    fm_total_train_time = pc()-fm_timer
                
    # --- Last Mile Agent ---
    lm_timer = pc()
    # Split into features and target
    target_LM = lm_foll_data_aggregated['LM_emissions']
    features_LM = lm_foll_data_aggregated.drop(['LM_objective','LM_best_bound','LM_gap','LM_sol_time','LM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_LM, features_test_LM, target_train_LM, target_test_LM = train_test_split(features_LM, target_LM, test_size=0.8, random_state=42)
    
    # Fit model for last mile agent
    print('Training LM regression models')    
    mlp_lm = GradientBoostingRegressor(random_state=1)
    # Initialize GridSearchCV
    grid_search_lm = GridSearchCV(mlp_lm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_lm.fit(features_train_LM, target_train_LM)
    # Get the best model
    mlp_regr_lastm = grid_search_lm.best_estimator_
    test_score_l = mlp_regr_lastm.score(features_test_LM, target_test_LM)
    print("LM test score:", round(test_score_l,2))
    lm_total_train_time = pc()-lm_timer
 
    return mlp_regr_firstm, mlp_regr_lastm, test_score_f, test_score_l, fm_total_train_time, lm_total_train_time

def train_regression_models_grid_search_rf(fm_foll_data_aggregated, lm_foll_data_aggregated):
    print("RF")
    # Define the parameter grid to search
    param_grid = {'n_estimators': [10,25],
                'max_features': [1.0],
                'max_depth' : [2,3,4]
            }

    
    # --- First Mile Agent ---
    fm_timer = pc()
    # Split into features and target
    target_FM = fm_foll_data_aggregated['FM_emissions']
    features_FM = fm_foll_data_aggregated.drop(['FM_objective','FM_best_bound','FM_gap','FM_sol_time','FM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_FM, features_test_FM, target_train_FM, target_test_FM = train_test_split(features_FM, target_FM, test_size=0.8, random_state=42)
    
    # Fit model for first mile agent
    print('Training FM regression model')
    mlp_fm = RandomForestRegressor(random_state=1)
    # Initialize GridSearchCV
    grid_search_fm = GridSearchCV(mlp_fm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_fm.fit(features_train_FM, target_train_FM)
    # Get the best model
    mlp_regr_firstm = grid_search_fm.best_estimator_
    test_score_f = mlp_regr_firstm.score(features_test_FM, target_test_FM)
    print("FM test score:", round(test_score_f,2))
    fm_total_train_time = pc()-fm_timer
                
    # --- Last Mile Agent ---
    lm_timer = pc()
    # Split into features and target
    target_LM = lm_foll_data_aggregated['LM_emissions']
    features_LM = lm_foll_data_aggregated.drop(['LM_objective','LM_best_bound','LM_gap','LM_sol_time','LM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_LM, features_test_LM, target_train_LM, target_test_LM = train_test_split(features_LM, target_LM, test_size=0.8, random_state=42)
    
    # Fit model for last mile agent
    print('Training LM regression models')    
    mlp_lm = RandomForestRegressor(random_state=1)
    # Initialize GridSearchCV
    grid_search_lm = GridSearchCV(mlp_lm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_lm.fit(features_train_LM, target_train_LM)
    # Get the best model
    mlp_regr_lastm = grid_search_lm.best_estimator_
    test_score_l = mlp_regr_lastm.score(features_test_LM, target_test_LM)
    print("LM test score:", round(test_score_l,2))
    lm_total_train_time = pc()-lm_timer
 
    return mlp_regr_firstm, mlp_regr_lastm, test_score_f, test_score_l, fm_total_train_time, lm_total_train_time



def train_regression_models_grid_search_lr(fm_foll_data_aggregated, lm_foll_data_aggregated):
    print("Linear Regression with Ridge Regularization")
    # Define the parameter grid to search
    param_grid = {'alpha': [0.01, 0.1, 1, 10]}

    # --- First Mile Agent ---
    fm_timer = pc()
    # Split into features and target
    target_FM = fm_foll_data_aggregated['FM_emissions']
    features_FM = fm_foll_data_aggregated.drop(['FM_objective', 'FM_best_bound', 'FM_gap', 'FM_sol_time', 'FM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_FM, features_test_FM, target_train_FM, target_test_FM = train_test_split(features_FM, target_FM, test_size=0.8, random_state=42)
    
    # Fit model for first mile agent
    print('Training FM regression model')
    ridge_fm = Ridge()
    # Initialize GridSearchCV
    grid_search_fm = GridSearchCV(ridge_fm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_fm.fit(features_train_FM, target_train_FM)
    # Get the best model
    best_regr_firstm = grid_search_fm.best_estimator_
    test_score_f = best_regr_firstm.score(features_test_FM, target_test_FM)
    print("FM test score:", round(test_score_f, 2))
    fm_total_train_time = pc() - fm_timer

    # --- Last Mile Agent ---
    lm_timer = pc()
    # Split into features and target
    target_LM = lm_foll_data_aggregated['LM_emissions']
    features_LM = lm_foll_data_aggregated.drop(['LM_objective', 'LM_best_bound', 'LM_gap', 'LM_sol_time', 'LM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_LM, features_test_LM, target_train_LM, target_test_LM = train_test_split(features_LM, target_LM, test_size=0.8, random_state=42)
    
    # Fit model for last mile agent
    print('Training LM regression model')    
    ridge_lm = Ridge()
    # Initialize GridSearchCV
    grid_search_lm = GridSearchCV(ridge_lm, param_grid, cv=5, scoring='r2')
    # Fit the model
    grid_search_lm.fit(features_train_LM, target_train_LM)
    # Get the best model
    best_regr_lastm = grid_search_lm.best_estimator_
    test_score_l = best_regr_lastm.score(features_test_LM, target_test_LM)
    print("LM test score:", round(test_score_l, 2))
    lm_total_train_time = pc() - lm_timer

    return best_regr_firstm, best_regr_lastm, test_score_f, test_score_l, fm_total_train_time, lm_total_train_time

