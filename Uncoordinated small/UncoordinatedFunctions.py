# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:42:46 2024

@author: ade.fajemisin
"""

import pickle
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import itertools
import random
import networkx as nx
from geopy.distance import geodesic
import gurobipy as gp
from gurobipy import GRB
from time import perf_counter as pc

def run_instance(city_name, problem_instance):
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
    selected_locker_capacities = [max_locker_capacities[i] for i in selected_locker_nodes]
    package_ids, Pf, destinations, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs,city_sub_instance = get_problem_instance_info(city_instance_df, problem_config, locations_and_windows, selected_locker_nodes)
    city_sub_instance.reset_index(drop=True,inplace=True)
  
    
    # --- Optimization ---
    # Get locker preferences
    assigt_timer = pc()
    rankings = get_locker_rankings(city_sub_instance, selected_locker_nodes, destinations)
    # Assign packages to lockers
    locker_assignments = assign_packages_to_lockers(package_ids, rankings, selected_locker_capacities, selected_locker_nodes)
    lamda = generate_lamda(locker_assignments, max_num_packages, max_num_nodes)
    # Assign packages to LM DSPs
    # we assume that each DSP has enough capacity. 
    current_num_DSPs = len(dsp_d_nodes)
    each_dsp_cap = np.ceil((1.8*len(package_ids)/current_num_DSPs)) #+ (len(package_ids)/10)
    dsp_capacities = []
    for i in range(current_num_DSPs):
        dsp_capacities.append(each_dsp_cap)        
    y_dsp_rankings = get_DSP_rankings(city_sub_instance, dsp_depots, destinations)    
    y_small= assign_packages_to_DSPs(package_ids, y_dsp_rankings, dsp_capacities)
    y = np.zeros((max_num_packages,max_num_lockers))
    y[0:len(package_ids), 0:current_num_DSPs] = y_small
    assign_total_time = pc() - assigt_timer

    
    # --- Solve parameterized follower problems ---
    mip_time_limit = 300
    mip_emphasis = 0 # default: balance feasibility and optimality  
    
    fm_follower_timer = pc()
    fm_Objs, fm_Sol_dfs = solve_param_first_mile_followers_static_single(lamda, Pf, fm_f_nodes, fm_f_arcs, fm_depots, cost_per_km_for_FM, 
                                      num_vehicles_per_FM, selected_locker_nodes, distance_matrix, travel_time_matrix,
                                      bigM_matrix, earliest, latest)
    total_FM_foll_time  = pc() - fm_follower_timer

    lm_follower_timer = pc()
    lm_Objs, lm_Sol_dfs = solve_param_last_mile_followers_static_single(y, lamda, dsp_d_nodes, dsp_d_arcs, dsp_depots,
                                                  cost_per_km_for_DSP, package_ids, destinations, selected_locker_nodes,
                                                   distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, 
                                                  time_violation_penalty)
    total_LM_foll_time  = pc() - lm_follower_timer
    
    opt_timings = []
    opt_timings.append(assign_total_time)
    opt_timings.append(total_FM_foll_time)
    opt_timings.append(total_LM_foll_time)
    
    # FM stuff approximations start
    pack_FM_relation_df = city_sub_instance[['package_id','first_miler_of_origin']].dropna()
    grouped = pack_FM_relation_df.groupby('first_miler_of_origin')['package_id']
    package_FM_lists = {name: group.tolist() for name, group in grouped}
    fm_satellites_visited = find_satellites_visited(package_FM_lists, lamda)
    for key in fm_satellites_visited:
        if len(fm_satellites_visited[key]) == 1 and fm_satellites_visited[key][0] == selected_locker_nodes[0]:
            fm_satellites_visited[key].append(selected_locker_nodes[0]+1)   
    fm_all_nodes_visited = generate_fm_nodes_visited(fm_satellites_visited, fm_depots)
    all_possible_paths = generate_paths_append_first(fm_all_nodes_visited)
    route_dataframes = create_route_dataframes(all_possible_paths)
    fm_all_distances = []
    for i in range(len(route_dataframes)):
        i_distances = []
        for j in range(len(route_dataframes[i])):
            w_sol = route_dataframes[i][j]
            distances = 0
            for k in range(len(w_sol)):
                i_index = w_sol.loc[k]['i']
                j_index = w_sol.loc[k]['j']
                distances += distance_matrix[i_index][j_index]
            i_distances.append(distances)
        fm_all_distances.append(i_distances)

    indexes_of_shortest = [lst.index(max(lst)) for lst in fm_all_distances]
    fm_Sol_dfs = []
    for m in range(len(route_dataframes)):
        fm_Sol_dfs.append(route_dataframes[m][indexes_of_shortest[m]])
    # FM stuff end

    # Compute real solution
    instance_solution = compute_upper_bound_greedy(fm_Sol_dfs, lm_Sol_dfs, emissions_matrix_ICE, emissions_matrix_EV)
    
    # Pickle lamda, y, FM_sols, LM_sols
    with open('output/lamda_sol_final_'+instance_name+'_greedy.pickle', 'wb') as handle:
        pickle.dump(lamda, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('output/y_sol_final_'+instance_name+'_greedy.pickle', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('output/first_m_sol_dfs_final_'+instance_name+'_greedy.pickle', 'wb') as handle:
        pickle.dump(fm_Sol_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('output/last_m_sol_dfs_final_'+instance_name+'_greedy.pickle', 'wb') as handle:
        pickle.dump(lm_Sol_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Write solution to file
    write_instance_results_to_file_greedy(instance_name, lamda, fm_Sol_dfs, lm_Sol_dfs, distance_matrix, 
                                   emissions_matrix_EV, emissions_matrix_ICE, destinations, instance_solution, opt_timings,time_violation_penalty)
    
   
    print('Instance solved.\n')
    return 

def get_problem_instance_info(city_instance_df, problem_config, locations_and_windows, selected_locker_nodes):
    loc_latitudes = []
    loc_longitudes = []
    for i in range(len(locations_and_windows)):
        loc_latitudes.append(locations_and_windows[i]['latitude'])
        loc_longitudes.append(locations_and_windows[i]['longitude'])
    city_instance_df['latitude'] = loc_latitudes
    city_instance_df['longitude'] = loc_longitudes
        
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

    package_ids = list(city_sub_instance.loc[city_sub_instance['type']=='package']['package_id'])
    package_ids = [int(x) for x in package_ids]
    
    # Get which packages come from which first-miler
    filtered_city = city_sub_instance.dropna(subset=['package_id'])
    orig_dict = filtered_city[['package_id', 'first_miler_of_origin']].to_dict(orient='list')
    fm_origins = get_firstmiler_package_origins(orig_dict)
    all_origins_bounds = []
    for i in range(len(fm_origins)):
        all_origins_bounds.append(len(fm_origins[i]))
    Pf = create_bounds(all_origins_bounds)
    
    # Get package destinations
    destinations = get_destinations(city_sub_instance)
    
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
    
    return package_ids, Pf, destinations, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs,city_sub_instance

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

def get_bigM_matrix_single(instance_tt_travel_times, numNodes):    
    # Put in matrix from
    max_travel_time_per_arc = []
    for i in range(len(instance_tt_travel_times)):
        max_travel_time_per_arc.append(2*(instance_tt_travel_times[i]['travel_times']))
    bigM_matrix = [max_travel_time_per_arc[i:i + numNodes] for i in range(0, len(max_travel_time_per_arc), numNodes)]
    return bigM_matrix

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

def compute_upper_bound_greedy(first_m_sol_dfs, last_m_sol_dfs, emissions_matrix_ICE, emissions_matrix_EV):
    # Compute first mile emissions
    fm_emissions = []
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = first_m_sol_dfs[i]
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

def get_distance_and_emissions(lm_Sol_dfs, distance_matrix, emissions_matrix_EV):
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

def find_satellites_visited(package_FM_lists, lamda):
    w = {key: [] for key in package_FM_lists.keys()}
    for key, package_list in package_FM_lists.items():
        indices_set = set()
        for package_id in package_list:
            row = int(package_id)
            indices_set.update(np.where(lamda[row] != 0)[0])
        w[key] = list(indices_set)
    return w

def generate_fm_nodes_visited(fm_satellites_visited, fm_depots):
    output = {}
    depot_index = 0
    for key in fm_satellites_visited.keys():
        depot = fm_depots[depot_index]
        output[key] = [depot] + fm_satellites_visited[key]
        depot_index += 1
        if depot_index >= len(fm_depots):
            depot_index = 0
    return output

def generate_paths_append_first(fm_all_nodes_visited):
    all_paths = {}
    for key, nodes in fm_all_nodes_visited.items():
        first_node = nodes[0]
        remaining_nodes = nodes[1:]
        paths = list(itertools.permutations(remaining_nodes))
        all_paths[key] = [tuple([first_node] + list(path) + [first_node]) for path in paths]
    return all_paths

def create_route_dataframes(all_paths):
    route_dataframes = {}
    for key, paths in all_paths.items():
        dataframes = []
        for path in paths:
            df = pd.DataFrame({'i': path[:-1], 'j': path[1:]})
            dataframes.append(df)
        route_dataframes[key] = dataframes
    return route_dataframes

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

def get_FM_distance_and_emissions_greedy(first_m_sol_dfs, distance_matrix, emissions_matrix_ICE):
    dist_res = []
    emm_res = []
    
    # Compute first mile emissions
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = first_m_sol_dfs[i]
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

def get_FM_follower_objectives(first_m_sol_dfs, distance_matrix):
    follower_objectives = []        
    # Compute follower objectives
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = first_m_sol_dfs[i]#extract_w_static_single(first_m_sol_dfs[i])          
            distances = []
            for j in range(len(w_sol)):
                i_index = w_sol.loc[j]['i']
                j_index = w_sol.loc[j]['j']
                distances.append(distance_matrix[i_index][j_index])     
            follower_objectives.append(sum(distances))  
        else:
            follower_objectives.append(0)  
    return follower_objectives

def get_LM_follower_objectives(last_m_sol_dfs_final, distance_matrix, time_violation_penalty):
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

def compute_follower_objs_first_and_optimal(fm_FINAL, lm_FINAL, distance_matrix, time_violation_penalty):
    fm_objs_optimal = get_FM_follower_objectives(fm_FINAL, distance_matrix)    
    lm_objs_optimal = get_LM_follower_objectives(lm_FINAL, distance_matrix, time_violation_penalty)
    
    objcols = []
    for f in range(len(fm_objs_optimal)):
        objcols.append('FM ' + str(f) + ' Obj optimal')
    for d in range(len(lm_objs_optimal)):
        objcols.append('LM ' + str(d) + ' Obj optimal')
    
    objectives = fm_objs_optimal + lm_objs_optimal
    obj_df = pd.DataFrame([objectives], columns = objcols)   
    return obj_df

def write_instance_results_to_file_greedy(instance_name, lamda_sol_cl, fm_Sol_dfs, lm_Sol_dfs, distance_matrix, 
                                   emissions_matrix_EV, emissions_matrix_ICE, destinations, instance_solution, opt_timings, time_violation_penalty):
    res_filename = 'output/'+instance_name + '_greedy_results.xlsx'
    
    # Get distance travelled and total first-mile emissions
    FM_dist_emm_df = get_FM_distance_and_emissions_greedy(fm_Sol_dfs, distance_matrix, emissions_matrix_ICE)
    
    # Get distance travelled and total last-mile emissions
    dist_emm_df = get_distance_and_emissions(lm_Sol_dfs, distance_matrix, emissions_matrix_EV)
    
    # Get package assignments to Lockers    
    locker_assignments = get_locker_assignments(lamda_sol_cl)
    locker_ass_df = pd.DataFrame([(k, v) for k, vals in locker_assignments.items() for v in vals], columns=['Locker node', 'Package ID'])
    # Get package arrival times
    times_at_dest_df = get_times_at_destinations(lm_Sol_dfs, destinations)
    
    # Instance solution
    inst_sol = opt_timings + [instance_solution] + [sum(opt_timings)]
    inst_cols=['Assignment time (s)', 'Total FM solve time (s)', 'Total LM solve time (s)','Instance solution','Total Opt time (s)']
    sol_df = pd.DataFrame([inst_sol], columns=inst_cols)
    
    # Get follower objectives
    foll_objs_df = compute_follower_objs_first_and_optimal(fm_Sol_dfs, lm_Sol_dfs, distance_matrix, time_violation_penalty)
    
    # Write all to file
    with pd.ExcelWriter(res_filename) as writer:  
        sol_df.to_excel(writer, sheet_name='Greedy Instance Solution', index=False)
        FM_dist_emm_df.to_excel(writer, sheet_name='FM Distance and Emissions')
        dist_emm_df.to_excel(writer, sheet_name='LM Distance and Emissions')
        locker_ass_df.to_excel(writer, sheet_name='Assignments to Lockers', index=False)
        times_at_dest_df.to_excel(writer, sheet_name='Package Arrival Times', index=False)
        foll_objs_df.to_excel(writer, sheet_name='Follower Objectives', index=False)
    return

def compute_distance(all_nodes_df, locker_num, deliv_loc):
    # 1. Get closest lockers to delivery locations
    # This is using basic distance. We can improve on this later...
    origin = (all_nodes_df.loc[locker_num]["latitude"], all_nodes_df.loc[locker_num]["longitude"])
    destination = (all_nodes_df.loc[deliv_loc]["latitude"], all_nodes_df.loc[deliv_loc]["longitude"])
    distance = round(geodesic(origin, destination).meters, 4)    
    return distance

def get_locker_rankings(all_nodes_df, locker_nodes, delivery_nodes):
    rankings = []
    for i in delivery_nodes:
        distances = {}
        for j in locker_nodes:
            dist = compute_distance(all_nodes_df, j, i)
            distances[j] = dist
        distances = sorted(distances, key = distances.get)
        rankings.append(distances)    
    return rankings

def get_DSP_rankings(all_nodes_df, dsp_depots, destinations):
    rankings = []
    for i in destinations:
        distances = {}
        for j in dsp_depots:
            dist = compute_distance(all_nodes_df, j, i)
            distances[j] = dist
        distances = sorted(distances, key = distances.get)
        rankings.append(distances)    
    return rankings

def assign_packages_to_lockers(items, rankings, original_capacities, locker_nodes):
    capacities = [i for i in original_capacities]
    # Create a dictionary to store items for each list
    lists_dict = {i: [] for i in range(len(capacities))}    
    # Combine items, rankings, and capacities into a list of tuples
    item_data = list(zip(items, rankings))    
    # Sort items based on rankings in descending order
    sorted_items = sorted(item_data, key=lambda x: x[1])
    # Iterate through sorted items and assign them to lists based on rankings and capacities
    for item, ranking in sorted_items:
        assigned = False
        # Sort lists based on remaining capacity in ascending order to prioritize lists with more space
        for i in sorted(lists_dict, key=lambda x: capacities[x]):
            if capacities[i] > 0:
                lists_dict[i].append(item)
                capacities[i] -= 1
                assigned = True
                break
        if not assigned:
            # If all lists are full, assign the item to the list with the highest ranking
            max_ranking_list = max(lists_dict, key=lambda x: item_data[x][1])
            lists_dict[max_ranking_list].append(item)
            capacities[max_ranking_list] -= 1    
    # Renaming keys
    lists_dict = dict(zip(locker_nodes, lists_dict.values()))    
    return lists_dict

def generate_lamda(assignments, num_packages, num_nodes):
    lamda_ps = np.zeros((num_packages, num_nodes))
    for key, indices in assignments.items():
        lamda_ps[indices, key] = 1 
    return lamda_ps

def generate_assignment_array(input_dict):
    max_key = max(input_dict.keys())
    max_value = max(max(input_dict.values(), key=lambda x: max(x)))
    output_array = np.zeros((max_value + 1, max_key + 1))
    for key, indices in input_dict.items():
        for index in indices:
            output_array[index, key] = 1
    return output_array

def assign_packages_to_DSPs(items, rankings, dsp_capacities):
    capacities = [i for i in dsp_capacities]
    # Create a dictionary to store items for each list
    lists_dict = {i: [] for i in range(len(capacities))}    
    # Combine items, rankings, and capacities into a list of tuples
    item_data = list(zip(items, rankings))    
    # Sort items based on rankings in descending order
    sorted_items = sorted(item_data, key=lambda x: x[1])
    # Iterate through sorted items and assign them to lists based on rankings and capacities
    for item, ranking in sorted_items:
        assigned = False
        # Sort lists based on remaining capacity in ascending order to prioritize lists with more space
        for i in sorted(lists_dict, key=lambda x: capacities[x]):
            if capacities[i] > 0:
                lists_dict[i].append(item)
                capacities[i] -= 1
                assigned = True
                break
        if not assigned:
            # If all lists are full, assign the item to the list with the highest ranking
            max_ranking_list = max(lists_dict, key=lambda x: item_data[x][1])
            lists_dict[max_ranking_list].append(item)
            capacities[max_ranking_list] -= 1
    
    dsp_ids = list(range(len(dsp_capacities)))
    # Renaming keys
    lists_dict = dict(zip(dsp_ids, lists_dict.values()))     
    
    # Iterate over the dictionary to find keys with empty lists
    for key, value in lists_dict.items():
        if not value:  # Check if the list is empty
            # Get a list of keys that have non-empty lists
            non_empty_keys = [k for k in lists_dict if lists_dict[k]]        
            # Select a random key from the non-empty keys
            donor_key = random.choice(non_empty_keys)        
            # Move an item from the donor key's list to the current empty list
            item_to_move = lists_dict[donor_key].pop()
            lists_dict[key].append(item_to_move)        
            # Break after the first transfer, since only one move is needed
            break
    assignments = generate_assignment_array(lists_dict)    
    return assignments   

def solve_param_first_mile_followers_static_single(lamda_sol_cl, Pf, fm_f_nodes, fm_f_arcs, fm_depots, cost_per_km_for_FM, 
                                  num_vehicles_per_FM, selected_locker_nodes, distance_matrix, travel_time_matrix,
                                  bigM_matrix, earliest, latest):
    Objs = []
    Sol_dfs = []

    for f in range(len(fm_depots)):  
        obj_val, sol_df = first_mile_follower_static_single(lamda_sol_cl, Pf[f], fm_f_nodes[f], fm_f_arcs[f], fm_depots[f], cost_per_km_for_FM[f], 
                                  num_vehicles_per_FM[f], selected_locker_nodes, distance_matrix, travel_time_matrix,
                                  bigM_matrix, earliest, latest)
        
        Objs.append(obj_val) 
        Sol_dfs.append(sol_df)
        
    return Objs, Sol_dfs

def first_mile_follower_static_single(lamda, Pf, V1f, A1f, fol_depot, cost_per_km, num_vehicles_for_follower, locker_nodes, 
                        distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest):
    model = gp.Model('First_Mile_Follower_Static_Single')
    model.setParam('OutputFlag', False)
    model.Params.timelimit = 300
    model.Params.threads = 4
    model.Params.MIPGap = 1e-2
    
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

    var_names = []
    var_values = []
    for v in model.getVars():
        if v.x != 0:
            var_names.append(v.varName)
            var_values.append(v.x)
    
    sol_df = pd.DataFrame({'name': var_names, 'value': var_values})        
    return obj_val, sol_df

def solve_param_last_mile_followers_static_single(y_sol_cl, lamda_sol_cl, dsp_d_nodes, dsp_d_arcs, dsp_depots,
                                              cost_per_km_for_DSP, package_ids, destinations, selected_locker_nodes,
                                               distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, 
                                              time_violation_penalty):
    Objs = []
    Sol_dfs = []

    for d in range(len(dsp_depots)):  
        print('Solving for LM:', d)
        obj_val, sol_df = last_mile_follower_static_single(y_sol_cl[:, d], lamda_sol_cl, dsp_d_nodes[d], dsp_d_arcs[d], dsp_depots[d],
                                              cost_per_km_for_DSP[d], package_ids, destinations, selected_locker_nodes,
                                               distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, 
                                              time_violation_penalty)
        
        Objs.append(obj_val) 
        Sol_dfs.append(sol_df)
        
    return Objs, Sol_dfs

def last_mile_follower_static_single(y, lamda, V2d, A2d, fol_depot, cost_per_km, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty):
    
    model = gp.Model('Last_Mile_Follower_Static_Single')
    model.setParam('OutputFlag', False)
    model.Params.timelimit = 300
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

    var_names = []
    var_values = []
    for v in model.getVars():
        if v.x != 0:
            var_names.append(v.varName)
            var_values.append(v.x)
    
    sol_df = pd.DataFrame({'name': var_names, 'value': var_values})
    
    return obj_val, sol_df

