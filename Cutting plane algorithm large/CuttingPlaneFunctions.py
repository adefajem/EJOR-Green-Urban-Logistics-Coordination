# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:42:46 2024

@author: ade.fajemisin
"""

import pickle
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from time import perf_counter as pc
from docplex.mp.model import Model


def run_instance(city_name, problem_instance):
    # --- Read in complete network ---
    distance_and_times_filename = 'instances/'+city_name+'_distances_times_2_5_5_200.pickle'
    locations_and_windows_filename = 'instances/'+city_name+'_locations_windows_2_5_5_200.pickle'
    problem_info_filename = 'instances/'+city_name+'_2_5_5_200.xlsx'
    
    with open(distance_and_times_filename, 'rb') as handle:
        distance_and_times = pickle.load(handle)
    with open(locations_and_windows_filename, 'rb') as handle:
        locations_and_windows = pickle.load(handle)
        
    # Get distances and times for whole network
    distances, travel_times = get_distances_and_times_single(distance_and_times)     
    max_num_FMs = 2
    max_num_DSPs = 5
    max_num_lockers = 5
    max_num_packages = 200
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
    last_milers_df = pd.read_excel(problem_info_filename, sheet_name='last_milers')
    num_vehicles_per_DSP = list(last_milers_df['num_vehicles'])
    cost_per_km_for_DSP = list(last_milers_df['cost_per_km_for_DSP'])
    cost_per_DSP_vehicle = 100

    # Satellite nodes
    max_locker_nodes = list(city_instance_df.loc[city_instance_df['type']=='locker']['node'])
    # Total satellite capacities
    max_locker_capacities = {}
    max_locker_capacities[7] = 50
    max_locker_capacities[8] = 50
    max_locker_capacities[9] = 40
    max_locker_capacities[10] = 40
    max_locker_capacities[11] = 20
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
    num_FirstMilers = sum(fm_config)
    num_DSPs = sum(lm_config)
    print('Solving', instance_name)
    selected_locker_nodes = [a*b for a,b in zip(locker_config, list(max_locker_capacities.keys()))]
    selected_locker_nodes = [s for s in selected_locker_nodes if s>0]
    package_ids, Pf, destinations, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs = get_problem_instance_info(city_instance_df, problem_config, selected_locker_nodes)
    # All nodes
    all_nodes_first_echelon = fm_depots + selected_locker_nodes
    all_nodes_second_echelon = selected_locker_nodes + dsp_depots + list(destinations.values())
    final_leave_time=23
      
    
    # --- Optimization: Cutting Plane Algorithm --- 
    # Problem runtime time limit (seconds)
    time_limit_per_iteration = 3600
    problem_time_limit = 3600
    try:
        timer = pc()
        bilevel_HPR_emissions_model = HPR_model_emissions_cp(time_limit_per_iteration, package_ids, destinations, selected_locker_nodes, max_locker_capacities, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, all_nodes_first_echelon,
                             all_nodes_second_echelon, emissions_matrix_ICE, emissions_matrix_EV, Pf, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs,
                             travel_time_matrix, bigM_matrix, earliest, latest)


        # Follower & interdiction cuts #  cutting_plane_algorithm_FOLLOWER_AND_INTERDICTION cutting_plane_algorithm_MODIFIED
        hpr_sol_final, lamda_sol_final, w_sol_final, xm_sol_final, y_sol_final, first_m_sol_dfs_final, last_m_sol_dfs_final, first_m_sol_dfs_FIRSTITER, last_m_sol_dfs_FIRSTITER, all_iter_results = cutting_plane_algorithm_FOLLOWER_AND_INTERDICTION_cp(bilevel_HPR_emissions_model, Pf, package_ids, destinations, selected_locker_nodes, max_num_nodes, 
        num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, cost_per_km_for_FM, cost_per_km_for_DSP,cost_per_DSP_vehicle, emissions_matrix_ICE, emissions_matrix_EV, 
                                    distance_matrix, travel_time_matrix, bigM_matrix,  final_leave_time, earliest, latest, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs, 
                                    time_violation_penalty, True, problem_time_limit)


        locker_assignments = get_locker_assignments(lamda_sol_final)
        total_time = pc()-timer

        # --- Write solution to file ---
        # Optimization diagnostics and Instance results:
        write_instance_results_to_file(instance_name, all_iter_results, total_time, y_sol_final, locker_assignments, first_m_sol_dfs_final, last_m_sol_dfs_final, first_m_sol_dfs_FIRSTITER, last_m_sol_dfs_FIRSTITER,
                                                 num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, emissions_matrix_EV, emissions_matrix_ICE, fm_depots, dsp_depots, selected_locker_nodes, package_ids, destinations,
                                                time_violation_penalty)
        print('Instance solved.\n')

    except:
        print('No feasible solution for instance could be found in the allotted timeframe.\n')
              
    return 


def get_problem_instance_info(city_instance_df, problem_config, selected_locker_nodes):
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
    
    return package_ids, Pf, destinations, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs

# Get leave_times, arrive_times, distances and travel_times into 'matrix' form
def get_distances_and_times(distance_and_times):
    distances = []
    travel_times = []
    leave_times = []
    arrive_times = []

    for i in range(len(distance_and_times)):
        a = distance_and_times[i]['from']
        b = distance_and_times[i]['to']

        arc_distances_dict = {'from': int(a), 'to': int(b), 'distances': distance_and_times[i]['times']['distance'].to_list()}
        distances.append(arc_distances_dict)

        arc_travel_times_dict = {'from': int(a), 'to': int(b), 'travel_times': distance_and_times[i]['times']['travel_time'].to_list()}
        travel_times.append(arc_travel_times_dict)

        arc_leave_times_dict = {'from': int(a), 'to': int(b), 'leave_times': distance_and_times[i]['times']['leave_time'].to_list()}
        leave_times.append(arc_leave_times_dict)

        arc_arrive_times_dict = {'from': int(a), 'to': int(b), 'arrive_times': distance_and_times[i]['times']['arrive_time'].to_list()}
        arrive_times.append(arc_arrive_times_dict)

    # Sort so that array goes from 0 to numNodes
    distances = sorted(distances, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    travel_times = sorted(travel_times, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    leave_times = sorted(leave_times, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    arrive_times = sorted(arrive_times, key=lambda k: (k.get('from', 0),k.get('to', 0)))
    
    
    return distances, travel_times, leave_times, arrive_times

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

def get_destinations(df):
    destinations = {}
    for index, row in df.iterrows():
        if row['type'] == 'package':
            destinations[int(row['package_id'])] = int(row['node'])
    return destinations

def get_num_time_periods_matrix(instance_tt_leave_times, numNodes): 
    # Find the number of time periods for each arc
    num_time_periods = []
    for i in range(len(instance_tt_leave_times)):
        num_time_periods.append(len(instance_tt_leave_times[i]))#['travel_times']))

    num_time_periods_matrix = [num_time_periods[i:i + numNodes] for i in range(0, len(num_time_periods), numNodes)]
    
    return num_time_periods_matrix

def get_bigM_matrix(instance_tt_arrive_times, latest, numNodes):    
    # Put in matrix from
    max_arrive_time_per_arc = []
    for i in range(len(instance_tt_arrive_times)):
        max_arrive_time_per_arc.append(max(instance_tt_arrive_times[i]['arrive_times']))

    bigM_matrix = [max_arrive_time_per_arc[i:i + numNodes] for i in range(0, len(max_arrive_time_per_arc), numNodes)]
    
    for i in range(len(bigM_matrix)):
        for j in range(len(bigM_matrix[i])):
            bigM_matrix[i][j] = bigM_matrix[i][j] + latest[i]
    
    return bigM_matrix

def get_bigM_matrix_single(instance_tt_travel_times, numNodes):    
    # Put in matrix from
    max_travel_time_per_arc = []
    for i in range(len(instance_tt_travel_times)):
        max_travel_time_per_arc.append(2*(instance_tt_travel_times[i]['travel_times']))

    bigM_matrix = [max_travel_time_per_arc[i:i + numNodes] for i in range(0, len(max_travel_time_per_arc), numNodes)]
    
#     for i in range(len(bigM_matrix)):
#         for j in range(len(bigM_matrix[i])):
#             bigM_matrix[i][j] = 2*bigM_matrix[i][j]# + latest[i]
    
    return bigM_matrix

def get_distance_matrix(instance_tt_distances, numNodes):   
    # Put in matrix from
    all_distances = []
    for i in range(len(instance_tt_distances)):
        all_distances.append(list(np.array(instance_tt_distances[i]['distances'])/1000)) # convert to kilometers

    distance_matrix = [all_distances[i:i + numNodes] for i in range(0, len(all_distances), numNodes)]
    
    return distance_matrix

def get_distance_matrix_single(instance_tt_distances, numNodes):   
    # Put in matrix from
    all_distances = []
    for i in range(len(instance_tt_distances)):
        all_distances.append(instance_tt_distances[i]['distances']/1000) # convert to kilometers

    distance_matrix = [all_distances[i:i + numNodes] for i in range(0, len(all_distances), numNodes)]
    
    return distance_matrix

def get_travel_time_matrix(instance_tt_travel_times, numNodes):    
    # Put in matrix from
    all_travel_times = []
    for i in range(len(instance_tt_travel_times)):
        all_travel_times.append(instance_tt_travel_times[i]['travel_times'])

    travel_time_matrix = [all_travel_times[i:i + numNodes] for i in range(0, len(all_travel_times), numNodes)]
    
    return travel_time_matrix

def get_travel_time_matrix_single(instance_tt_travel_times, numNodes):    
    # Put in matrix from
    all_travel_times = []
    for i in range(len(instance_tt_travel_times)):
        all_travel_times.append(instance_tt_travel_times[i]['travel_times'])

    travel_time_matrix = [all_travel_times[i:i + numNodes] for i in range(0, len(all_travel_times), numNodes)]
    
    return travel_time_matrix

def get_leave_time_matrix(instance_tt_leave_times, numNodes, final_leave_time):  
    # Put in matrix from
    all_leave_times = []
    all_leave_times_shifted = []
    for i in range(len(instance_tt_leave_times)):
        all_leave_times.append(instance_tt_leave_times[i]['leave_times'])
        
        # shifted so that we can have the upper bound on leave times: b_ijm+1
        shifted = instance_tt_leave_times[i]['leave_times'][1:]
        shifted.append(final_leave_time)
        all_leave_times_shifted.append(shifted)

    leave_time_start = [all_leave_times[i:i + numNodes] for i in range(0, len(all_leave_times), numNodes)]
    leave_time_end = [all_leave_times_shifted[i:i + numNodes] for i in range(0, len(all_leave_times_shifted), numNodes)]
    
    return leave_time_start, leave_time_end
    
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

def compute_emissions_ICE(fm_engine_params, load, distance_matrix, travel_time_matrix, num_time_periods_matrix, num_nodes):    
    
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
            emissions_m = []
            for m in range(num_time_periods_matrix[i][j]): 
                
                if travel_time_matrix[i][j][m]!=0:
                    emission = lamda * (k_e * N_e * V_e * travel_time_matrix[i][j][m] \
                                        + gamma * beta * distance_matrix[i][j][m] * (distance_matrix[i][j][m] / travel_time_matrix[i][j][m])**2 \
                                                   + gamma * alpha * (mu_eng + load) * distance_matrix[i][j][m])                    
                    emissions_m.append(emission*2680)  #conversion to gCO2
                else:
                    emissions_m.append(0)
            
            emission_ij.append(emissions_m)
    
    # Put in matrix form
    E_ijm = [emission_ij[i:i+num_nodes] for i in range(0, len(emission_ij), num_nodes)]
    
    return E_ijm

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
    

def compute_emissions_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_time_periods_matrix, num_nodes):
    emission_ij = []
    
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):  
            emissions_m = []
            for m in range(num_time_periods_matrix[i][j]): 
                if travel_time_matrix[i][j][m]!=0:
                    kwh_consumption = battery_capacity * distance_matrix[i][j][m] / 100 # kilowatt-hours/100 km
                    emission = EVCO2(kwh_consumption, emission_factor, generation_percentage)
                                  
                    emissions_m.append(emission)
                else:
                    emissions_m.append(0)
            
            emission_ij.append(emissions_m)
                    
    # Put in matrix form
    E_ijm = [emission_ij[i:i+num_nodes] for i in range(0, len(emission_ij), num_nodes)]
                    
    return E_ijm

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

def create_bounds(num_vehicles_per_DSP):
    veh_ranges = np.cumsum(num_vehicles_per_DSP)
    bounds = []
    bounds.append(range(veh_ranges[0]))
    for i in range(len(veh_ranges)):
        if i != 0:
            bounds.append(range(veh_ranges[i-1], veh_ranges[i]))

    return bounds

def get_firstmiler_package_origins(data):
    groups = {}
    
    for idx, origin in enumerate(data['first_miler_of_origin']):
        if origin not in groups:
            groups[origin] = []
        groups[origin].append(idx)
    
    return groups


def write_assignments_to_file(y_col, d, assignment_file_path):
    with open(assignment_file_path, 'w') as file:       
        # Find indices where the value is 1
        indices = np.where(y_col == 1)[0]
        
        # Generate the output string
        output_string = "\n".join([f"package_id {index}" for index in indices])

        # Write the output to a file
        with open(assignment_file_path, 'w') as file:
            file.write("Packages assigned to last-miler " + str(d) +":\n")
            file.write(output_string)
    return

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

def write_locker_assignments_to_file(locker_assignments, file_path):
    with open(file_path, 'w') as file:
        for locker, package_ids in locker_assignments.items():
            file.write(f"Packages assigned to locker {locker}:\n")
            for package_id in package_ids:
                file.write(f"package_id {package_id}\n")
            file.write("\n")
    return

def add_dsp_depot_column(df1, df2):
    # Assuming 'd' column is present in df2
    df2['dsp_depot'] = df2['d'].apply(lambda x: df1['LM_id'][0] if x == 0 else df1['LM_id'][1])
    return df2

def get_path(xm_sol_final, d):
    # Create a directed graph from the DataFrame
    graph = nx.from_pandas_edgelist(xm_sol_final, 'i', 'j', create_using=nx.DiGraph())    
    # Perform depth-first search to find the path
    path = nx.dfs_preorder_nodes(graph, source=d)    
    path = list(path)
    
    return list(map(int, path))

def write_dsp_path_to_file(my_list, file_path):
    with open(file_path, 'w') as file:
        file.write(" -> ".join(map(str, my_list)))
    return

def extract_solutions_and_write(xm_sol_final, y, locker_assignments, lmd, dsp_depots):  
    print('Writing solutions to file...')
    
    # Write package assignments to Lockers
    locker_file_path = 'output/Package_assignments_to_lockers.txt'
    write_locker_assignments_to_file(locker_assignments, locker_file_path)
    
    # Write DSP path to file
    xm_sol_final = add_dsp_depot_column(lmd, xm_sol_final)
    for i in dsp_depots:
        xm_d = xm_sol_final[xm_sol_final['dsp_depot'] == i]    
        if len(xm_d) != 0:
            # Get path
            dsp_path = get_path(xm_d, i)
            # Write to file
            dd = lmd.index[lmd['LM_id'] == i].to_list()[0]
            file_path = 'output/last-miler_'+ str(dd) + '_path.txt'
            write_dsp_path_to_file(dsp_path, file_path)

    for d in range(y.shape[1]):
        # Write package assignments to DSPs
        assignment_file_path = 'output/Package_assignments_to_last-miler_'+ str(d) + '.txt'
        write_assignments_to_file(y[:, d], d, assignment_file_path)
        
    print('Done!')

    return

def get_lastmiler_assgt(y_sol_final):
    y_sol_final = np.round(y_sol_final)
    indices = []
    for row in y_sol_final:
        index = next((i for i, x in enumerate(row) if x == 1), None)
        indices.append(index if index is not None else -1)
    return indices


def get_times_at_destinations(last_m_sol_dfs_final, destinations, num_vehicles_per_DSP, distance_matrix):       
    all_arrive_times = []
    
    for d in range(len(last_m_sol_dfs_final)):    
        # If solution for DSP d exists:
        if len(last_m_sol_dfs_final[d]) > 0:
            x_temp = extract_x_static_single(last_m_sol_dfs_final[d], num_vehicles_per_DSP, distance_matrix)            
            t_temp = extract_t_static_single(last_m_sol_dfs_final[d])
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

def get_distance_and_emissions(last_m_sol_dfs_final, num_vehicles_per_DSP, distance_matrix, emissions_matrix_EV):
    dist_res = []
    emm_res = []
    for d in range(len(last_m_sol_dfs_final)):    
        # If solution exists:
        if len(last_m_sol_dfs_final[d]) > 0:
            # Compute total distance by last-miler
            xm_sol_final_d = extract_x_static_single(last_m_sol_dfs_final[d], num_vehicles_per_DSP, distance_matrix)
            
            dist_res.append(xm_sol_final_d['c_ijm'].sum())  

            # Compute total emissions by last-miler
            total_emissions_d = 0
            for i in range(len(xm_sol_final_d)):
                i_index = int(xm_sol_final_d.loc[i]['i'])
                j_index = int(xm_sol_final_d.loc[i]['j'])
                total_emissions_d += emissions_matrix_EV[i_index][j_index]
            emm_res.append(total_emissions_d)

        else:
            dist_res.append(0) 
            emm_res.append(0)

    # Create columns based on number of last-milers
    cols = []
    for i in range(len(last_m_sol_dfs_final)):
        cols.append('Last Miler ' + str(i))

    # create dataframe and write to file
    dist_emm_df = pd.DataFrame([dist_res, emm_res], columns = cols, index = (['Total distance (km)', 'Total CO2 emissions (gCO2eq)']))
    return dist_emm_df

def get_FM_distance_and_emissions(first_m_sol_dfs, num_vehicles_per_FM, distance_matrix, emissions_matrix_ICE):
    dist_res = []
    emm_res = []
    
    # Compute first mile emissions
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = extract_w(first_m_sol_dfs[i], num_vehicles_per_FM, distance_matrix)
            dist_res.append(w_sol['c_ij'].sum())  
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

def get_FM_follower_objectives(first_m_sol_dfs, num_vehicles_per_FM, distance_matrix):
    follower_objectives = []        
    # Compute follower objectives
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = extract_w(first_m_sol_dfs[i], num_vehicles_per_FM, distance_matrix)
            follower_objectives.append(w_sol['c_ij'].sum())  
        else:
            follower_objectives.append(0)  
    return follower_objectives

def get_LM_follower_objectives(last_m_sol_dfs_final, num_vehicles_per_DSP, distance_matrix, time_violation_penalty):
    follower_objectives = []
    for d in range(len(last_m_sol_dfs_final)):    
        # If solution exists:
        if len(last_m_sol_dfs_final[d]) > 0:
            # Compute total distance by last-miler
            xm_sol_final_d = extract_x_static_single(last_m_sol_dfs_final[d], num_vehicles_per_DSP, distance_matrix)
            total_distance = xm_sol_final_d['c_ijm'].sum()
            # Early and late penalties            
            alphaEarly, alphaLate = extract_alphas_follower(last_m_sol_dfs_final[d])
            total_early_penalty = alphaEarly['value'].sum()
            total_late_penalty = alphaLate['value'].sum()            
            obj_value = total_distance + time_violation_penalty*(total_early_penalty + total_late_penalty)            
            follower_objectives.append(obj_value)  
        else:
            follower_objectives.append(0)             
    return follower_objectives

def compute_follower_objs_first_and_optimal(fm_FIRST, fm_FINAL, lm_FIRST, lm_FINAL, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, time_violation_penalty):
    fm_objs_first_iter = get_FM_follower_objectives(fm_FIRST, num_vehicles_per_FM, distance_matrix)
    fm_objs_optimal = get_FM_follower_objectives(fm_FINAL, num_vehicles_per_FM, distance_matrix)    
    lm_objs_first_iter = get_LM_follower_objectives(lm_FIRST, num_vehicles_per_DSP, distance_matrix, time_violation_penalty)
    lm_objs_optimal = get_LM_follower_objectives(lm_FINAL, num_vehicles_per_DSP, distance_matrix, time_violation_penalty)
    
    objcols = []
    for f in range(len(fm_objs_first_iter)):
        objcols.append('FM ' + str(f) + ' Obj 1st iter')
    for d in range(len(lm_objs_first_iter)):
        objcols.append('LM ' + str(d) + ' Obj 1st iter')
    for f in range(len(fm_objs_optimal)):
        objcols.append('FM ' + str(f) + ' Obj optimal')
    for d in range(len(lm_objs_optimal)):
        objcols.append('LM ' + str(d) + ' Obj optimal')
    
    objectives = fm_objs_first_iter + lm_objs_first_iter + fm_objs_optimal + lm_objs_optimal
    obj_df = pd.DataFrame([objectives], columns = objcols)   
    return obj_df

def compute_follower_objs_MO(fm_FINAL, lm_FINAL, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, time_violation_penalty):
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

def write_instance_results_to_file(instance_name, all_iter_results, total_time, y_sol_final, locker_assignments, first_m_sol_dfs_final, last_m_sol_dfs_final, first_m_sol_dfs_FIRSTITER, last_m_sol_dfs_FIRSTITER, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, emissions_matrix_EV, emissions_matrix_ICE, fm_depots, dsp_depots, locker_nodes, packages, destinations, time_violation_penalty):
    res_filename = 'output/' + instance_name + '_bilevel_results.xlsx'
    
    # Algorithm results
    algo_col_names = ['Iteration number', 'LB', 'UB', 'FM_emissions', 'LM_emissions', 'Gap (%)', 'runtime(s)']    
    algo_df = pd.DataFrame(all_iter_results, columns = algo_col_names) 
    
    # Get distance travelled and total first-mile emissions
    FM_dist_emm_df = get_FM_distance_and_emissions(first_m_sol_dfs_final, num_vehicles_per_FM, distance_matrix, emissions_matrix_ICE)
    
    # Get distance travelled and total last-mile emissions
    dist_emm_df = get_distance_and_emissions(last_m_sol_dfs_final, num_vehicles_per_DSP, distance_matrix, emissions_matrix_EV)
          
    # Get package assignments to Lockers
    locker_ass_df = pd.DataFrame([(k, v) for k, vals in locker_assignments.items() for v in vals], columns=['Locker node', 'Package ID'])
    
    # Get package arrival times
    times_at_dest_df = get_times_at_destinations(last_m_sol_dfs_final, destinations,num_vehicles_per_DSP, distance_matrix)
    
    # Get follower objectives
    foll_objs_df = compute_follower_objs_first_and_optimal(first_m_sol_dfs_FIRSTITER, first_m_sol_dfs_final, last_m_sol_dfs_FIRSTITER, last_m_sol_dfs_final, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, time_violation_penalty)
    
    # Non-competitive
    noncomp_FM_dist_emm_df = get_FM_distance_and_emissions(first_m_sol_dfs_FIRSTITER, num_vehicles_per_FM, distance_matrix, emissions_matrix_ICE)
    noncomp_total_FM_CO2_emissions = noncomp_FM_dist_emm_df.loc['Total CO2 emissions (gCO2eq)'].sum()
    noncomp_dist_emm_df = get_distance_and_emissions(last_m_sol_dfs_FIRSTITER, num_vehicles_per_DSP, distance_matrix, emissions_matrix_EV)
    noncomp_total_LM_CO2_emissions = noncomp_dist_emm_df.loc['Total CO2 emissions (gCO2eq)'].sum()
    noncomp_leader_obj_func = noncomp_total_FM_CO2_emissions + noncomp_total_LM_CO2_emissions
    noncomp_df = pd.DataFrame([noncomp_leader_obj_func], columns=['Total CO2 emissions (gCO2eq)'])
    
    # Write all to file
    with pd.ExcelWriter(res_filename) as writer:  
        algo_df.to_excel(writer, sheet_name='Algorithm results', index = False)
        FM_dist_emm_df.to_excel(writer, sheet_name='FM Distance and Emissions')
        dist_emm_df.to_excel(writer, sheet_name='LM Distance and Emissions')
        locker_ass_df.to_excel(writer, sheet_name='Assignments to Lockers', index=False)
        times_at_dest_df.to_excel(writer, sheet_name='Package Arrival Times', index=False)
        foll_objs_df.to_excel(writer, sheet_name='Follower Objectives', index=False)
        noncomp_df.to_excel(writer, sheet_name='Noncomp emissions', index=False)
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
    
    
    assignments = generate_assignment_array(lists_dict)
    
    return assignments

def HPR_model_emissions(timelimitSecs, packages, destinations, locker_nodes, locker_capacities, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, all_nodes_first_echelon,
                        all_nodes_second_echelon, emissions_matrix_ICE, emissions_matrix_EV, Pf, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs,
                        travel_time_matrix, bigM_matrix, earliest, latest):
    model = gp.Model(name = 'HPR')
    model.setParam('OutputFlag', True)
    model.Params.timelimit = timelimitSecs
    model.Params.MIPFocus  = 1
    model.Params.threads = 4
    model.Params.MIPGap = 1e-2
       
    # ----- Sets -----
    P = packages 
    satellites = locker_nodes
    F = range(num_FirstMilers)
    D = range(num_DSPs)
    
    total_num_vehicles_First_Mile = sum(num_vehicles_per_FM)
    KF = range(total_num_vehicles_First_Mile) # The set of all vehicles in the first-echelon
    
    total_num_vehicles_Last_Mile = sum(num_vehicles_per_DSP)
    KD = range(total_num_vehicles_Last_Mile)
    
    V1 = all_nodes_first_echelon
    A1 = [(i,j) for i in V1 for j in V1 if i!=j]
    
    V2 = all_nodes_second_echelon
    A2 = [(i,j) for i in V2 for j in V2 if i!=j]
    

    vehicle_bounds_FMs = create_bounds(num_vehicles_per_FM)
    vehicle_bounds_DSPs = create_bounds(num_vehicles_per_DSP)
    
    
    # ----- Variables -----
    # ----- Leader Variables -----
    # y_pd = 1 if parcel p is offered to DSP d by the leader
    y = {(p,d): model.addVar(vtype=GRB.BINARY, name='y_%d_%d' % (p,d)) for p in P for d in D}
    
    # lamda_ps = 1 if parcel p is placed at satellite s
    lamda = {(p,s): model.addVar(vtype=GRB.BINARY, name='lamda_%d_%d' % (p,s)) for p in P for s in satellites}
    
    # ----- First-Mile Variables -----
    # w_kij = 1 if arc (i,j) is traversed by vehicle k  
    w = {(k,i,j): model.addVar(vtype=GRB.BINARY, name='w_%d_%d_%d' % (k,i,j)) for k in KF for i in V1 for j in V1 if i!=j}
        
    # Arrival time of vehicle k at node i - tau_ki
    tau = {(k,i): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='tau_%d_%d' % (k,i)) for k in KF for i in V1} 
        
    # ----- Last-Mile Variables -----
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(k,i,j): model.addVar(vtype=GRB.BINARY, name='x_%d_%d_%d' % (k,i,j)) for k in KD for i in V2 for j in V2 if i!=j}
   
    # Arrival time of vehicle k at node i - t_ki
    t = {(k,i): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='t_%d_%d' % (k,i)) for k in KD for i in V2}  
       
    # Variable theta used to linearize the non-convex quadratic constraint
    theta = {(k,p,s):model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='theta_%d_%d_%d' % (k,p,s)) for k in KD for p in P for s in satellites}
    
    # Variables for earliest and latest times
    alpha_early = {(k,i): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='alphaEarly_%d_%d' % (k,i)) for k in KD for i in V2}
    alpha_late = {(k,i): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='alphaLate_%d_%d' % (k,i)) for k in KD for i in V2}    
    
    # Variable z used to linearize the quadratic constraint
    z = {(k,p,s,j):model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='z_%d_%d_%d_%d' % (k,p,s,j)) for k in KD for p in P for s in satellites for j in V2}
       
    # ----- Leader Objective Function -----
    # Minimize emissions in both echelons  
    objective = gp.quicksum(emissions_matrix_ICE[i][j] * w[k,i,j] for (i,j) in A1 for k in KF)
    objective += gp.quicksum(emissions_matrix_EV[i][j] * x[k,i,j] for (i,j) in A2 for k in KD)
    model.setObjective(objective, sense=GRB.MINIMIZE)
    
    # ----- Leader Constraints -----
    # Respect satellites' capacity constraint
    for s in satellites:
        model.addConstr(gp.quicksum(lamda[p,s] for p in P) <= locker_capacities[s])
    
    # A parcel should only be assigned to one satellite
    for p in P:
        model.addConstr(gp.quicksum(lamda[p,s] for s in satellites) == 1)    
    
    # Only one DSP should be assigned to each parcel
    for p in P:
        model.addConstr(gp.quicksum(y[p,d] for d in D) == 1)    
    
    ################ Equity constraint ########################    
#     for d in D:
#         model.addConstr(gp.quicksum(y[p,d] for p in P) >= 3)
    ################ Equity constraint ########################    
    
        
    # ----- First-Mile Follower Constraints -----
    # Link assignment and routing variables
    for f in F:
        for p in Pf[f]:
            for s in satellites:
                model.addConstr(gp.quicksum(w[k,i,s] for i in fm_f_nodes[f] for k in vehicle_bounds_FMs[f] if i!=s) >= lamda[p,s], name="hprLINKVARS")
    
    # Leave depot
    for f in F:
        for k in vehicle_bounds_FMs[f]:
            model.addConstr(gp.quicksum(w[k,fm_depots[f],s] for s in satellites) == 1, name="hprLEAVEDEPOT")
        
    # Flow conservation in first echelon
    for f in F:
        for i in fm_f_nodes[f]:
            for k in vehicle_bounds_FMs[f]:
                model.addConstr(gp.quicksum(w[k,j,i] for j in fm_f_nodes[f] if i!=j) - gp.quicksum(w[k,i,j] for j in fm_f_nodes[f] if i!=j) == 0, name="hprFLOW")  
          
    # Arrival time
    for f in F:
        for (i,j) in fm_f_arcs[f]:
            if j==fm_depots[f]: continue
            for k in vehicle_bounds_FMs[f]:
                model.addConstr(tau[k,i] + travel_time_matrix[i][j] <= 
                                     tau[k,j] + (1 - w[k,i,j]) * 2*bigM_matrix[i][j], name="hprARRIVAL")
     
    # Earliest and latest time windows
    for f in F:
        for k in vehicle_bounds_FMs[f]:
            for i in fm_f_nodes[f]:
                model.addConstr(earliest[i]*gp.quicksum(w[k,i,j] for j in fm_f_nodes[f] if i!=j) <= tau[k,i], name="hprEARLYWINDOW")
                model.addConstr(tau[k,i] <= latest[i]*gp.quicksum(w[k,i,j] for j in fm_f_nodes[f] if i!=j), name="hprLATEWINDOW")
    
    
    
    # ----- Last-Mile Follower Constraints -----
    # Time Violation Constraints
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for i in dsp_d_nodes[d]:
                model.addConstr(alpha_early[k,i] >= earliest[i]*gp.quicksum(x[k,i,j] for j in dsp_d_nodes[d] if i!=j) - t[k,i], name="hprTIMEVIOL1")
                model.addConstr(alpha_late[k,i] >= t[k,i] - latest[i]*gp.quicksum(x[k,i,j] for j in dsp_d_nodes[d] if i!=j), name="hprTIMEVIOL2")
          
    # If a vehicle k picks up a package p, it should go to the destination of p
    for d in D:
        for p in P:
            for k in vehicle_bounds_DSPs[d]:
                model.addConstr(gp.quicksum(x[k,j,destinations[p]] for j in dsp_d_nodes[d] if j != destinations[p]) == y[p,d], name="hprGOTODEST")
    
    # A package p should be picked up from its satellite s
    for d in D:
        for p in P:
            for k in vehicle_bounds_DSPs[d]:
                model.addConstr(gp.quicksum(z[k,p,s,j] for s in satellites for j in dsp_d_nodes[d] if s!=j) >= y[p,d], name="hprPICKUP1")     
    for d in D:
        for p in P:            
            for s in satellites:
                for j in dsp_d_nodes[d]:
                    if j==s: continue
                    for k in vehicle_bounds_DSPs[d]:
                        model.addConstr(z[k,p,s,j] >= lamda[p,s] + x[k,s,j] - 1, name="hprPICKUP2")
                        model.addConstr(z[k,p,s,j] <= lamda[p,s], name="hprPICKUP3")
                        model.addConstr(z[k,p,s,j] <= x[k,s,j], name="hprPICKUP4")               
                
    # A vehicle shoud go from a depot to a satellite
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            model.addConstr(gp.quicksum(x[k,dsp_depots[d],s] for s in satellites) <= 1, name="hprDEPOTTOSAT")    
    
    # A vehicle may go from a satellite to another node
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for s in satellites:
                model.addConstr(gp.quicksum(x[k,s,j] for j in dsp_d_nodes[d] if s!=j) <= 1, name="hprSATTONODE")
    
    
    # Flow conservation at all nodes in second echelon
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for j in dsp_d_nodes[d]:    
                model.addConstr(gp.quicksum(x[k,i,j] for i in dsp_d_nodes[d] if i!=j) - gp.quicksum(x[k,j,i] for i in dsp_d_nodes[d] if i!=j) == 0, name="hprFLOWCONS")
     
    # Satellite Time constraints - Non-convex quadratic constraint that has been linearized
    for p in P:
        for d in D:
            for k in vehicle_bounds_DSPs[d]:
                model.addConstr(gp.quicksum(theta[k,p,s] + lamda[p,s]*travel_time_matrix[s][destinations[p]] for s in satellites) <= 
                                     t[k, destinations[p]] + (1-y[p,d])*max(max(bigM_matrix)), name="hprSAT TIME1")
    
    for p in P:
        for d in D:
            for k in vehicle_bounds_DSPs[d]:
                for s in satellites:
                    model.addConstr(theta[k,p,s] <= lamda[p,s] * max(max(bigM_matrix)), name="hprSAT TIME2")
                    model.addConstr(theta[k,p,s] <= t[k,s] * max(max(bigM_matrix)), name="hprSAT TIME3")
                    model.addConstr(theta[k,p,s] >= t[k,s] - (1-lamda[p,s])*max(max(bigM_matrix)), name="hprSAT TIME4")
    
    # Arrival time
    for d in D:
        for (i,j) in dsp_d_arcs[d]:
            if j==dsp_depots[d]: continue
            for k in vehicle_bounds_DSPs[d]:
                model.addConstr(t[k,i] + x[k,i,j]* travel_time_matrix[i][j] <= 
                                     t[k,j] + (1 - x[k,i,j]) * 3*max(max(bigM_matrix)), name="hprARRIVETIME")
       
    return model

def first_mile_follower_static_single(lamda, Pf, V1f, A1f, fol_depot, cost_per_km, num_vehicles_for_follower, locker_nodes, 
                        distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest):
    model = gp.Model('First_Mile_Follower_Static_Single')
    model.setParam('OutputFlag', False)
    model.Params.timelimit = 600
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

def last_mile_follower_static_single(y, lamda, V2d, A2d, fol_depot, cost_per_km, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty):
    
    model = gp.Model('Last_Mile_Follower_Static_Single')
    model.setParam('OutputFlag', True)
    model.Params.timelimit = 600
    model.Params.MIPFocus  = 1
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

def extract_alphas(hpr_sol, num_vehicles_per_DSP):
    alpha_early_df = None
    alpha_late_df = None
    
    if len(hpr_sol) > 0:
        alpha_early_rows = hpr_sol[hpr_sol['name'].str.startswith('alphaEarly')]
        alpha_early_rows = alpha_early_rows.reset_index(drop=True)  
        
        alpha_late_rows = hpr_sol[hpr_sol['name'].str.startswith('alphaLate')]
        alpha_late_rows = alpha_late_rows.reset_index(drop=True)
        
        vehicle_bounds = create_bounds(num_vehicles_per_DSP)
        
        # Find out which vehicle belongs to which DSP
        list_of_vehicle_bounds = []
        for i in range(len(vehicle_bounds)):
            list_of_vehicle_bounds.append(list(vehicle_bounds[i]))
            
        dk_match = pd.DataFrame(columns = ['d', 'k'])
        for d in range(len(list_of_vehicle_bounds)):
            for k in list_of_vehicle_bounds[d]:
                dk_match.loc[len(dk_match)] = [d, float(k)]
                
               
        alpha_early_df = pd.DataFrame(columns=['k','i','value'])
        for i in range(alpha_early_rows.shape[0]):
            row = alpha_early_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(alpha_early_rows['value'][i])
            alpha_early_df.loc[len(alpha_early_df)] = row        
        alpha_early_df = alpha_early_df.merge(dk_match, on='k', how='left')
        
      
        alpha_late_df = pd.DataFrame(columns=['k','i','value'])
        for i in range(alpha_late_rows.shape[0]):
            row = alpha_late_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(alpha_late_rows['value'][i])
            alpha_late_df.loc[len(alpha_late_df)] = row
        alpha_late_df = alpha_late_df.merge(dk_match, on='k', how='left')
        
    return alpha_early_df,alpha_late_df 

def extract_alphas_follower(sol_df):
    alpha_early_df = None
    alpha_late_df = None    
    if len(sol_df) > 0:
        alpha_early_df = sol_df[sol_df['name'].str.startswith('alphaEarly')]
        alpha_early_df = alpha_early_df.reset_index(drop=True)          
        alpha_late_df = sol_df[sol_df['name'].str.startswith('alphaLate')]
        alpha_late_df = alpha_late_df.reset_index(drop=True)        
    return alpha_early_df,alpha_late_df 


def extract_x_hpr_static_single(sol_df, num_vehicles_per_DSP, distance_matrix):    
    if len(sol_df) > 0:
        x_rows = sol_df[sol_df['name'].str.startswith('x')]
        x_rows = x_rows.reset_index(drop=True)
        vehicle_bounds = create_bounds(num_vehicles_per_DSP)

        # Remove possible duplicate
        if len(x_rows) > 0:
            for i in range(len(x_rows)):
                if x_rows.loc[i]['value'] < 0.1:
                    x_rows.drop([i], inplace=True)
        x_rows.reset_index(inplace=True)
        
        # Find out which vehicle belongs to which DSP
        list_of_vehicle_bounds = []
        for i in range(len(vehicle_bounds)):
            list_of_vehicle_bounds.append(list(vehicle_bounds[i]))

        dk_match = pd.DataFrame(columns = ['d', 'k'])
        for d in range(len(list_of_vehicle_bounds)):
            for k in list_of_vehicle_bounds[d]:
                dk_match.loc[len(dk_match)] = [d, float(k)]

        x_df = pd.DataFrame(columns=['k','i','j','c_ijm'])
        for i in range(x_rows.shape[0]):
            row = x_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(distance_matrix[row[1]][row[2]])
            x_df.loc[len(x_df)] = row
            
        x_df = x_df.merge(dk_match, on='k', how='left')
        x_df['d'] = x_df['d'].astype('int') 

    return x_df

def extract_x_static_single(sol_df, num_vehicles_per_DSP, distance_matrix):    
    if len(sol_df) > 0:
        x_rows = sol_df[sol_df['name'].str.startswith('x')]
        x_rows = x_rows.reset_index(drop=True)

        # Remove possible duplicate
        if len(x_rows) > 0:
            for i in range(len(x_rows)):
                if x_rows.loc[i]['value'] < 0.1:
                    x_rows.drop([i], inplace=True)
        x_rows.reset_index(inplace=True)
        
        
        x_df = pd.DataFrame(columns=['i','j','c_ijm'])
        for i in range(x_rows.shape[0]):
            row = x_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(distance_matrix[row[0]][row[1]])
            x_df.loc[len(x_df)] = row

    return x_df

def extract_w(hpr_sol, num_vehicles_per_FM, distance_matrix):
    w_rows = hpr_sol[hpr_sol['name'].str.startswith('w')]
    w_rows = w_rows.reset_index(drop=True)
    vehicle_bounds = create_bounds(num_vehicles_per_FM)
    
    # Remove possible duplicates
    if len(w_rows) > 0:
        for i in range(len(w_rows)):
            if w_rows.loc[i]['value'] < 0.1:
                w_rows.drop([i], inplace=True)
    w_rows.reset_index(inplace=True)
    
    # Find out which vehicle belongs to which First Miler
    list_of_vehicle_bounds = []
    for i in range(len(vehicle_bounds)):
        list_of_vehicle_bounds.append(list(vehicle_bounds[i]))
    
    fk_match = pd.DataFrame(columns = ['f', 'k'])
    
    for f in range(len(list_of_vehicle_bounds)):
        for k in list_of_vehicle_bounds[f]:
            fk_match.loc[len(fk_match)] = [f, float(k)]

    w_df = pd.DataFrame(columns=['k','i','j','c_ij'])
    for i in range(w_rows.shape[0]):
        row = w_rows['name'][i].split('_')
        row.pop(0);    
        row = [int(i) for i in row]
        row.append(distance_matrix[row[1]][row[2]])
        w_df.loc[len(w_df)] = row

    w_df = w_df.merge(fk_match, on='k', how='left')
    w_df['f'] = w_df['f'].astype('int')
    
    return w_df

def extract_tau(firstmiler_final_sol):
    tau_rows = firstmiler_final_sol[firstmiler_final_sol['name'].str.startswith('tau')]
    tau_rows = tau_rows.reset_index(drop=True)
    
    # Remove possible duplicates
    if len(tau_rows) > 0:
        for i in range(len(tau_rows)):
            if tau_rows.loc[i]['value'] < 0.1:
                tau_rows.drop([i], inplace=True)
    tau_rows.reset_index(inplace=True)

    tau_df = pd.DataFrame(columns=['k','i','time'])
    for i in range(tau_rows.shape[0]):
        row = tau_rows['name'][i].split('_')
        row.pop(0);    
        row = [int(i) for i in row]
        row.append(tau_rows['value'][i])
        tau_df.loc[len(tau_df)] = row
    
    return tau_df

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
        t_rows.reset_index(inplace=True)

        t_df = pd.DataFrame(columns=['k','i','time'])
        for i in range(t_rows.shape[0]):
            row = t_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(t_rows['value'][i])
            t_df.loc[len(t_df)] = row    
      
    return t_df

def extract_t_static_single(lastmiler_final_sol):
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

def extract_t_all_LMs(hpr_sol, num_vehicles_per_DSP):    
    t_df_temp = extract_t(hpr_sol)
    
    # which k belongs to which d?
    vehicle_bounds = create_bounds(num_vehicles_per_DSP)
    # Find out which vehicle belongs to which last Miler
    list_of_vehicle_bounds = []
    for i in range(len(vehicle_bounds)):
        list_of_vehicle_bounds.append(list(vehicle_bounds[i]))

    dk_match = pd.DataFrame(columns = ['d', 'k'])
    for d in range(len(list_of_vehicle_bounds)):
        for k in list_of_vehicle_bounds[d]:
            dk_match.loc[len(dk_match)] = [d, float(k)]

    t_df = t_df_temp
    t_df = t_df.merge(dk_match, on='k', how='left')
    t_df['d'] = t_df['d'].astype('int')    
    
    return t_df

# Compute FM followers'
def compute_first_mile_follower_obj(w_sol, num_FirstMilers, cost_per_km_for_FM):
    V_hat = []
    
    # For each FM follower...
    for f in range(num_FirstMilers):
        # If there is a solution...
        if len(w_sol) != 0:
            cost = w_sol.loc[w_sol['f'] == f, 'c_ij'].sum()
            cost = cost * cost_per_km_for_FM[f]
            V_hat.append(cost)
        else:
            V_hat.append(0)
        
    return V_hat

# Solve parameterized follower problem for each FM
def solve_param_first_mile_followers(lamda, Pf, num_FirstMilers, num_vehicles_per_FM, cost_per_km_for_FM, fm_depots, fm_f_nodes, fm_f_arcs,
                                     locker_nodes, distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest):
    V = []
    sol_df_vec = []
    
    for f in range(num_FirstMilers):    
        obj, sol_df = first_mile_follower_static_single(lamda, Pf[f], fm_f_nodes[f], fm_f_arcs[f], fm_depots[f], cost_per_km_for_FM[f], num_vehicles_per_FM[f], locker_nodes, distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest)

        V.append(obj) 
        sol_df_vec.append(sol_df)
    
    return V, sol_df_vec


def compute_last_mile_follower_obj(xm_sol, t_sol, num_DSPs, cost_per_km_for_DSP, earliest, latest, time_violation_penalty):
    V_hat = []    
    # For each last-mile follower...
    for d in range(num_DSPs):
        # If there is a solution...
        if len(xm_sol) != 0:
            cost = cost_per_km_for_DSP[d] * (xm_sol.loc[xm_sol['d'] == d, 'c_ijm'].sum())
            
            penalty_early = 0
            penalty_late = 0
            xm_sol_d = xm_sol.loc[xm_sol['d'] == d]
            xm_sol_d = xm_sol_d.reset_index(drop=True)            
            t_sol_d = t_sol.loc[t_sol['d'] == d]
            t_sol_d = t_sol_d.reset_index(drop=True)       
            
            for x in range(len(t_sol_d)):
                i_index = int(t_sol_d.loc[x]['i'])
                k_index = int(t_sol_d.loc[x]['k'])
                t_minus_l = float(t_sol_d.loc[t_sol_d['k'] == k_index][t_sol_d['i'] == i_index]['time'] - latest[i_index])
                penalty_late += max(t_minus_l, 0)    
                e_minus_t = float(earliest[i_index] - t_sol_d.loc[t_sol_d['k'] == k_index][t_sol_d['i'] == i_index]['time'])
                penalty_early += max(e_minus_t, 0)
               
            V_hat.append(cost + time_violation_penalty*(penalty_early + penalty_late))
            
        else:
            V_hat.append(0)
    
    return V_hat

# Solve parameterized follower problem for each DSP
def solve_param_last_mile_followers(y_sol, lamda_sol, num_DSPs, num_vehicles_per_DSP, cost_per_km_for_DSP, dsp_depots, dsp_d_nodes, dsp_d_arcs, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty):
    V = []
    sol_df_vec = []

    for d in range(num_DSPs):      
        obj, sol_df = last_mile_follower_static_single_cp(y_sol[:, d], lamda_sol, dsp_d_nodes[d], dsp_d_arcs[d], dsp_depots[d], cost_per_km_for_DSP[d], packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty)

        V.append(obj)
        sol_df_vec.append(sol_df)
        
    return V, sol_df_vec

def do_check(epss, diff):    
    checks = []*len(diff)

    for item in diff:
        if item > epss:
            c = True
        else:
            c = False
        checks.append(c)

    return checks

def extract_lamda_sol(hpr_sol, num_packages, num_nodes):
    lamda_subset = hpr_sol[hpr_sol['name'].str.startswith('lamda')]
    lamda_sol = np.zeros((num_packages, num_nodes))    
    for _, row in lamda_subset.iterrows():
        _, i, j = row['name'].split('_')
        i = int(i)  # row index
        j = int(j)  # column index    
        lamda_sol[i, j] = row['value']    
    return lamda_sol

def extract_y_sol(hpr_sol, num_packages, num_DSPs):
    y_subset = hpr_sol[hpr_sol['name'].str.startswith('y')]
    y_sol = np.zeros((num_packages, num_DSPs))    
    for _, row in y_subset.iterrows():
        _, i, j = row['name'].split('_')
        i = int(i)  # row index
        j = int(j)  # column index    
        y_sol[i, j] = row['value']    
    return y_sol

def solve_HPR_model(model, packages, num_nodes, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, verboseTorF):    
     # Solve
    model.optimize()
    hpr_obj_value = model.ObjBound # best bound
    var_names = []
    var_values = []
    for v in model.getVars():
        if v.x != 0:
            var_names.append(v.varName)
            var_values.append(v.x)
    
    hpr_sol = pd.DataFrame({'name': var_names, 'value': var_values})
    w_sol = extract_w(hpr_sol, num_vehicles_per_FM, distance_matrix)      
    x_sol = extract_x_hpr_static_single(hpr_sol, num_vehicles_per_DSP, distance_matrix)
    t_sol = extract_t_all_LMs(hpr_sol, num_vehicles_per_DSP)
                               
    lamda_sol = extract_lamda_sol(hpr_sol, len(packages), num_nodes)
    # removing numerical inconsistencies
    for p in range(len(packages)):
        for n in range(num_nodes):
            if lamda_sol[p,n] != np.nan:
                if lamda_sol[p,n] <= 0.1:
                    lamda_sol[p,n] =  0.0

    y_sol = extract_y_sol(hpr_sol, len(packages), num_DSPs)
    # removing numerical inconsistencies
    for p in range(len(packages)):
        for d in range(num_DSPs):
            if y_sol[p,d] <= 0.1:
                y_sol[p,d] = 0.0               
            
    return hpr_sol, lamda_sol, w_sol, x_sol, y_sol, hpr_obj_value, t_sol

def solve_HPR_model_cp(model, packages, num_nodes, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, verboseTorF):    
    # Solve
    solution = model.solve(log_output = verboseTorF)   
    hpr_obj_value = model.get_solve_details().best_bound
   
    hpr_sol = None
    lamda_sol = None
    y_sol = None
        
    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        hpr_sol = solution.as_df()        
        w_sol = extract_w(hpr_sol, num_vehicles_per_FM, distance_matrix)      
        x_sol = extract_x_hpr_static_single(hpr_sol, num_vehicles_per_DSP, distance_matrix)
        t_sol = extract_t_all_LMs(hpr_sol, num_vehicles_per_DSP)
                     
        lamda_sol = np.zeros((len(packages), num_nodes))
        for p in range(len(packages)):
            for n in range(num_nodes):
                lamda_sol[p,n] = model.get_var_by_name('lamda_%d_%d' % (p,n))
        # removing numerical inconsistencies
        for p in range(len(packages)):
            for n in range(num_nodes):
                if lamda_sol[p,n] != np.nan:
                    if lamda_sol[p,n] <= 0.1:
                        lamda_sol[p,n] =  0.0
                            
        y_sol = np.zeros((len(packages), num_DSPs))
        for p in range(len(packages)):
            for d in range(num_DSPs):
                y_sol[p,d] = model.get_var_by_name('y_%d_%d' % (p,d))
        # removing numerical inconsistencies
        for p in range(len(packages)):
            for d in range(num_DSPs):
                if y_sol[p,d] <= 0.1:
                    y_sol[p,d] = 0.0               
            
    return hpr_sol, lamda_sol, w_sol, x_sol, y_sol, hpr_obj_value, t_sol


def compute_upper_bound(first_m_sol_dfs, last_m_sol_dfs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix,
                       emissions_matrix_ICE, emissions_matrix_EV, cost_per_DSP_vehicle):
    # Compute first mile emissions
    fm_emissions = []
    for i in range(len(first_m_sol_dfs)):
        if len(first_m_sol_dfs[i]) > 0:
            w_sol = extract_w(first_m_sol_dfs[i], num_vehicles_per_FM, distance_matrix)
            for t in range(len(w_sol)):
                arc_emiss_fm = emissions_matrix_ICE[int(w_sol.loc[t]['i'])][int(w_sol.loc[t]['j'])]
                fm_emissions.append(arc_emiss_fm)
    
    # Compute last mile emissions
    lm_emissions = []
    for i in range(len(last_m_sol_dfs)):
        if len(last_m_sol_dfs[i]) > 0:
            xm_sol = extract_x_static_single(last_m_sol_dfs[i], num_vehicles_per_DSP, distance_matrix)    
            for t in range(len(xm_sol)):
                arc_emiss_lm = emissions_matrix_EV[int(xm_sol.loc[t]['i'])][int(xm_sol.loc[t]['j'])]
                lm_emissions.append(arc_emiss_lm)
                
    # Sum up and output
    FM_emissions = sum(fm_emissions)
    LM_emissions = sum(lm_emissions)
    total = FM_emissions + LM_emissions
    return total, FM_emissions, LM_emissions

def compute_big_Ms_for_cuts(distance_matrix, fm_f_arcs, dsp_d_arcs, cost_per_km_for_FM, cost_per_km_for_DSP, time_violation_penalty, final_leave_time, earliest, latest):
    bigMcutF = []
    bigMcutD = []
    
    # First-mile followers
    for f in range(len(fm_f_arcs)):
        Mf = 0
        for a in range(len(fm_f_arcs[f])):
            i_index = int(fm_f_arcs[f][a][0])
            j_index = int(fm_f_arcs[f][a][1])
            Mf += cost_per_km_for_FM[f] * distance_matrix[i_index][j_index]
        bigMcutF.append(Mf)
               
    # Last-mile followers
    for d in range(len(dsp_d_arcs)):
        Md_lhs = 0
        # Routing cost
        for b in range(len(dsp_d_arcs[d])):
            i_index = int(dsp_d_arcs[d][b][0])
            j_index = int(dsp_d_arcs[d][b][1])        
            Md_lhs += cost_per_km_for_DSP[d] * 50*distance_matrix[i_index][j_index] # taking the largest distance of all time periods
        # Add time violation penalty
        Md_rhs = time_violation_penalty * (max((final_leave_time - latest[i_index]), (earliest[i_index] - final_leave_time)))
        bigMcutD.append(Md_lhs + Md_rhs)
    
    return bigMcutF, bigMcutD

def cutting_plane_algorithm_FOLLOWER_AND_INTERDICTION(hpr_mod, Pf, packages, destinations, locker_nodes, num_nodes, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, cost_per_km_for_FM, cost_per_km_for_DSP,cost_per_DSP_vehicle, emissions_matrix_ICE, emissions_matrix_EV, 
                            distance_matrix, travel_time_matrix, bigM_matrix,  final_leave_time, earliest, latest, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs, 
                            time_violation_penalty, verboseTorF, problem_time_limit):
    num_iterations = 1
    epsilon = 1e-2#1e-4
    bigMcutF, bigMcutD = compute_big_Ms_for_cuts(distance_matrix, fm_f_arcs, dsp_d_arcs, cost_per_km_for_FM, cost_per_km_for_DSP, time_violation_penalty, final_leave_time, earliest, latest)
    convergedFirst = False
    convergedLast = False
    converged = False
    start_time = pc()
    best_LB = -np.inf
    best_UB = np.inf
    best_FM_emissions = None
    best_LM_emissions = None
    lamda_opt = None
    y_opt = None
    lamda_first_iter = None
    y_first_iter = None
    first_m_sol_dfs_FIRST = None
    last_m_sol_dfs_FIRST = None
    first_m_sol_dfs_opt = None
    last_m_sol_dfs_opt = None
    all_iter_results = []
    
    veh_bounds_FM = create_bounds(num_vehicles_per_FM)
    veh_bounds_DSP = create_bounds(num_vehicles_per_DSP)
   
    while converged == False:
        print('\n>>> Iteration:', num_iterations)
        iter_res = []
        iter_res.append(num_iterations)
        
        # 1. Solve HPR to get lamda, w, x, y 
        print('Solving HPR...')
        hpr_sol, lamda_sol, hpr_w_sol, hpr_xm_sol, y_sol, hpr_obj_value, t_sol = solve_HPR_model(hpr_mod, packages, num_nodes, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, verboseTorF)      

        alpha_early_sol, alpha_late_sol = extract_alphas(hpr_sol,num_vehicles_per_DSP)
        
        # iteration lower bound
        if hpr_obj_value > best_LB:
            best_LB = hpr_obj_value
        iter_res.append(best_LB)
                            
        # 2. For each follower:        
        #    - evaluate FM follower response
        V_hat_FirstM = compute_first_mile_follower_obj(hpr_w_sol, num_FirstMilers, cost_per_km_for_FM)

        #    - solve FM Follower(lamda)
        print('Solving First-Mile follower problems...')
        V_FirstM, first_m_sol_dfs = solve_param_first_mile_followers(lamda_sol, Pf, num_FirstMilers, num_vehicles_per_FM, cost_per_km_for_FM, fm_depots, fm_f_nodes, fm_f_arcs, locker_nodes, distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest)

        #    - First Mile: If  V_f_hat - V_f > epsilon, generate optimality cut
        difference_FM = list(np.array(V_hat_FirstM) - np.array(V_FirstM))
        
        #    - evaluate LM follower response to y 
        V_hat_LastM = compute_last_mile_follower_obj(hpr_xm_sol, t_sol, num_DSPs, cost_per_km_for_DSP, earliest, latest, time_violation_penalty)

        #    - solve LM Follower(y)
        print('Solving Last-Mile follower problems...')
        V_LastM, last_m_sol_dfs = solve_param_last_mile_followers(y_sol, lamda_sol, num_DSPs, num_vehicles_per_DSP, cost_per_km_for_DSP, dsp_depots, dsp_d_nodes, dsp_d_arcs, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty)

        #    - Last Mile: If V_d_hat - V_d > epsilon, generate optimality cut
        difference_LM = list(np.array(V_hat_LastM) - np.array(V_LastM))  
        
        # Compute iteration upper bound
        iter_UB, iter_FM_emissions, iter_LM_emissions = compute_upper_bound(first_m_sol_dfs, last_m_sol_dfs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix,
                       emissions_matrix_ICE, emissions_matrix_EV, cost_per_DSP_vehicle)
        if iter_UB < best_UB:
            best_UB = iter_UB  
            lamda_opt = lamda_sol
            y_opt = y_sol
            first_m_sol_dfs_opt = first_m_sol_dfs
            last_m_sol_dfs_opt = last_m_sol_dfs
            best_FM_emissions = iter_FM_emissions
            best_LM_emissions = iter_LM_emissions
        iter_res.append(best_UB)
        iter_res.append(best_FM_emissions)
        iter_res.append(best_LM_emissions)
        
        print('best_UB = ', best_UB)
        print('best_LB = ', best_LB)
                
        # Save non-competitive emissions (first iteration)
        if num_iterations == 1:
            lamda_first_iter = lamda_sol
            y_first_iter = y_sol
            first_m_sol_dfs_FIRST = first_m_sol_dfs
            last_m_sol_dfs_FIRST = last_m_sol_dfs 
            
        
        # Compute iteration gap
        iter_Gap = 100*(best_UB - best_LB)/best_UB 
        iter_res.append(iter_Gap)
        
        
        checks_FM = do_check(epsilon, difference_FM)
        
        checks_LM = do_check(epsilon, difference_LM)

        
        if any(checks_FM): 
            print('Adding FM cuts...')
            # Add optimality cuts to HPR  for violating follower only
            for f in range(num_FirstMilers):
                if checks_FM[f] == True:                    
                    # get indices where lamda_p_s == 1
                    one_indices = np.argwhere(lamda_sol[Pf[f], :] == 1)
                    # Split one_indices into two
                    one_p_indices = one_indices[:, 0]
                    one_s_indices = one_indices[:, 1]

                    # get indices where lamda_p_s == 0
                    zero_indices = np.argwhere(lamda_sol[Pf[f], :] == 0)
                    # Split zero_indices into two
                    zero_p_indices = zero_indices[:, 0]
                    zero_s_indices = zero_indices[:, 1]

                    value_ones = gp.quicksum(lamda_sol[Pf[f], :][p][s] - hpr_mod.getVarByName('lamda_%d_%d' % (p,s)) for p in one_p_indices for s in one_s_indices)
                    value_zeros = gp.quicksum(hpr_mod.getVarByName('lamda_%d_%d' % (p,s)) - lamda_sol[Pf[f], :][p][s] for p in zero_p_indices for s in zero_s_indices)

                    # add constraint
                    hpr_mod.addConstr(gp.quicksum(cost_per_km_for_FM[f]*distance_matrix[i][j] * hpr_mod.getVarByName('w_%d_%d_%d' % (k,i,j)) for k in veh_bounds_FM[f] for i in fm_f_nodes[f] for j in fm_f_nodes[f] if i!=j)   
                                               <= V_FirstM[f] + (value_ones + value_zeros)*bigMcutF[f], "FMcut")
                    hpr_mod.update()
        else:
            convergedFirst = True
        
        if any(checks_LM):
            print('Adding LM cuts...')
            # Add optimality cuts to HPR  for violating follower only
            for d in range(num_DSPs):
                if checks_LM[d] == True:
                    # get rows where y_sol == 1
                    one_indices = np.argwhere(y_sol[:, d] == 1).flatten() 
                    # get rows where y_sol == 0
                    zero_indices = np.argwhere(y_sol[:, d] == 0).flatten()                

                    value_ones = gp.quicksum(y_sol[:, d][p] - hpr_mod.getVarByName('y_%d_%d' % (p,d)) for p in one_indices)
                    value_zeros = gp.quicksum(hpr_mod.getVarByName('y_%d_%d' % (p,d)) - y_sol[:, d][p] for p in zero_indices)

                    # add constraint
                    hpr_mod.addConstr(gp.quicksum(cost_per_km_for_DSP[d]*distance_matrix[i][j] * hpr_mod.getVarByName('x_%d_%d_%d' % (k,i,j)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d] for j in dsp_d_nodes[d] if i!=j)
                                           + time_violation_penalty * gp.quicksum(hpr_mod.getVarByName('alphaEarly_%d_%d' % (k,i)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d])
                                           + time_violation_penalty * gp.quicksum(hpr_mod.getVarByName('alphaLate_%d_%d' % (k,i)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d])
                                           <= V_LastM[d] + (value_ones + value_zeros)*bigMcutD[d], "LMcut")
                    hpr_mod.update()
                    
        else:
            convergedLast = True
                  
        
        # update number of iterations
        num_iterations += 1   
        iter_res.append(pc()-start_time)
        all_iter_results.append(iter_res)
        
                
        if convergedFirst == True and convergedLast == True:
            print('All followers optimal. Terminating.')            
            converged = True  
        
        if iter_Gap <= epsilon:
            print('Gap closed. Terminating.') 
            converged = True
        else:
            print('Adding interdiction cut')
            hpr_mod.addConstr(gp.quicksum(hpr_mod.getVarByName('lamda_%d_%d' % (p,s)) * (1 - lamda_sol[p, s])  
            + (1 - hpr_mod.getVarByName('lamda_%d_%d' % (p,s))) * lamda_sol[p, s] for p in packages for s in locker_nodes)
            + gp.quicksum(hpr_mod.getVarByName('y_%d_%d' % (p,d)) * (1 - y_sol[p, d])  
            + (1 - hpr_mod.getVarByName('y_%d_%d' % (p,d))) * y_sol[p, d] for p in packages for d in range(num_DSPs)) 
            >= 1, "Intdcut")     
            hpr_mod.update()

       
        # update time
        current_time = pc()-start_time        
        if current_time > problem_time_limit:
            print('End of time limit. Terminating.')
            converged = True

    return hpr_sol, lamda_opt, hpr_w_sol, hpr_xm_sol, y_opt, first_m_sol_dfs_opt, last_m_sol_dfs_opt, first_m_sol_dfs_FIRST, last_m_sol_dfs_FIRST, all_iter_results
    

def cutting_plane_algorithm_MODIFIED(hpr_mod, Pf, packages, destinations, locker_nodes, num_nodes, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, cost_per_km_for_FM, cost_per_km_for_DSP,cost_per_DSP_vehicle, emissions_matrix_ICE, emissions_matrix_EV, 
                            distance_matrix, travel_time_matrix, bigM_matrix,  final_leave_time, earliest, latest, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs, 
                            time_violation_penalty, verboseTorF, problem_time_limit):
    num_iterations = 1
    epsilon = 1e-2#1e-4
    bigMcutF, bigMcutD = compute_big_Ms_for_cuts(distance_matrix, fm_f_arcs, dsp_d_arcs, cost_per_km_for_FM, cost_per_km_for_DSP, time_violation_penalty, final_leave_time, earliest, latest)
    convergedFirst = False
    convergedLast = False
    converged = False
    start_time = pc()
    best_LB = -np.inf
    best_UB = np.inf
    best_FM_emissions = None
    best_LM_emissions = None
    lamda_opt = None
    y_opt = None
    lamda_first_iter = None
    y_first_iter = None
    first_m_sol_dfs_FIRST = None
    last_m_sol_dfs_FIRST = None
    first_m_sol_dfs_opt = None
    last_m_sol_dfs_opt = None
    all_iter_results = []
    
    veh_bounds_FM = create_bounds(num_vehicles_per_FM)
    veh_bounds_DSP = create_bounds(num_vehicles_per_DSP)
   
    while converged == False:
        print('\n>>> Iteration:', num_iterations)
        iter_res = []
        iter_res.append(num_iterations)
        
        # 1. Solve HPR to get lamda, w, x, y 
        print('Solving HPR...')
        hpr_sol, lamda_sol, hpr_w_sol, hpr_xm_sol, y_sol, hpr_obj_value, t_sol = solve_HPR_model(hpr_mod, packages, num_nodes, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, verboseTorF)      

        alpha_early_sol, alpha_late_sol = extract_alphas(hpr_sol,num_vehicles_per_DSP)
        
        # iteration lower bound
        if hpr_obj_value > best_LB:
            best_LB = hpr_obj_value
        iter_res.append(best_LB)
                            
        # 2. For each follower:        
        #    - evaluate FM follower response
        V_hat_FirstM = compute_first_mile_follower_obj(hpr_w_sol, num_FirstMilers, cost_per_km_for_FM)

        #    - solve FM Follower(lamda)
        print('Solving First-Mile follower problems...')
        V_FirstM, first_m_sol_dfs = solve_param_first_mile_followers(lamda_sol, Pf, num_FirstMilers, num_vehicles_per_FM, cost_per_km_for_FM, fm_depots, fm_f_nodes, fm_f_arcs, locker_nodes, distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest)

        #    - First Mile: If  V_f_hat - V_f > epsilon, generate optimality cut
        difference_FM = list(np.array(V_hat_FirstM) - np.array(V_FirstM))
        
        #    - evaluate LM follower response to y 
        V_hat_LastM = compute_last_mile_follower_obj(hpr_xm_sol, t_sol, num_DSPs, cost_per_km_for_DSP, earliest, latest, time_violation_penalty)

        #    - solve LM Follower(y)
        print('Solving Last-Mile follower problems...')
        V_LastM, last_m_sol_dfs = solve_param_last_mile_followers(y_sol, lamda_sol, num_DSPs, num_vehicles_per_DSP, cost_per_km_for_DSP, dsp_depots, dsp_d_nodes, dsp_d_arcs, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty)

        #    - Last Mile: If V_d_hat - V_d > epsilon, generate optimality cut
        difference_LM = list(np.array(V_hat_LastM) - np.array(V_LastM))  
        
        # Compute iteration upper bound
        iter_UB, iter_FM_emissions, iter_LM_emissions = compute_upper_bound(first_m_sol_dfs, last_m_sol_dfs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix,
                       emissions_matrix_ICE, emissions_matrix_EV, cost_per_DSP_vehicle)
        if iter_UB < best_UB:
            best_UB = iter_UB  
            lamda_opt = lamda_sol
            y_opt = y_sol
            first_m_sol_dfs_opt = first_m_sol_dfs
            last_m_sol_dfs_opt = last_m_sol_dfs
            best_FM_emissions = iter_FM_emissions
            best_LM_emissions = iter_LM_emissions
        iter_res.append(best_UB)
        iter_res.append(best_FM_emissions)
        iter_res.append(best_LM_emissions)
        
        print('best_UB = ', best_UB)
        print('best_LB = ', best_LB)
                
        # Save non-competitive emissions (first iteration)
        if num_iterations == 1:
            lamda_first_iter = lamda_sol
            y_first_iter = y_sol
            first_m_sol_dfs_FIRST = first_m_sol_dfs
            last_m_sol_dfs_FIRST = last_m_sol_dfs 
            
        
        # Compute iteration gap
        iter_Gap = 100*(best_UB - best_LB)/best_UB 
        iter_res.append(iter_Gap)
        
        
        checks_FM = do_check(epsilon, difference_FM)
        
        checks_LM = do_check(epsilon, difference_LM)

        
        if any(checks_FM): 
            print('Adding FM cuts...')
            # Add optimality cuts to HPR  for violating follower only
            for f in range(num_FirstMilers):
                if checks_FM[f] == True:                    
                    # get indices where lamda_p_s == 1
                    one_indices = np.argwhere(lamda_sol[Pf[f], :] == 1)
                    # Split one_indices into two
                    one_p_indices = one_indices[:, 0]
                    one_s_indices = one_indices[:, 1]

                    # get indices where lamda_p_s == 0
                    zero_indices = np.argwhere(lamda_sol[Pf[f], :] == 0)
                    # Split zero_indices into two
                    zero_p_indices = zero_indices[:, 0]
                    zero_s_indices = zero_indices[:, 1]

                    value_ones = gp.quicksum(lamda_sol[Pf[f], :][p][s] - hpr_mod.getVarByName('lamda_%d_%d' % (p,s)) for p in one_p_indices for s in one_s_indices)
                    try:
                        value_zeros = gp.quicksum(hpr_mod.getVarByName('lamda_%d_%d' % (p,s)) - lamda_sol[Pf[f], :][p][s] for p in zero_p_indices for s in zero_s_indices)
                    except:
                        value_zeros = 0

                    # add constraint
                    hpr_mod.addConstr(gp.quicksum(cost_per_km_for_FM[f]*distance_matrix[i][j] * hpr_mod.getVarByName('w_%d_%d_%d' % (k,i,j)) for k in veh_bounds_FM[f] for i in fm_f_nodes[f] for j in fm_f_nodes[f] if i!=j)   
                                               <= V_FirstM[f] + (value_ones + value_zeros)*bigMcutF[f], "FMcut")
                    hpr_mod.update()
        else:
            convergedFirst = True
        
        if any(checks_LM):
            print('Adding LM cuts...')
            # Add optimality cuts to HPR  for violating follower only
            for d in range(num_DSPs):
                if checks_LM[d] == True:
                    # get indices where lamda_p_s == 1
                    lamda_one_indices = np.argwhere(lamda_sol[:, :] == 1)
                    # Split one_indices into two
                    lamda_one_p_indices = lamda_one_indices[:, 0]
                    lamda_one_s_indices = lamda_one_indices[:, 1]

                    # get indices where lamda_p_s == 0
                    lamda_zero_indices = np.argwhere(lamda_sol[:, :] == 0)
                    # Split zero_indices into two
                    lamda_zero_p_indices = lamda_zero_indices[:, 0]
                    lamda_zero_s_indices = lamda_zero_indices[:, 1]
                    
                    try:
                        lamda_value_ones = gp.quicksum(lamda_sol[:, :][p][s] - hpr_mod.getVarByName('lamda_%d_%d' % (p,s)) for p in lamda_one_p_indices for s in lamda_one_s_indices)
                    except:
                        lamda_value_ones = 0
                    
                    try:
                        lamda_value_zeros = gp.quicksum(hpr_mod.getVarByName('lamda_%d_%d' % (p,s)) - lamda_sol[:, :][p][s] for p in lamda_zero_p_indices for s in lamda_zero_s_indices)
                    except:
                        lamda_value_zeros = 0
                    
                    # get rows where y_sol == 1
                    one_indices = np.argwhere(y_sol[:, d] == 1).flatten() 
                    # get rows where y_sol == 0
                    zero_indices = np.argwhere(y_sol[:, d] == 0).flatten()                

                    try:
                        value_ones = gp.quicksum(y_sol[:, d][p] - hpr_mod.getVarByName('y_%d_%d' % (p,d)) for p in one_indices)
                    except:
                        value_ones = 0
                    try:
                        value_zeros = gp.quicksum(hpr_mod.getVarByName('y_%d_%d' % (p,d)) - y_sol[:, d][p] for p in zero_indices)
                    except:
                        value_zeros = 0

                    # add constraint
                    hpr_mod.addConstr(gp.quicksum(cost_per_km_for_DSP[d]*distance_matrix[i][j] * hpr_mod.getVarByName('x_%d_%d_%d' % (k,i,j)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d] for j in dsp_d_nodes[d] if i!=j)
                                           + time_violation_penalty * gp.quicksum(hpr_mod.getVarByName('alphaEarly_%d_%d' % (k,i)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d])
                                           + time_violation_penalty * gp.quicksum(hpr_mod.getVarByName('alphaLate_%d_%d' % (k,i)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d])
                                           <= V_LastM[d] + (lamda_value_ones + lamda_value_zeros + value_ones + value_zeros)*bigMcutD[d], "LMcut")
                    hpr_mod.update()
                    
        else:
            convergedLast = True
                  
        
        # update number of iterations
        num_iterations += 1   
        iter_res.append(pc()-start_time)
        all_iter_results.append(iter_res)
        
                
        if convergedFirst == True and convergedLast == True:
            print('All followers optimal. Terminating.')            
            converged = True  
        
        if iter_Gap <= epsilon:
            print('Gap closed. Terminating.') 
            converged = True
        else:
            print('Adding interdiction cut')
            hpr_mod.addConstr(gp.quicksum(hpr_mod.getVarByName('lamda_%d_%d' % (p,s)) * (1 - lamda_sol[p, s])  
            + (1 - hpr_mod.getVarByName('lamda_%d_%d' % (p,s))) * lamda_sol[p, s] for p in packages for s in locker_nodes)
            + gp.quicksum(hpr_mod.getVarByName('y_%d_%d' % (p,d)) * (1 - y_sol[p, d])  
            + (1 - hpr_mod.getVarByName('y_%d_%d' % (p,d))) * y_sol[p, d] for p in packages for d in range(num_DSPs)) 
            >= 1, "Intdcut")     
            hpr_mod.update()

       
        # update time
        current_time = pc()-start_time        
        if current_time > problem_time_limit:
            print('End of time limit. Terminating.')
            converged = True

    return hpr_sol, lamda_opt, hpr_w_sol, hpr_xm_sol, y_opt, first_m_sol_dfs_opt, last_m_sol_dfs_opt, first_m_sol_dfs_FIRST, last_m_sol_dfs_FIRST, all_iter_results
    
def HPR_model_emissions_cp(timelimitSecs, packages, destinations, locker_nodes, locker_capacities, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, all_nodes_first_echelon,
                        all_nodes_second_echelon, emissions_matrix_ICE, emissions_matrix_EV, Pf, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs,
                        travel_time_matrix, bigM_matrix, earliest, latest):
    model = Model(name = 'HPR')
    model.parameters.timelimit = timelimitSecs
    model.parameters.emphasis.mip = 1 # emphasize feasibility
    model.parameters.threads = 4   
    
    # ----- Sets -----
    P = packages 
    satellites = locker_nodes
    F = range(num_FirstMilers)
    D = range(num_DSPs)
    
    total_num_vehicles_First_Mile = sum(num_vehicles_per_FM)
    KF = range(total_num_vehicles_First_Mile) # The set of all vehicles in the first-echelon
    
    total_num_vehicles_Last_Mile = sum(num_vehicles_per_DSP)
    KD = range(total_num_vehicles_Last_Mile)
    
    V1 = all_nodes_first_echelon
    A1 = [(i,j) for i in V1 for j in V1 if i!=j]
    
    V2 = all_nodes_second_echelon
    A2 = [(i,j) for i in V2 for j in V2 if i!=j]
    

    vehicle_bounds_FMs = create_bounds(num_vehicles_per_FM)
    vehicle_bounds_DSPs = create_bounds(num_vehicles_per_DSP)
    
    
    # ----- Variables -----
    # ----- Leader Variables -----
    # y_pd = 1 if parcel p is offered to DSP d by the leader
    y = {(p,d):model.binary_var(name='y_%d_%d' % (p,d)) for p in P for d in D}
    
    # lamda_ps = 1 if parcel p is placed at satellite s
    lamda = {(p,s):model.binary_var(name='lamda_%d_%d' % (p,s)) for p in P for s in satellites}
    
    # ----- First-Mile Variables -----
    # w_kij = 1 if arc (i,j) is traversed by vehicle k  
    w = {(k,i,j):model.binary_var(name='w_%d_%d_%d' % (k,i,j)) for k in KF for i in V1 for j in V1 if i!=j}
        
    # Arrival time of vehicle k at node i - tau_ki
    tau = {(k,i):model.continuous_var(lb=0.0, name='tau_%d_%d' % (k,i)) for k in KF for i in V1} 
    
    # ----- Last-Mile Variables -----
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(k,i,j):model.binary_var(name='x_%d_%d_%d' % (k,i,j)) for k in KD for i in V2 for j in V2 if i!=j}
   

    # Arrival time of vehicle k at node i - t_ki
    t = {(k,i):model.continuous_var(lb=0.0, name='t_%d_%d' % (k,i)) for k in KD for i in V2}  
    
    # Variable theta used to linearize the non-convex quadratic constraint
    theta = {(k,p,s):model.continuous_var(lb=0.0, name='theta_%d_%d_%d' % (k,p,s)) for k in KD for p in P for s in satellites}
    
    # Variables for earliest and latest times
    alpha_early = {(k,i): model.continuous_var(lb=0.0, name='alphaEarly_%d_%d' % (k,i)) for k in KD for i in V2}
    alpha_late = {(k,i): model.continuous_var(lb=0.0, name='alphaLate_%d_%d' % (k,i)) for k in KD for i in V2}

    # Variable z used to linearize the quadratic constraint
    z = {(k,p,s,j):model.continuous_var(lb=0.0, name='z_%d_%d_%d_%d' % (k,p,s,j)) for k in KD for p in P for s in satellites for j in V2}
    
  
    
    # ----- Leader Objective Function -----
    # Minimize emissions in both echelons  
    model.minimize(model.sum(emissions_matrix_ICE[i][j] * w[k,i,j] for (i,j) in A1 for k in KF)
                   + model.sum(emissions_matrix_EV[i][j] * x[k,i,j] for (i,j) in A2 for k in KD)
                  )  
    
      
    # ----- Leader Constraints -----
    # Respect satellites' capacity constraint
    for s in satellites:
        model.add_constraint(model.sum(lamda[p,s] for p in P) <= locker_capacities[s])
    
    # A parcel should only be assigned to one satellite
    for p in P:
        model.add_constraint(model.sum(lamda[p,s] for s in satellites) == 1)    
    
    # Only one DSP should be assigned to each parcel
    for p in P:
        model.add_constraint(model.sum(y[p,d] for d in D) == 1)    
    
    #### At least one package to a DSP ########################    
#     for d in D:
#         model.add_constraint(model.sum(y[p,d] for p in P) >= 1)
    #### At least one package to a DSP ########################
    
        
    # ----- First-Mile Follower Constraints -----
    # Link assignment and routing variables
    for f in F:
        for p in Pf[f]:
            for s in satellites:
                model.add_constraint(model.sum(w[k,i,s] for i in fm_f_nodes[f] for k in vehicle_bounds_FMs[f] if i!=s) >= lamda[p,s], ctname="hprLINKVARS")
    
    # Leave depot
    for f in F:
        for k in vehicle_bounds_FMs[f]:
            model.add_constraint(model.sum(w[k,fm_depots[f],s] for s in satellites) == 1, ctname="hprLEAVEDEPOT")
        
    # Flow conservation in first echelon
    for f in F:
        for i in fm_f_nodes[f]:
            for k in vehicle_bounds_FMs[f]:
                model.add_constraint(model.sum(w[k,j,i] for j in fm_f_nodes[f] if i!=j) - model.sum(w[k,i,j] for j in fm_f_nodes[f] if i!=j) == 0,ctname="hprFLOW")  
          
    # Arrival time
    for f in F:
        for (i,j) in fm_f_arcs[f]:
            if j==fm_depots[f]: continue
            for k in vehicle_bounds_FMs[f]:
                model.add_constraint(tau[k,i] + travel_time_matrix[i][j] <= 
                                     tau[k,j] + (1 - w[k,i,j]) * 2*bigM_matrix[i][j], ctname="hprARRIVAL")
     
    # Earliest and latest time windows
    for f in F:
        for k in vehicle_bounds_FMs[f]:
            for i in fm_f_nodes[f]:
                model.add_constraint(earliest[i]*model.sum(w[k,i,j] for j in fm_f_nodes[f] if i!=j) <= tau[k,i], ctname="hprEARLYWINDOW")
                model.add_constraint(tau[k,i] <= latest[i]*model.sum(w[k,i,j] for j in fm_f_nodes[f] if i!=j), ctname="hprLATEWINDOW")
    
    
    
    # ----- Last-Mile Follower Constraints -----
    # Time Violation Constraints
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for i in dsp_d_nodes[d]:
                model.add_constraint(alpha_early[k,i] >= earliest[i]*model.sum(x[k,i,j] for j in dsp_d_nodes[d] if i!=j) - t[k,i],ctname="hprTIMEVIOL1")
                model.add_constraint(alpha_late[k,i] >= t[k,i] - latest[i]*model.sum(x[k,i,j] for j in dsp_d_nodes[d] if i!=j),ctname="hprTIMEVIOL2")
          
    # If a vehicle k picks up a package p, it should go to the destination of p
    for d in D:
        for p in P:
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(model.sum(x[k,j,destinations[p]] for j in dsp_d_nodes[d] if j != destinations[p]) == y[p,d], ctname="hprGOTODEST")
    
    # A package p should be picked up from its satellite s
#     for d in D:
#         for p in P:
#             for k in vehicle_bounds_DSPs[d]:
#                 model.add_constraint(model.sum(lamda[p,s]*x[k,s,j] for s in satellites for j in dsp_d_nodes[d] if s!=j) >= mu[p,k])      
    for d in D:
        for p in P:
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(model.sum(z[k,p,s,j] for s in satellites for j in dsp_d_nodes[d] if s!=j) >= y[p,d], ctname="hprPICKUP1")     
    for d in D:
        for p in P:            
            for s in satellites:
                for j in dsp_d_nodes[d]:
                    if j==s: continue
                    for k in vehicle_bounds_DSPs[d]:
                        model.add_constraint(z[k,p,s,j] >= lamda[p,s] + x[k,s,j] - 1, ctname="hprPICKUP2")
                        model.add_constraint(z[k,p,s,j] <= lamda[p,s],ctname="hprPICKUP3")
                        model.add_constraint(z[k,p,s,j] <= x[k,s,j],ctname="hprPICKUP4")               
                
    # A vehicle shoud go from a depot to a satellite
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            model.add_constraint(model.sum(x[k,dsp_depots[d],s] for s in satellites) <= 1,ctname="hprDEPOTTOSAT")    
    
    # A vehicle may go from a satellite to another node
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for s in satellites:
                model.add_constraint(model.sum(x[k,s,j] for j in dsp_d_nodes[d] if s!=j) <= 1,ctname="hprSATTONODE")
    
    
    # Flow conservation at all nodes in second echelon
    for d in D:
        for k in vehicle_bounds_DSPs[d]:
            for j in dsp_d_nodes[d]:    
                model.add_constraint(model.sum(x[k,i,j] for i in dsp_d_nodes[d] if i!=j) - model.sum(x[k,j,i] for i in dsp_d_nodes[d] if i!=j) == 0, ctname="hprFLOWCONS")
     
    # Satellite Time constraints - Non-convex quadratic constraint that has been linearized
    for p in P:
        for d in D:
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(model.sum(theta[k,p,s] + lamda[p,s]*travel_time_matrix[s][destinations[p]] for s in satellites) <= 
                                     t[k, destinations[p]] + (1-y[p,d])*max(max(bigM_matrix)),ctname="hprSAT TIME1")
    
    for p in P:
        for d in D:
            for k in vehicle_bounds_DSPs[d]:
                for s in satellites:
                    model.add_constraint(theta[k,p,s] <= lamda[p,s] * max(max(bigM_matrix)),ctname="hprSAT TIME2")
                    model.add_constraint(theta[k,p,s] <= t[k,s] * max(max(bigM_matrix)),ctname="hprSAT TIME3")
                    model.add_constraint(theta[k,p,s] >= t[k,s] - (1-lamda[p,s])*max(max(bigM_matrix)),ctname="hprSAT TIME4")
    
    # Arrival time
    for d in D:
        for (i,j) in dsp_d_arcs[d]:
            if j==dsp_depots[d]: continue
            for k in vehicle_bounds_DSPs[d]:
                model.add_constraint(t[k,i] + x[k,i,j]* travel_time_matrix[i][j] <= 
                                     t[k,j] + (1 - x[k,i,j]) * 3*max(max(bigM_matrix)),ctname="hprARRIVETIME")
       
    return model

def cutting_plane_algorithm_FOLLOWER_AND_INTERDICTION_cp(hpr_mod, Pf, packages, destinations, locker_nodes, num_nodes, num_FirstMilers, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, cost_per_km_for_FM, cost_per_km_for_DSP,cost_per_DSP_vehicle, emissions_matrix_ICE, emissions_matrix_EV, 
                            distance_matrix, travel_time_matrix, bigM_matrix,  final_leave_time, earliest, latest, fm_depots, fm_f_nodes, fm_f_arcs, dsp_depots, dsp_d_nodes, dsp_d_arcs, 
                            time_violation_penalty, verboseTorF, problem_time_limit):
    num_iterations = 1
    epsilon = 1e-2
    bigMcutF, bigMcutD = compute_big_Ms_for_cuts(distance_matrix, fm_f_arcs, dsp_d_arcs, cost_per_km_for_FM, cost_per_km_for_DSP, time_violation_penalty, final_leave_time, earliest, latest)
    convergedFirst = False
    convergedLast = False
    converged = False
    start_time = pc()
    best_LB = -np.inf
    best_UB = np.inf
    best_FM_emissions = None
    best_LM_emissions = None
    lamda_opt = None
    y_opt = None
    lamda_first_iter = None
    y_first_iter = None
    first_m_sol_dfs_FIRST = None
    last_m_sol_dfs_FIRST = None
    first_m_sol_dfs_opt = None
    last_m_sol_dfs_opt = None
    all_iter_results = []
    
    veh_bounds_FM = create_bounds(num_vehicles_per_FM)
    veh_bounds_DSP = create_bounds(num_vehicles_per_DSP)
   
    while converged == False:
        print('\n>>> Iteration:', num_iterations)
        iter_res = []
        iter_res.append(num_iterations)
        
        # 1. Solve HPR to get lamda, w, x, y 
        print('Solving HPR...')
        hpr_sol, lamda_sol, hpr_w_sol, hpr_xm_sol, y_sol, hpr_obj_value, t_sol = solve_HPR_model_cp(hpr_mod, packages, num_nodes, num_DSPs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix, verboseTorF)      

        alpha_early_sol, alpha_late_sol = extract_alphas(hpr_sol,num_vehicles_per_DSP)
        
        # iteration lower bound
        if hpr_obj_value > best_LB:
            best_LB = hpr_obj_value
        iter_res.append(best_LB)
                            
        # 2. For each follower:        
        #    - evaluate FM follower response
        V_hat_FirstM = compute_first_mile_follower_obj(hpr_w_sol, num_FirstMilers, cost_per_km_for_FM)
#         print('V_hat_FirstM = ', V_hat_FirstM)
        #    - solve FM Follower(lamda)
        print('Solving First-Mile follower problems...')
        V_FirstM, first_m_sol_dfs = solve_param_first_mile_followers(lamda_sol, Pf, num_FirstMilers, num_vehicles_per_FM, cost_per_km_for_FM, fm_depots, fm_f_nodes, fm_f_arcs, locker_nodes, distance_matrix, travel_time_matrix, bigM_matrix, earliest, latest)
#         print('V_FirstM = ', V_FirstM)
        #    - First Mile: If  V_f_hat - V_f > epsilon, generate optimality cut
        difference_FM = list(np.array(V_hat_FirstM) - np.array(V_FirstM))
#         print('difference_FM =',difference_FM)
        
        #    - evaluate LM follower response to y 
        V_hat_LastM = compute_last_mile_follower_obj(hpr_xm_sol, t_sol, num_DSPs, cost_per_km_for_DSP, earliest, latest, time_violation_penalty)
#         print('V_hat_LastM = ', V_hat_LastM)

        #    - solve LM Follower(y)
        print('Solving Last-Mile follower problems...')
        V_LastM, last_m_sol_dfs = solve_param_last_mile_followers(y_sol, lamda_sol, num_DSPs, num_vehicles_per_DSP, cost_per_km_for_DSP, dsp_depots, dsp_d_nodes, dsp_d_arcs, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty)
#         print('V_LastM = ', V_LastM)        
        #    - Last Mile: If V_d_hat - V_d > epsilon, generate optimality cut
        difference_LM = list(np.array(V_hat_LastM) - np.array(V_LastM))  
#         print('difference_LM =',difference_LM)
        
        # Compute iteration upper bound
        iter_UB, iter_FM_emissions, iter_LM_emissions = compute_upper_bound(first_m_sol_dfs, last_m_sol_dfs, num_vehicles_per_FM, num_vehicles_per_DSP, distance_matrix,
                       emissions_matrix_ICE, emissions_matrix_EV, cost_per_DSP_vehicle)
        if iter_UB < best_UB:
            best_UB = iter_UB  
            lamda_opt = lamda_sol
            y_opt = y_sol
            first_m_sol_dfs_opt = first_m_sol_dfs
            last_m_sol_dfs_opt = last_m_sol_dfs
            best_FM_emissions = iter_FM_emissions
            best_LM_emissions = iter_LM_emissions
#             print('y_opt=',y_opt)
        iter_res.append(best_UB)
        iter_res.append(best_FM_emissions)
        iter_res.append(best_LM_emissions)
                
        # Save non-competitive emissions (first iteration)
        if num_iterations == 1:
            lamda_first_iter = lamda_sol
            y_first_iter = y_sol
            first_m_sol_dfs_FIRST = first_m_sol_dfs
            last_m_sol_dfs_FIRST = last_m_sol_dfs 
            
        
        # Compute iteration gap
        iter_Gap = 100*(best_UB - best_LB)/best_UB 
        iter_res.append(iter_Gap)
        
        print('best_UB = ', best_UB)
        print('best_LB = ', best_LB)
        
        
        checks_FM = do_check(epsilon, difference_FM)
        # print('checks_FM:', checks_FM)
        
        checks_LM = do_check(epsilon, difference_LM)
        # print('checks_LM:', checks_LM)
        
        if any(checks_FM): 
            print('Adding FM cuts...')
            # Add optimality cuts to HPR  for violating follower only
            for f in range(num_FirstMilers):
                if checks_FM[f] == True:                    
                    # get indices where lamda_p_s == 1
                    one_indices = np.argwhere(lamda_sol[Pf[f], :] == 1)
                    # Split one_indices into two
                    one_p_indices = one_indices[:, 0]
                    one_s_indices = one_indices[:, 1]

                    # get indices where lamda_p_s == 0
                    zero_indices = np.argwhere(lamda_sol[Pf[f], :] == 0)
                    # Split zero_indices into two
                    zero_p_indices = zero_indices[:, 0]
                    zero_s_indices = zero_indices[:, 1]

                    value_ones = hpr_mod.sum(lamda_sol[Pf[f], :][p][s] - hpr_mod.get_var_by_name('lamda_%d_%d' % (p,s)) for p in one_p_indices for s in one_s_indices)
                    value_zeros = hpr_mod.sum(hpr_mod.get_var_by_name('lamda_%d_%d' % (p,s)) - lamda_sol[Pf[f], :][p][s] for p in zero_p_indices for s in zero_s_indices)

                    # add constraint
                    hpr_mod.add_constraint(hpr_mod.sum(cost_per_km_for_FM[f]*distance_matrix[i][j] * hpr_mod.get_var_by_name('w_%d_%d_%d' % (k,i,j)) for k in veh_bounds_FM[f] for i in fm_f_nodes[f] for j in fm_f_nodes[f] if i!=j)   
                                               <= V_FirstM[f] + (value_ones + value_zeros)*bigMcutF[f], "FMcut")
        else:
            convergedFirst = True
        
        if any(checks_LM):
            print('Adding LM cuts...')
            # Add optimality cuts to HPR  for violating follower only
            for d in range(num_DSPs):
                if checks_LM[d] == True:
                    # get rows where y_sol == 1
                    one_indices = np.argwhere(y_sol[:, d] == 1).flatten() 
                    # get rows where y_sol == 0
                    zero_indices = np.argwhere(y_sol[:, d] == 0).flatten()                

                    value_ones = hpr_mod.sum(y_sol[:, d][p] - hpr_mod.get_var_by_name('y_%d_%d' % (p,d)) for p in one_indices)
                    value_zeros = hpr_mod.sum(hpr_mod.get_var_by_name('y_%d_%d' % (p,d)) - y_sol[:, d][p] for p in zero_indices)

                    # add constraint
                    hpr_mod.add_constraint(hpr_mod.sum(cost_per_km_for_DSP[d]*distance_matrix[i][j] * hpr_mod.get_var_by_name('x_%d_%d_%d' % (k,i,j)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d] for j in dsp_d_nodes[d] if i!=j)
                                           + time_violation_penalty * hpr_mod.sum(hpr_mod.get_var_by_name('alphaEarly_%d_%d' % (k,i)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d])
                                           + time_violation_penalty * hpr_mod.sum(hpr_mod.get_var_by_name('alphaLate_%d_%d' % (k,i)) for k in veh_bounds_DSP[d] for i in dsp_d_nodes[d])
                                           <= V_LastM[d] + (value_ones + value_zeros)*bigMcutD[d], "LMcut")
                    
        else:
            convergedLast = True
                  
        
        # update number of iterations
        num_iterations += 1   
        iter_res.append(pc()-start_time)
        all_iter_results.append(iter_res)
        
                
        if convergedFirst == True and convergedLast == True:
            print('All followers optimal. Terminating.')            
            converged = True  
        
        if iter_Gap <= epsilon:
            print('Gap closed. Terminating.') 
            converged = True
        else:
            print('Adding interdiction cut')
            cut = hpr_mod.add_constraint(hpr_mod.sum(hpr_mod.get_var_by_name('lamda_%d_%d' % (p,s)) * (1 - lamda_sol[p, s])  
            + (1 - hpr_mod.get_var_by_name('lamda_%d_%d' % (p,s))) * lamda_sol[p, s] for p in packages for s in locker_nodes)
            + hpr_mod.sum(hpr_mod.get_var_by_name('y_%d_%d' % (p,d)) * (1 - y_sol[p, d])  
            + (1 - hpr_mod.get_var_by_name('y_%d_%d' % (p,d))) * y_sol[p, d] for p in packages for d in range(num_DSPs)) 
            >= 1, "Intdcut")     
#             print(cut)
       
        # update time
        current_time = pc()-start_time        
        if current_time > problem_time_limit:
            print('End of time limit. Terminating.')
            converged = True

        #         print('all_iter_results = ', all_iter_results)

    return hpr_sol, lamda_opt, hpr_w_sol, hpr_xm_sol, y_opt, first_m_sol_dfs_opt, last_m_sol_dfs_opt, first_m_sol_dfs_FIRST, last_m_sol_dfs_FIRST, all_iter_results

def last_mile_follower_static_single_cp(y, lamda, V2d, A2d, fol_depot, cost_per_km, packages, destinations, locker_nodes, distance_matrix, travel_time_matrix, earliest, latest, bigM_matrix, time_violation_penalty):
    model = Model(name = 'Last_Mile_Follower_Static_Single')
    model.parameters.timelimit = 70#600
    model.parameters.emphasis.mip = 1 # emphasize feasibility
    model.parameters.threads = 4
    model.parameters.mip.tolerances.mipgap = 1e-2
    
    # --- Sets ---
    P = packages 
    satellites = locker_nodes

    # ----- Variables -----     
        
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(i,j):model.binary_var(name='x_%d_%d' % (i,j)) for i in V2d for j in V2d if i!=j}
        
    # Arrival time of vehicle k at node i - t_ki
    t = {i:model.continuous_var(lb=0.0, name='t_%d' % (i)) for i in V2d}  
        
    # Variables for earliest and latest times
    alpha_early = {i: model.continuous_var(lb=0.0, name='alphaEarly_%d' % (i)) for i in V2d}
    alpha_late = {i: model.continuous_var(lb=0.0, name='alphaLate_%d' % (i)) for i in V2d}
    
    # Variable z used to linearize the quadratic constraint
    z = {(p,s,j):model.continuous_var(lb=0.0, name='z_%d_%d_%d' % (p,s,j)) for p in P for s in satellites for j in V2d}
    
    # ----- Objective function: Minimize the travel cost ----- 
    model.minimize(model.sum(cost_per_km*distance_matrix[i][j] * x[i,j] for (i,j) in A2d)
                   + time_violation_penalty * model.sum(alpha_early[i] for i in V2d)
                   + time_violation_penalty * model.sum(alpha_late[i] for i in V2d)
                  )
    
    # ----- Time Violation Constraints ----- 
    for i in V2d:
        model.add_constraint(alpha_early[i] >= earliest[i]*model.sum(x[i,j] for j in V2d if i!=j) - t[i],ctname="TIMEVIOL1")
        model.add_constraint(alpha_late[i] >= t[i] - latest[i]*model.sum(x[i,j] for j in V2d if i!=j), ctname="TIMEVIOL2")
    
    # ----- Constraints -----            
    # If a vehicle k picks up a package p, it should go to the destination of p
    for p in P:
        model.add_constraint(model.sum(x[j,destinations[p]] for j in V2d if j != destinations[p]) == y[p], ctname="GOTODEST")
    
    # A package p should be picked up from its satellite s   
    for p in P:        
        model.add_constraint(model.sum(z[p,s,j] for s in satellites for j in V2d if s!=j) >= y[p],ctname="PICKUP1")     
    for p in P:
        for s in satellites:
            for j in V2d:
                if j==s: continue        
                model.add_constraint(z[p,s,j] >= lamda[p,s] + x[s,j] - 1,ctname="PICKUP2")
                model.add_constraint(z[p,s,j] <= lamda[p,s],ctname="PICKUP3")
                model.add_constraint(z[p,s,j] <= x[s,j],ctname="PICKUP4")

    # A vehicle shoud go from a depot to a satellite
    model.add_constraint(model.sum(x[fol_depot,s] for s in satellites) <= 1,ctname="DEPOTTOSAT")
    
    # A vehicle may go from a satellite to another node    
    for s in satellites:
        model.add_constraint(model.sum(x[s,j] for j in V2d if s!=j) <= 1,ctname="SATTONODE")
    
    # Flow conservation at all nodes
    for j in V2d:    
        model.add_constraint(model.sum(x[i,j] for i in V2d if i!=j) - model.sum(x[j,i] for i in V2d if i!=j) == 0, ctname="FLOWCONS")
 
    
    # Satellite Time constraints
    for p in P:
        model.add_constraint(model.sum(lamda[p,s]*(t[s] + travel_time_matrix[s][destinations[p]]) for s in satellites) <= t[destinations[p]] + (1-y[p])*max(max(bigM_matrix)),ctname="SAT TIME")#bigN[p][k])
                
    # Arrival time
    for (i,j) in A2d:
        if j==fol_depot: continue
        model.add_constraint(t[i] + x[i,j]* travel_time_matrix[i][j] <= t[j] + (1 - x[i,j]) *3*max(max(bigM_matrix)),ctname="ARRIVETIME")
    
    # Solve
    solution = model.solve(log_output = True)   
    obj_val = None
    sol_df = None
    best_bound = None
    mip_gap = None
    solve_time = None

    
    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        obj_val = solution.get_objective_value()
        sol_df = solution.as_df()
        best_bound = model.get_solve_details().best_bound
        mip_gap = model.solve_details.mip_relative_gap
        solve_time = model.solve_details.time
        
    return obj_val, sol_df#best_bound, mip_gap, solve_time, sol_df
