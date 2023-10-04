# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:27:05 2023

@author: 11481
"""

import random, math, pickle, time, os
import argparse
import numpy as np
from parameters import Parameters
from functions import initialize_demand, initialize_vehicle, get_current_location, matching, robust_model_function, robust_model_function_interval, distributionally_robust_model, optimization
from collections import defaultdict
from datetime import datetime, timedelta


parser = argparse.ArgumentParser(description='reb-opt')

parser.add_argument("--engine", type=str, default="true_demand", help="type of optimization engine")
parser.add_argument("--CI", type=int, default=95, help="confidence interval for calculating range")

fleet_size = 2000
args = parser.parse_args()
matching_engine = args.engine
if matching_engine == "graph_lstm_interval":
    output_path = "output/graph_lstm_poisson_0627/"
    ρ_list = [3]
    Γ_list = [0,1,2,3,4,5,6,7,8,9,10]
elif matching_engine == "graph_lstm_dro":
    output_path = "output/graph_lstm_poisson_0627_dro/"
    ρ_list = [3]
    Γ_list = [5]    
elif matching_engine == "historical_interval":
    output_path = "output/historical_poisson_0627/"
    ρ_list = [3]
    Γ_list = [0,1,2,3,4,5,6,7,8,9,10]
elif matching_engine == "true_demand":
    output_path = "output/true_demand_0627/"
    ρ_list = [3]
    Γ_list = [5]

if not os.path.exists(output_path):
    os.makedirs(output_path)

params = Parameters(args.CI)

for ρ in ρ_list:
    for Γ in Γ_list:

        # Initialize demand and vehicle objects
        demand_list, demand_id_dict = initialize_demand(params.demand_data, params.zone_to_road_node_dict, params.zone_index_id_dict)
        vehicle_list, vehicle_id_dict = initialize_vehicle(fleet_size, params.n, params.zone_to_road_node_dict)
    
        simulation_start_time = datetime(2019,6,27,params.start_time[0],0,0)
        simulation_end_time = datetime(2019,6,27,params.end_time[0],0,0)
        simulation_time = simulation_start_time
            
        while True:
            if simulation_time >= simulation_end_time:
                break
    
            time_index = int((simulation_time - simulation_start_time).total_seconds() / params.time_interval_length)
            number_of_intervals = int(params.rebalancing_time_length / params.time_interval_length) # number of intervals in one optimization step
            end_time_index = time_index + min(params.d[:, :, time_index:].shape[2], number_of_intervals) # end time index for current optimization step
    
            P_matrix = params.P[:,:,time_index:end_time_index] # Probility of vehicle staying occupied
            Q_matrix = params.Q[:,:,time_index:end_time_index] # Probability of occupied vehicle becomes vacant
    
            # Find initial occupied & vacant vehicle distributions
            V_init = np.zeros(params.n) # vacant vehicles
            O_init = np.zeros(params.n) # occupied vehicles
            zone_vacant_veh_dict = defaultdict(list)
            for veh in vehicle_list:
                veh_loc = veh.current_location
                vehicle_zone = params.road_node_to_zone_dict[veh_loc]
                if veh.occupied:
                    O_init[params.zones_index.index(vehicle_zone)] += 1 # zone index from 0 to 62
                else:
                    V_init[params.zones_index.index(vehicle_zone)] += 1
                    zone_vacant_veh_dict[vehicle_zone].append(veh.id)
    
            print("Rebalancing Phase ")
            print(simulation_time)
    
            K_sub = end_time_index - time_index
            a_sub = params.a[:,:,time_index:end_time_index] # if traveling time is bigger than rebalancing threshold
            b_sub = params.b[:,:,time_index:end_time_index] # if traveling time is bigger than maximum waiting time
            d_sub = params.d[:,:,time_index:end_time_index] # zone centroids distance 
    
            if matching_engine == "graph_lstm":
                μ = params.graph_lstm_mean[:, time_index:end_time_index] # predicted mean
                σ = params.graph_lstm_var[:, time_index:end_time_index] # predicted standard deviation
                rebalancing_decision = robust_model_function(μ, σ, ρ, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, params.β, params.γ)
            elif matching_engine == "graph_lstm_interval":
                μ = params.graph_lstm_mean[:, time_index:end_time_index] # predicted mean
                lb = params.graph_lstm_lb[:, time_index:end_time_index]
                ub = params.graph_lstm_ub[:, time_index:end_time_index]
                rebalancing_decision = robust_model_function_interval(μ, lb, ub, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, params.β, params.γ)   
            elif matching_engine == "graph_lstm_dro":
                μ = params.graph_lstm_mean[:, time_index:end_time_index] # predicted mean
                σ = params.graph_lstm_var[:, time_index:end_time_index] # predicted standard deviation
                lb = params.graph_lstm_lb[:, time_index:end_time_index]
                ub = params.graph_lstm_ub[:, time_index:end_time_index]
                rebalancing_decision = distributionally_robust_model(μ, σ, lb, ub, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, params.β, params.γ)                          
            # elif matching_engine == "historical":
            #     μ = demand_mean[:, time_index:end_time_index]
            #     σ = demand_std[:, time_index:end_time_index]
            #     rebalancing_decision = robust_model_function(μ, σ, ρ, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)
            elif matching_engine == "historical_interval":
                μ = params.demand_mean[:, time_index:end_time_index] # predicted mean
                lb = params.demand_lb[:, time_index:end_time_index]
                ub = params.demand_ub[:, time_index:end_time_index]
                rebalancing_decision = robust_model_function_interval(μ, lb, ub, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, params.β, params.γ)     
            elif matching_engine == "true_demand":
                r = params.true_demand[:, time_index:end_time_index]
                rebalancing_decision = optimization(r, V_init, O_init, P_matrix, Q_matrix, params.n, K_sub, a_sub, b_sub, d_sub, params.β, params.γ)
    
            rebalancing_decision = (np.floor(rebalancing_decision[:,:,0])).astype(int)
    
            # Rebalancing vacant vehicles
            for i in range(params.n):
                for j in range(params.n):
                    rebalancing_veh_number = rebalancing_decision[i,j]
                    if rebalancing_veh_number <= 0:
                        continue
                    #Random.seed!(2020)
                    rebalancing_veh_list = random.sample(zone_vacant_veh_dict[params.zones_index[i]], rebalancing_veh_number)
                    for veh_id in rebalancing_veh_list:
                        veh = vehicle_id_dict[veh_id]
                        random_number = 0
                        Flag = False
                        #Random.seed!(2020)
                        while True:
                            dest_node = random.choice(params.zone_to_road_node_dict[params.zones_index[j]])
                            rebalancing_dist = params.road_distance_matrix[veh.current_location, dest_node]
                            rebalancing_time = (rebalancing_dist / params.average_speed) * 3600
                            if rebalancing_time <= params.time_interval_length:
                                break
                            random_number += 1
                            # Sample 10 times to get points in the two zones between which the traveling time is less than the maximum
                            if random_number >= 10:
                                Flag = True
                                break
                        if Flag:
                            continue
                        # Update Vehicle Objects
                        veh.rebalancing_travel_distance.append(rebalancing_dist)
                        veh.rebalancing_trips.append(1)
                        veh.location = dest_node
                        veh.current_location = dest_node
                        veh.available_time = simulation_time + timedelta(seconds=(int(math.floor(rebalancing_time))))
    
            # update current location for vehicles arrival at their destinations during the current time interval
            for veh in vehicle_list:
                if simulation_time <= veh.available_time < simulation_time + timedelta(seconds=params.time_interval_length):
                    veh.current_location = veh.location
    
            # Matching engine in the simulation
            matching_simulation_time = simulation_time
            while True:
                print("Matching phase ")
                print(f"{matching_simulation_time}")
                if matching_simulation_time >= simulation_time + timedelta(seconds=params.time_interval_length):
                    break
    
                available_vehicles = []
                for veh in vehicle_list:
                    if veh.available_time < matching_simulation_time + timedelta(seconds=params.matching_window):
                        available_vehicles.append(veh)
    
                requesting_demands = []
                for dem in demand_list:
                    if simulation_start_time <= dem.request_time < matching_simulation_time + timedelta(seconds=params.matching_window):
                        if dem.assign_time is None:
                            if not dem.leave_system:
                                requesting_demands.append(dem)
    
                matching_list, unserved_pax_list = matching(available_vehicles, requesting_demands)
    
                # Update Passengers not matched
                for pax_id in unserved_pax_list:
                    pax = demand_id_dict[pax_id]
                    pax.wait_time += params.matching_window
                    if pax.wait_time >= params.maximum_waiting_time:
                        pax.leave_system = True
    
                for ((veh_id, pax_id), pickup_dist) in matching_list:
                    pax = demand_id_dict[pax_id]
                    pickup_time = pickup_dist / params.average_speed * 3600
                    pax.wait_time = pickup_time + params.matching_window + datetime.timestamp(matching_simulation_time) - datetime.timestamp(pax.request_time)
                    pax.travel_time = params.road_distance_matrix[pax.origin, pax.destination] / params.average_speed * 3600
                    pax.assign_time = matching_simulation_time + timedelta(seconds=params.matching_window)
                    pax.arrival_time = pax.assign_time + timedelta(seconds=math.floor(pickup_time)) + timedelta(seconds=math.floor(pax.travel_time))
    
                    veh = vehicle_id_dict[veh_id]
                    veh.location = pax.destination
                    veh.available_time = pax.arrival_time
                    veh.served_passenger.append(pax.id)
                    veh.pickup_travel_distance.append(pickup_dist)
                    veh.occupied_travel_distance.append(params.road_distance_matrix[pax.origin, pax.destination])
    
                matching_simulation_time += timedelta(seconds=params.matching_window)
    
            # Update vehicle status for next rebalancing time window (availability and position when next time window starts)
            matching_time = matching_simulation_time
            for veh in vehicle_list:
                if matching_time <= veh.available_time:
                    veh.occupied = True
                    veh.current_location = get_current_location(matching_time, veh, demand_id_dict)
                else:
                    veh.occupied = False
                    veh.current_location = veh.location
    
            simulation_time += timedelta(seconds=params.time_interval_length)
    
        print("Simulation Ends")
        output = dict()
        # Output simulation results
        vehicle_served_passenger_list = []
        vehicle_occupied_dist_list = []
        vehicle_pickup_dist_list = []
        vehicle_rebalancing_dist_list = []
        vehicle_rebalancing_trip_list = []
        for veh in vehicle_list:
            vehicle_served_passenger_list.append(veh.served_passenger)
            vehicle_occupied_dist_list.append(veh.occupied_travel_distance)
            vehicle_pickup_dist_list.append(veh.pickup_travel_distance)
            vehicle_rebalancing_dist_list.append(veh.rebalancing_travel_distance)
            vehicle_rebalancing_trip_list.append(veh.rebalancing_trips)
    
        output["vehicle_served_passenger"] = vehicle_served_passenger_list
        output["vehicle_occupied_dist"] = vehicle_occupied_dist_list
        output["vehicle_pickup_dist"] = vehicle_pickup_dist_list
        output["vehicle_rebalancing_dist"] = vehicle_rebalancing_dist_list
        output["vehicle_rebalancing_trip"] = vehicle_rebalancing_trip_list
    
        pax_wait_time_list = []
        pax_travel_time_list = []
        pax_leave_list = []
        pax_request_time_list = []
        pax_leave_number = 0
        total_pax_number = 0
        for pax in demand_list:
            total_pax_number += 1
            pax_request_time_list.append(pax.request_time)
            if pax.wait_time > 0 and not pax.leave_system:
                pax_wait_time_list.append(pax.wait_time)
            if pax.travel_time > 0 and not pax.leave_system:
                pax_travel_time_list.append(pax.travel_time)
            if pax.leave_system:
                pax_leave_list.append(1)
                pax_leave_number += 1
    
        output["pax_wait_time"] = pax_wait_time_list
        output["pax_travel_time"] = pax_travel_time_list
        output["pax_leaving"] = pax_leave_list
        output["pax_leaving_rate"] = [pax_leave_number / total_pax_number]
        output["pax_request_time"] = pax_request_time_list
    
        print(f"Unserved rate: {pax_leave_number / total_pax_number}")
    
        if matching_engine == "true_demand":
            with open(output_path + matching_engine + "_" + str(params.start_time[0]) + "_"*str(params.end_time[0]) + "_0627_results.json","wb") as f:
                pickle.dump(output, f)
            break
        elif matching_engine == "graph_lstm_dro":
            with open(output_path + str(params.start_time[0]) + "_" + str(params.end_time[0]) + f"_results_{args.CI}.json","wb") as f:
                pickle.dump(output, f)
        else:
            with open(output_path + str(ρ) + "_" + str(Γ) + "_" + str(params.start_time[0]) + "_" + str(params.end_time[0]) + f"_results_{args.CI}.json","wb") as f:
                pickle.dump(output, f)                
