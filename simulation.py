# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:27:05 2023

@author: 11481
"""

import random, math, pickle
import numpy as np
from parameters import demand_data, zone_to_road_node_dict, zone_index_id_dict, road_node_to_zone_dict, road_distance_matrix,\
                       time_interval_length, rebalancing_time_length, start_time, end_time, average_speed, matching_window, maximum_waiting_time,\
                       P, Q, a, b, d, n, β, γ
from functions import initialize_demand, initialize_vehicle, get_current_location, matching, robust_model_function, robust_model_function_interval
from collections import defaultdict
from datetime import datetime, timedelta


fleet_size = 2000
matching_engine = "graph_lstm"
output_path = "output/graph_lstm_poisson_nb_0627/"
ρ_list = [3]
Γ_list = [0]

for ρ in ρ_list, Γ in Γ_list:

    # Initialize demand and vehicle objects
    demand_list, demand_id_dict = initialize_demand(demand_data, zone_to_road_node_dict, zone_index_id_dict)
    vehicle_list, vehicle_id_dict = initialize_vehicle(fleet_size, n, zone_to_road_node_dict)

    simulation_start_time = datetime(2019,6,27,start_time[0],0,0)
    simulation_end_time = datetime(2019,6,27,end_time[0],0,0)
    simulation_time = simulation_start_time
        
    while True:
        if simulation_time >= simulation_end_time:
            break

        time_index = int((simulation_time - simulation_start_time).total_seconds() / time_interval_length)
        number_of_intervals = int(rebalancing_time_length / time_interval_length) # number of intervals in one optimization step
        end_time_index = time_index + min(d[:, :, time_index:].shape[2], number_of_intervals) # end time index for current optimization step

        P_matrix = P[:,:,time_index:end_time_index] # Probility of vehicle staying occupied
        Q_matrix = Q[:,:,time_index:end_time_index] # Probability of occupied vehicle becomes vacant

        # Find initial occupied & vacant vehicle distributions
        V_init = np.zeros(n) # vacant vehicles
        O_init = np.zeros(n) # occupied vehicles
        zone_vacant_veh_dict = defaultdict(list)
        for veh in vehicle_list:
            veh_loc = veh.current_location
            vehicle_zone = road_node_to_zone_dict[veh_loc]
            if veh.occupied:
                O_init[vehicle_zone] += 1 # zone index from 0 to 62
            else:
                V_init[vehicle_zone] += 1
                zone_vacant_veh_dict[vehicle_zone].append(veh.id)

        print("Rebalancing Phase ")
        print(simulation_time)

        K_sub = end_time_index - time_index + 1
        a_sub = a[:,:,time_index:end_time_index] # if traveling time is bigger than rebalancing threshold
        b_sub = b[:,:,time_index:end_time_index] # if traveling time is bigger than maximum waiting time
        d_sub = d[:,:,time_index:end_time_index] # zone centroids distance 

        if matching_engine == "graph_lstm":
            μ = graph_lstm_mean[:, time_index:end_time_index] # predicted mean
            σ = graph_lstm_var[:, time_index:end_time_index] # predicted standard deviation
            rebalancing_decision = robust_model_function(μ, σ, ρ, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)
        elif matching_engine == "graph_lstm_interval":
            μ = graph_lstm_mean[:, time_index:end_time_index] # predicted mean
            lb = graph_lstm_lb[:, time_index:end_time_index]
            ub = graph_lstm_ub[:, time_index:end_time_index]
            rebalancing_decision = robust_model_function_interval(μ, lb, ub, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)          
        elif matching_engine == "historical":
            μ = demand_mean[:, time_index:end_time_index]
            σ = demand_std[:, time_index:end_time_index]
            rebalancing_decision = robust_model_function(μ, σ, ρ, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)
        elif matching_engine == "historical_interval":
            μ = demand_mean[:, time_index:end_time_index] # predicted mean
            lb = demand_lb[:, time_index:end_time_index]
            ub = demand_ub[:, time_index:end_time_index]
            rebalancing_decision = robust_model_function_interval(μ, lb, ub, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)            
        # elif matching_engine == "true_demand":
        #     r = true_demand[:, time_index:end_time_index]
        #     rebalancing_decision = optimization(r, V_init, O_init, P_matrix, Q_matrix, n, K_sub, a_sub, b_sub, d_sub, β, γ)

        rebalancing_decision = (Int.(floor.(rebalancing_decision[:,:,1])))

        # Rebalancing vacant vehicles
        for i in range(n):
            for j in range(n):
                rebalancing_veh_number = rebalancing_decision[i,j]
                if rebalancing_veh_number <= 0:
                    continue
                #Random.seed!(2020)
                rebalancing_veh_list = random.sample(zone_vacant_veh_dict[i], rebalancing_veh_number)
                for veh_id in rebalancing_veh_list:
                    veh = vehicle_id_dict[veh_id]
                    random_number = 0
                    Flag = False
                    #Random.seed!(2020)
                    while True:
                        dest_node = random.choice(zone_to_road_node_dict[j])
                        rebalancing_dist = road_distance_matrix[veh.current_location, dest_node]
                        rebalancing_time = (rebalancing_dist / average_speed) * 3600
                        if rebalancing_time <= time_interval_length:
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
            if simulation_time <= veh.available_time < simulation_time + timedelta(seconds=time_interval_length):
                veh.current_location = veh.location

        # Matching engine in the simulation
        matching_simulation_time = simulation_time
        while True:
            print("Matching phase ")
            print(f"{matching_simulation_time}")
            if matching_simulation_time >= simulation_time + timedelta(seconds=time_interval_length):
                break

            available_vehicles = []
            for veh in vehicle_list:
                if veh.available_time < matching_simulation_time + timedelta(seconds=matching_window):
                    available_vehicles.append(veh)

            requesting_demands = []
            for dem in demand_list:
                if simulation_start_time <= dem.request_time < matching_simulation_time + timedelta(seconds=matching_window):
                    if dem.assign_time is None:
                        if not dem.leave_system:
                            requesting_demands.append(dem)

            matching_list, unserved_pax_list = matching(available_vehicles, requesting_demands)

            # Update Passengers not matched
            for pax_id in unserved_pax_list:
                pax = demand_id_dict[pax_id]
                pax.wait_time += matching_window
                if pax.wait_time >= maximum_waiting_time:
                    pax.leave_system = True

            for ((veh_id, pax_id), pickup_dist) in matching_list:
                pax = demand_id_dict[pax_id]
                pickup_time = pickup_dist / average_speed * 3600
                pax.wait_time = pickup_time + matching_window + matching_simulation_time.total_seconds() - pax.request_time.total_seconds()
                pax.travel_time = road_distance_matrix[pax.origin, pax.destination] / average_speed * 3600
                pax.assign_time = matching_simulation_time + timedelta(seconds=matching_window)
                pax.arrival_time = pax.assign_time + timedelta(seconds=math.floor(pickup_time)) + timedelta(seconds=math.floor(pax.travel_time))

                veh = vehicle_id_dict[veh_id]
                veh.location = pax.destination
                veh.available_time = pax.arrival_time
                veh.served_passenger.append(pax.id)
                veh.pickup_travel_distance.append(pickup_dist)
                veh.occupied_travel_distance.append(road_distance_matrix[pax.origin, pax.destination])

            matching_simulation_time += timedelta(seconds=matching_window)

        # Update vehicle status for next rebalancing time window (availability and position when next time window starts)
        matching_time = matching_simulation_time
        for veh in vehicle_list:
            if matching_time <= veh.available_time:
                veh.occupied = True
                veh.current_location = get_current_location(matching_time, veh, demand_id_dict)
            else:
                veh.occupied = False
                veh.current_location = veh.location

        simulation_time += timedelta(seconds=time_interval_length)

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

    if matching_engine=="true_demand":
        with open(output_path*matching_engine*"_"*str(start_time[0])*"_"*str(end_time[0])*"_0627_results.json","w") as f:
            pickle.dump(output, f)
        break
    else:
        with open(output_path*str(ρ)*"_"*str(Γ)*"_"*str(start_time[0])*"_"*str(end_time[0])*"_results_95_95.json","w") as f:
            pickle.dump(output, f)
