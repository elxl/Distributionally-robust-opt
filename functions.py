# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:40:41 2023

@author: 11481
"""
import random, math
import numpy as np
import gurobipy as gb
from datetime import datetime, timedelta
from parameters import start_timestamp, end_timestamp, start_time, matching_window, \
                        maximum_waiting_time, predecessor, road_distance_matrix, average_speed, BIG_M
from structures import Passenger, Vehicle

from rsome import dro, ro
from rsome import norm
from rsome import E
from rsome import grb_solver as grb


def initialize_demand(demand_data, zone_to_road_node_dict, zone_index_id_dict):
    """
    Generate passengers according to real data.

    zone_to_road_node_dict: included nodes of each zone
    zone_index_id_dict: zone id (4-263) to zone index (0-62)

    return: list of passengers, dictionary of passengers
    """
    demand_list = []
    demand_id_dict = dict()
    demand_ind = 0

    random.seed(2023)

    for i in range(demand_data.shape[0]):
        request_time = datetime.strptime(demand_data.loc[i,"pu_time"], "%Y-%m-%d %H:%M:%S")
        if (request_time >= start_timestamp) & (request_time < end_timestamp):
            pickup_zone = zone_index_id_dict[int(demand_data.loc[i, "pu_zone"])]
            dropoff_zone = zone_index_id_dict[int(demand_data.loc[i, "do_zone"])]
            while True:
                pickup_node = random.sample(zone_to_road_node_dict[pickup_zone], 1)[0]
                dropoff_node = random.sample(zone_to_road_node_dict[dropoff_zone], 1)[0]
                if pickup_node != dropoff_node:
                    break

            pax = Passenger(demand_ind, pickup_node, dropoff_node, request_time, None, 0.0, 0.0, None, False)
            demand_id_dict[demand_ind] = pax
            demand_list.append(pax)
            demand_ind += 1

    return demand_list, demand_id_dict


def initialize_vehicle(fleet_size, n, zone_to_road_node_dict):
    """
    Initialize vechiles.

    return: list of vehicles, dictionary of vehicles
    """
    vehicle_list = []
    vehicle_id_dict = dict()
    vehicle_ind = 0
    init_avail_time = datetime(2019,6,27,start_time[0],0,0)
    random.seed(2023)

    zone_vehicle_number = int(math.floor(fleet_size / n)) # number of vehicles in each zone
    for i in range(n):
        road_node_list = zone_to_road_node_dict[i]
        vehicle_loc_list = random.choices(road_node_list, k=zone_vehicle_number) # sample number of vehicles locations in zone
        for loc in vehicle_loc_list:
            veh = Vehicle(vehicle_ind, loc, init_avail_time, loc, False, [], [], [], [], [])
            vehicle_id_dict[vehicle_ind] = veh
            vehicle_list.append(veh)
            vehicle_ind += 1

    return vehicle_list, vehicle_id_dict

def matching(vehicles, demands):
    """
    Match available vechiles and requests.

    vehicles: list
    demands: list

    return: matched vehicle and passenger, unserved passenger
    """
    pickup_dist = {}
    # All possible matching schemes
    for veh in vehicles:
        veh_id = veh.id
        veh_loc = veh.current_location
        for dem in demands:
            dem_id = dem.id
            dem_loc = dem.origin
            pick_distance = road_distance_matrix[veh_loc, dem_loc]
            pickup_time = pick_distance / average_speed
            if pickup_time * 3600 <= maximum_waiting_time:
                pickup_dist[(veh_id, dem_id)] = pick_distance

    veh_dict = {veh.id: [] for veh in vehicles}
    dem_dict = {dem.id: [] for dem in demands}

    for i in pickup_dist:
        veh_dict[i[0]].append(i)
        dem_dict[i[1]].append(i)

    # Matching optimization
        m = gb.Model("matching")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 10)
        m.setParam('MIPGap', 0.01)

        # Decision variables
        x = {}
        for i in pickup_dist:
            x[i] = m.addVar(vtype='B', name="matching_" + str(i))

        # Decision variables for unserved matching
        y = {}
        for dem in dem_dict:
            y[dem] = m.addVar(vtype='B', name="unserved_" + str(dem))

        # Constraints
        m.addConstrs(gb.quicksum(x[i] for i in veh_dict[veh]) <= 1 for veh in veh_dict)
        m.addConstrs(gb.quicksum(x[i] for i in dem_dict[dem]) + y[dem] == 1 for dem in dem_dict)

        # Objectives
        m.setObjective(BIG_M * gb.quicksum(y[i] for i in dem_dict)
                       + gb.quicksum(x[i] * pickup_dist[i] for i in pickup_dist), gb.GRB.MINIMIZE)
        m.optimize()

    # Get matched vehicle and passenger as well as the pick up distance
    matching_list = []
    for i, var in x.items():
        if var.X == 1:
            matching_list.append((i, pickup_dist[i]))

    unserved_pax_list = []
    for i, var in y.items():
        if var.X == 1:
            unserved_pax_list.append(i)

    return matching_list, unserved_pax_list


def get_current_location(matching_time, veh, demand_id_dict):
    """
    demand_id_dict: id:passenger
    """
    pax = demand_id_dict[veh.served_passenger[-1]]

    if matching_time - timedelta(seconds=matching_window) < pax.assign_time <= matching_time:
        return veh.current_location
    else:
        vehicle_travel_time = matching_time.timestamp - pax.request_time.timestamp - pax.wait_time
        veh_start_loc = pax.origin
        veh_end_loc = pax.destination
        trip_path = []
        temp_node = veh_end_loc
        # Path of vehicle
        while True:
            pred = int(predecessor[veh_start_loc,temp_node])
            trip_path.insert(0, pred)
            if pred == veh_start_loc:
                break

            temp_node = pred
        trip_path.append(veh_end_loc)

    travel_time = 0
    for i in range(len(trip_path) - 1):
        start_node = trip_path[i]
        end_node = trip_path[i+1]
        segment_dist = road_distance_matrix[start_node,end_node]
        segment_time = segment_dist / average_speed * 3600
        travel_time += segment_time
        if travel_time >= vehicle_travel_time:
            return end_node


def DRO(data, V1, O1, P, Q, d, a, b, 𝛽, γ):
    """
    Solve problem by robust optimization.
    """
    n, K, sample = data.shape
    
    dhat = data
    d_ub = np.max(dhat, axis=2) # upper bound of demand
    d_lb = np.min(dhat, axis=2) # lower bound of demand
    S = sample
    epsilon = 0.25 # parameter of robustness
    w = 1/S # weights of scenarios
    
    dro_model = dro.Model(S)
    
    # Uncertainty
    dset = dro_model.rvar((n, K))
    fset = dro_model.ambiguity()
    for s in range(S):                                  # for each scenario
        fset[s].suppset(dset <= d_ub, dset >= d_lb,
                        norm(dset.reshape(n*K) - dhat[:,:,s].reshape(n*K)) <= epsilon)   # define the support set
        pr = dro_model.p                                    # an array of scenario weights
        fset.probset(pr == w)                           # specify scenario weights
    
    # Decision variables
    x = dro_model.dvar((n, n, K))
    y = dro_model.dvar((n, n, K))
    y.adapt(dset)
    
    O = dro_model.dvar((n, K))
    V = dro_model.dvar((n, K))
    S = dro_model.dvar((n, K))
    
    # Constraints
    # Set constraints to force initial position of occupied and vacant vehicles
    dro_model.st(V[i, 0] == V1[i] for i in range(n))
    dro_model.st(O[i, 0] == O1[i] for i in range(n))
    
    # Set constraints related to state transitions (1,2,3)
    dro_model.st(S[i,k] == V[i,k] + sum(x[j,i,k] for j in range(n)) - sum(x[i,j,k] for j in range(n)) for i in range(n) for k in range(K))
    dro_model.st(V[i,k+1] == S[i,k] - sum(y[j,i,k] for j in range(n)) + sum(Q[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
    dro_model.st(O[i,k+1] == sum(y[j,i,k] for j in range(n)) + sum(P[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))

    dro_model.st(x.sum(axis=1) <= V)
    
    # Set rebalancing constraint (4)
    dro_model.st(int(a[i,j,k]) * x[i,j,k] == 0 for i in range(n) for j in range(n) for k in range(K))

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    dro_model.st(y.sum(axis=0) <= S)
    dro_model.st(y.sum(axis=1) <= dset)

    # Set matching constraint (10)
    dro_model.st(int(b[i,j,k]) * y[i,j,k] == 0 for i in range(n) for j in range(n) for k in range(K))
    
    # Positive variable constraints
    dro_model.st(x >= 0,y >= 0,O >= 0,S >= 0,V >=0)


    # Set objective
    # dro_model.minsup(sum(x[i,j,k] * d[i,j,k] for i in range(n) for j in range(n) for k in range(K))
    #                         + 𝛽 * sum(y[i,j,k] * d[j,i,k] for i in range(n) for j in range(n) for k in range(K))
    #                         + γ * E(sum(dset[i,k]-sum(y[i,j,k] for j in range(n)) for i in range(n) for k in range(K))), fset)
    
    dro_model.minsup((x * d).sum()
                     + E(𝛽 * (y * np.transpose(d, [1,0,2])).sum()
                     + γ * (y.sum())), fset)


    dro_model.solve(grb)

    x_robust = x.get()
    
    return x_robust


def RO_random_sample(data, V1, O1, P, Q, d, a, b, 𝛽, γ):
    """
    Solve problem by robust optimization.
    """
    n, K, sample = data.shape
    
    dhat = data
    d_ub = np.max(dhat, axis=2) # upper bound of demand
    d_lb = np.min(dhat, axis=2) # lower bound of demand
    S = sample
    epsilon = 0.25 # parameter of robustness
    
    ro_model = ro.Model()
    
    # Uncertainty
    dset = ro_model.rvar((n, K))
    # Decision variables
    x = ro_model.dvar((n, n, K))
    y = ro_model.ldr((S, n, n, K))
    y.adapt(dset)
    
    w = ro_model.dvar(sample)
    O = ro_model.dvar((n, K))
    V = ro_model.dvar((n, K))
    S = ro_model.dvar((n, K))
       
    # Constraints
    # Set constraints to force initial position of occupied and vacant vehicles
    ro_model.st(V[i, 0] == V1[i] for i in range(n))
    ro_model.st(O[i, 0] == O1[i] for i in range(n))
    
    # Set constraints related to state transitions (1,2,3)
    # ro_model.st(S[i,k] == V[i,k] + sum(x[j,i,k] for j in range(n)) - sum(x[i,j,k] for j in range(n)) for i in range(n) for k in range(K))
    ro_model.st(S == V + x.sum(axis=0) - x.sum(axis=1))

    # ro_model.st(sum(x[i,j,k] for j in range(n)) <= V[i,k] for i in range(n) for k in range(K))
    ro_model.st(x.sum(axis=1) <= V)
    
    # Set rebalancing constraint (4)
    # ro_model.st(int(a[i,j,k]) * x[i,j,k] == 0 for i in range(n) for j in range(n) for k in range(K))
    ro_model.st(a.astype(int) * x == 0)

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    # ro_model.st(sum(y[j,i,k] for j in range(n)) <= S[i,k] for i in range(n) for k in range(K))
    for s in range(sample):                                  # for each scenario
        zset = (dset <= d_ub, dset >= d_lb,
                        (dset.reshape(n*K) - dhat[:,:,s].reshape(n*K)) <= epsilon)   # define the support set
        
        # ro_model.st(V[i,k+1] == S[i,k] - sum(y[j,i,k] for j in range(n)) + sum(Q[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
        ro_model.st(V[:,1:K] == S - y[s,:,:,:(K-1)].sum(axis=0) + (Q * O)[:,:,:(K-1)].sum(axis=0))
        # ro_model.st(O[i,k+1] == sum(y[j,i,k] for j in range(n)) + sum(P[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
        ro_model.st(O[:,1:K] == y[s,:,:,:(K-1)].sum(axis=0) + (P * O)[:,:,:(K-1)].sum(axis=0))
        
        ro_model.st((y[s].sum(axis=0) <= S).forall(zset))
        ro_model.st((w[s] >= 𝛽 * (y[s] * np.transpose(d, [1,0,2])).sum() + γ * (dset - y.sum(axis=1)).sum()).forall(zset))
        ro_model.st((y[s].sum(axis=1) <= dset).forall(zset))
        ro_model.st((y[s] >= 0).forall(zset))

        # Set matching constraint (10)
        ro_model.st(b.astype(int) * y[s] == 0)
    
    # Positive variable constraints
    ro_model.st(x >= 0,O >= 0,S >= 0,V >=0)


    # Set objective
    # dro_model.minsup(sum(x[i,j,k] * d[i,j,k] for i in range(n) for j in range(n) for k in range(K))
    #                         + 𝛽 * sum(y[i,j,k] * d[j,i,k] for i in range(n) for j in range(n) for k in range(K))
    #                         + γ * E(sum(dset[i,k]-sum(y[i,j,k] for j in range(n)) for i in range(n) for k in range(K))), fset)
    
    ro_model.min((x * d).sum() + (1/sample)*w.sum())


    ro_model.solve(grb)

    x_robust = x.get()
    
    return x_robust


def RO(data, V1, O1, P, Q, d, a, b, 𝛽, γ):
    """
    Solve problem by robust optimization.
    """
    n, K, sample = data.shape
    
    dhat = data
    d_ub = np.max(dhat, axis=2) # upper bound of demand
    d_lb = np.min(dhat, axis=2) # lower bound of demand
    S = sample
    epsilon = 0.25 # parameter of robustness
    
    ro_model = ro.Model()
    
    # Uncertainty
    dset = ro_model.rvar((n, K))
    # Decision variables
    x = ro_model.dvar((n, n, K))
    y = ro_model.ldr((S, n, n, K))
    y.adapt(dset)
    
    w = ro_model.dvar(sample)
    O = ro_model.dvar((n, K))
    V = ro_model.dvar((n, K))
    S = ro_model.dvar((n, K))
       
    # Constraints
    # Set constraints to force initial position of occupied and vacant vehicles
    ro_model.st(V[i, 0] == V1[i] for i in range(n))
    ro_model.st(O[i, 0] == O1[i] for i in range(n))
    
    # Set constraints related to state transitions (1,2,3)
    # ro_model.st(S[i,k] == V[i,k] + sum(x[j,i,k] for j in range(n)) - sum(x[i,j,k] for j in range(n)) for i in range(n) for k in range(K))
    ro_model.st(S == V + x.sum(axis=0) - x.sum(axis=1))
    # ro_model.st(V[i,k+1] == S[i,k] - sum(y[j,i,k] for j in range(n)) + sum(Q[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
    ro_model.st(V[:,1:K] == S - y[:,:,:(K-1)].sum(axis=0) + (Q * O)[:,:,:(K-1)].sum(axis=0))
    # ro_model.st(O[i,k+1] == sum(y[j,i,k] for j in range(n)) + sum(P[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
    ro_model.st(O[:,1:K] == y[:,:,:(K-1)].sum(axis=0) + (P * O)[:,:,:(K-1)].sum(axis=0))

    # ro_model.st(sum(x[i,j,k] for j in range(n)) <= V[i,k] for i in range(n) for k in range(K))
    ro_model.st(x.sum(axis=1) <= V)
    
    # Set rebalancing constraint (4)
    # ro_model.st(int(a[i,j,k]) * x[i,j,k] == 0 for i in range(n) for j in range(n) for k in range(K))
    ro_model.st(a.astype(int) * x == 0)

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    # ro_model.st(sum(y[j,i,k] for j in range(n)) <= S[i,k] for i in range(n) for k in range(K))
    for s in range(sample):                                  # for each scenario
        zset = (dset <= d_ub, dset >= d_lb,
                        norm(dset.reshape(n*K) - dhat[:,:,s].reshape(n*K)) <= epsilon)   # define the support set
        ro_model.st((y[s].sum(axis=0) <= S).forall(zset))
        ro_model.st((w[s] >= γ * sum((dset[i,k] - sum(y[i,j,k] for j in range(n))) for i in range(n) for k in range(K))).forall(zset))
        ro_model.st((y[s].sum(axis=1) <= dset).forall(zset))
        ro_model.st((y[s] >= 0).forall(zset))

        # Set matching constraint (10)
        ro_model.st(b.astype(int) * y[s] == 0)
    
    # Positive variable constraints
    ro_model.st(x >= 0,O >= 0,S >= 0,V >=0)


    # Set objective
    # dro_model.minsup(sum(x[i,j,k] * d[i,j,k] for i in range(n) for j in range(n) for k in range(K))
    #                         + 𝛽 * sum(y[i,j,k] * d[j,i,k] for i in range(n) for j in range(n) for k in range(K))
    #                         + γ * E(sum(dset[i,k]-sum(y[i,j,k] for j in range(n)) for i in range(n) for k in range(K))), fset)
    
    ro_model.min((x * d).sum()
                 + 𝛽 * (y * np.transpose(d, [1,0,2])).sum()
                 + (1/sample)*w.sum())


    ro_model.solve(grb)

    x_robust = x.get()
    
    return x_robust

def robust_model_function(μ, σ, ρ, Γ, V1, O1, P, Q, d, a, b, β, γ):
    """ 
    Robust optimization with mean, variance and manually set interval.
    """
    n = d.shape[1]
    K = μ.shape[1]

    model = ro.Model()

    # Decision variables
    x = model.dvar((n, n, K))
    y = model.dvar((n, n, K))
    O = model.dvar((n, K))
    V = model.dvar((n, K))
    S = model.dvar((n, K))
    w = model.dvar()

    # Positive variable constraints
    model.st(x >= 0,O >= 0,S >= 0,V >=0)

    # Uncertainty variable
    zeta = model.rvar((n, K))

    # Set constraints to force initial position of occupied and vacant vehicles
    model.st(V[:, 0] == V1)
    model.st(O[:, 0] == O1)

    # Uncertainty
    z_set = (abs(zeta) <= ρ, abs(zeta @ σ) <= Γ)

    model.st((μ + zeta * σ >= 0).forall(z_set))

    # Set constraints related to state transitions (1,2,3)
    # model.st(S[i,k] == V[i,k] + sum(x[j,i,k] for j in range(n)) - sum(x[i,j,k] for j in range(n)) for i in range(n) for k in range(K))
    model.st(S == V + x.sum(axis=0) - x.sum(axis=1))
    # model.st(V[i,k+1] == S[i,k] - sum(y[j,i,k] for j in range(n)) + sum(Q[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
    model.st(V[:,1:K] == S - y[:,:,:(K-1)].sum(axis=0) + (Q * O)[:,:,:(K-1)].sum(axis=0))
    # model.st(O[i,k+1] == sum(y[j,i,k] for j in range(n)) + sum(P[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
    model.st(O[:,1:K] == y[:,:,:(K-1)].sum(axis=0) + (P * O)[:,:,:(K-1)].sum(axis=0))    

    # Set rebalancing constraint (4)
    # model.st(int(a[i,j,k]) * x[i,j,k] == 0 for i in range(n) for j in range(n) for k in range(K))
    model.st(a.astype(int) * x == 0)

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    # model.st(sum(y[j,i,k] for j in range(n)) <= S[i,k] for i in range(n) for k in range(K))
    model.st((y.sum(axis=0) <= S))
    model.st((y.sum(axis=0) <= μ + zeta * σ).forall(z_set))

    # Set matching constraint (10)
    model.st(b.astype(int) * y == 0)

    model.st(((x * d).sum()
                + 𝛽 * (y * np.transpose(d, [1,0,2])).sum()
                + γ * (μ + zeta * σ - y.sum(axis=1)).sum() <= w).forall(z_set))
    
    model.min(w)
    model.solve(grb)

    x_robust = x.get()

    return x_robust

def robust_model_function_interval(μ, lb, ub, Γ, V1, O1, P, Q, d, a, b, β, γ):
    """ 
    Robust optimization with mean, variance and provided interval.
    """    

    n = d.shape[1]
    K = μ.shape[1]

    model = ro.Model()

    # Decision variables
    x = model.dvar((n, n, K))
    y = model.dvar((n, n, K))
    O = model.dvar((n, K))
    V = model.dvar((n, K))
    S = model.dvar((n, K))
    w = model.dvar()

    # Positive variable constraints
    model.st(x >= 0,O >= 0,S >= 0,V >=0)

    # Uncertainty variable
    r = model.rvar((n, K))

    # Set constraints to force initial position of occupied and vacant vehicles
    model.st(V[:, 0] == V1)
    model.st(O[:, 0] == O1)

    # Uncertainty
    z_set = (r <= ub, r>= lb, r>=0, abs((r-μ).sum()) <= Γ)

    # Set constraints related to state transitions (1,2,3)
    # model.st(S[i,k] == V[i,k] + sum(x[j,i,k] for j in range(n)) - sum(x[i,j,k] for j in range(n)) for i in range(n) for k in range(K))
    model.st(S == V + x.sum(axis=0) - x.sum(axis=1))
    # model.st(V[i,k+1] == S[i,k] - sum(y[j,i,k] for j in range(n)) + sum(Q[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
    model.st(V[:,1:K] == S - y[:,:,:(K-1)].sum(axis=0) + (Q * O)[:,:,:(K-1)].sum(axis=0))
    # model.st(O[i,k+1] == sum(y[j,i,k] for j in range(n)) + sum(P[j,i,k] * O[j,k] for j in range(n)) for i in range(n) for k in range(K-1))
    model.st(O[:,1:K] == y[:,:,:(K-1)].sum(axis=0) + (P * O)[:,:,:(K-1)].sum(axis=0))    

    # Set rebalancing constraint (4)
    # model.st(int(a[i,j,k]) * x[i,j,k] == 0 for i in range(n) for j in range(n) for k in range(K))
    model.st(a.astype(int) * x == 0)

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    # model.st(sum(y[j,i,k] for j in range(n)) <= S[i,k] for i in range(n) for k in range(K))
    model.st((y.sum(axis=0) <= S))
    model.st((y.sum(axis=0) <= r).forall(z_set))

    # Set matching constraint (10)
    model.st(b.astype(int) * y == 0)

    model.st(((x * d).sum()
                + 𝛽 * (y * np.transpose(d, [1,0,2])).sum()
                + γ * (r - y.sum(axis=1)).sum() <= w).forall(z_set))
    
    model.min(w)
    model.solve(grb)

    x_robust = x.get()

    return x_robust

def Distributionally_robust_model(data, V1, O1, P, Q, d, a, b, 𝛽, γ):
    """
    Distributional robust optimization
    """
    