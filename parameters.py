import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime


class Parameters():
    def __init__(self):
        
        self.start_time = (7, 0) # Hour, Minute
        self.end_time = (9, 0) # Hour, Minute
        
        self.time_interval_length = 300 # Seconds
        self.rebalancing_time_length = 1200 # 6 time intervals
        self.matching_window = 30 # Seconds
        data1 = pd.read_csv('data/NYC/road_network/distance.csv', header=None) / 1609.34
        self.road_distance_matrix = data1.values
        data2 = pd.read_csv('data/NYC/road_network/predecessor.csv', header=None)
        self.predecessor = data2.values
        
        self.start_bin = self.start_time[0] * 12
        self.end_bin = self.end_time[0] * 12 - 1
        self.start_timestamp = datetime(2019,6,27,self.start_time[0],0,0)
        self.end_timestamp = datetime(2019,6,27,self.end_time[0],0,0)
        
        # Demand
        data = pd.read_csv("data/NYC/processed_data/normalized_data.csv")
        data = data[(data['bin'] >= self.start_bin) & (data['bin'] <= self.end_bin)]
        prev_data = data[(data['month'] !=6) | (data['day'] < 27)]
        June_27_data = data[(data['month'] ==6) & (data['day'] == 27)]
        
        gd = prev_data.groupby(['day','month'])
        self.n = len(data['zone'].unique())
        K = len(data['bin'].unique())
        m = len(gd)
        self.data_points = np.zeros((self.n, K, m))
        index = 0
        for _, df in gd:
            y_i = df.loc[:,['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
            self.data_points[:,:,index] = y_i
            index += 1
            
        self.road_node_to_zone = pd.read_csv('data/NYC/road_node_to_zone.csv', header=None).values
        self.zone_to_road_node_dict = defaultdict(list)
        self.road_node_to_zone_dict = dict()
        for i in range(self.road_node_to_zone.shape[0]):
            road_node_id = self.road_node_to_zone[i, 0]
            zone_id = self.road_node_to_zone[i, 1]
            self.road_node_to_zone_dict[road_node_id] = zone_id
            self.zone_to_road_node_dict[zone_id].append(road_node_id)
            
        self. zone_centriod_node = pd.read_csv("data/NYC/centroid_ind_node.csv", header=None).values
        self.centroid_to_node_dict = dict()
        for i in range(self.zone_centriod_node.shape[0]):
            self.centroid_to_node_dict[self.zone_centriod_node[i,0]] = self.zone_centriod_node[i,1]
        
        self.zone_index_id = pd.read_csv("data/NYC/zone_index_id.csv", header=None).values
        self.zone_index_id_dict = dict()
        for i in range(self.zone_index_id.shape[0]):
            self.zone_index_id_dict[self.zone_index_id[i,1]] = self.zone_index_id[i,0]

        # Demand information used for solving optimization problems
        self.demand_mean = np.load("historical/0627_normal_mean.npy")
        self.demand_std = np.load("historical/0627_normal_std.npy")
        self.demand_lb = np.load("historical/0627_normal_95_lb.npy")
        self.demand_ub = np.load("historical/0627_normal_95_ub.npy")
        self.true_demand = June_27_data.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
            
        # Demand information used for solving optimizaiton problems
        self.demand_data = pd.read_csv("data/NYC/demand/fhv_records_06272019.csv")
        
        # Problem Parameters
        self.β = 1
        self.γ = 1e2
        self.average_speed = 20
        self.maximum_waiting_time = 300    # seconds
        self.maximum_rebalancing_time = self.time_interval_length
        self.big_M = 1e5
        
        d = np.load("data/NYC/distance_matrix.npy") # Zone centroid distances in miles
        self.d = np.repeat(d[:, :, np.newaxis], K, axis=2) # Repeat d to create a n x n x K matrix
        # Hourly travel time to 288 time intervals
        w_hourly = np.load("data/NYC/hourly_tt.npy")
        a = np.repeat(w_hourly[:,:,0][:, :, np.newaxis], 12, axis=2)
        for i in range(1,24):
            b = np.repeat(w_hourly[:,:,i][:, :, np.newaxis], 12, axis=2)
            a = np.concatenate((a, b), axis=2)
        
        w = a * 3600
        w = w[:,:,self.start_bin:self.end_bin+1] # 7 AM to 9 AM travel time matrix
        
        self.a = (w > self.maximum_rebalancing_time)
        self.b = (w > self.maximum_waiting_time)
        
        P = np.load("data/NYC/p_matrix_occupied.npy")
        Q = np.load("data/NYC/q_matrix_occupied.npy")
        self.P = np.repeat(P[:,:,np.newaxis], K, axis=2)
        self.Q = np.repeat(Q[:,:,np.newaxis], K, axis=2)
        
        graph_lstm_mean = pd.read_csv("graph_lstm/0627_poisson_mean.csv")
        self.graph_lstm_mean = graph_lstm_mean.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
        graph_lstm_var = pd.read_csv("graph_lstm/0627_poisson_std.csv")
        self.graph_lstm_var = graph_lstm_var.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values

        # Lower and upper bound from neural network
        graph_lstm_lb = pd.read_csv("graph_lstm/0627_poisson_95_lb.csv")
        self.graph_lstm_lb = graph_lstm_lb.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
        graph_lstm_ub = pd.read_csv("graph_lstm/0627_poisson_95_ub.csv")
        self.graph_lstm_ub = graph_lstm_ub.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values