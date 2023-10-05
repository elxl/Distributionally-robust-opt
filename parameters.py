import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime


class Parameters():
    def __init__(self, ci=50):

        nodes = [12,88,13,261,87,209,231,45, 125,211,144,148,232,158,249,113,114,79,4,246,68,90,234,107,224,137,186,164,170,100,233]

        self.zone_index_id = pd.read_csv("data/NYC/zone_index_id.csv", header=None).values
        self.zone_index_id_dict = dict()
        for i in range(self.zone_index_id.shape[0]):
            self.zone_index_id_dict[self.zone_index_id[i,1]] = self.zone_index_id[i,0]
        nodes_index = sorted([self.zone_index_id_dict[i] for i in nodes])
        self.zones_index = nodes_index
        
        self.start_time = (7, 0) # Hour, Minute
        self.end_time = (9, 0) # Hour, Minute
        
        self.time_interval_length = 300 # Seconds
        self.rebalancing_time_length = 900 # 6 time intervals
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
        data = data[(data['bin'] >= self.start_bin) & (data['bin'] <= self.end_bin) & (data['zone'].isin(nodes))]
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
        mask = np.isin(self.road_node_to_zone[:,1],nodes_index)
        self.road_node_to_zone = self.road_node_to_zone[mask, :]
        self.road_node = self.road_node_to_zone[:,0]
        self.zone_to_road_node_dict = defaultdict(list)
        self.road_node_to_zone_dict = dict()
        for i in range(self.road_node_to_zone.shape[0]):
            road_node_id = self.road_node_to_zone[i, 0]
            zone_id = self.road_node_to_zone[i, 1]
            self.road_node_to_zone_dict[road_node_id] = zone_id
            self.zone_to_road_node_dict[zone_id].append(road_node_id)
            
        self.zone_centriod_node = pd.read_csv("data/NYC/centroid_ind_node.csv", header=None).values
        mask = np.isin(self.zone_centriod_node[:,0],nodes_index)
        self.zone_centriod_node = self.zone_centriod_node[mask, :]
        self.centroid_to_node_dict = dict()
        for i in range(self.zone_centriod_node.shape[0]):
            self.centroid_to_node_dict[self.zone_centriod_node[i,0]] = self.zone_centriod_node[i,1]

        # Demand information used for solving optimization problems
        # self.demand_mean = np.load("historical/0627_poisson_mean.npy")[nodes_index,:]
        # self.demand_std = np.load("historical/0627_poisson_std.npy")[nodes_index,:]
        # self.demand_lb = np.load(f"historical/0627_poisson_{ci}_lb.npy")[nodes_index,:]
        # self.demand_ub = np.load(f"historical/0627_poisson_{ci}_ub.npy")[nodes_index,:]
        self.demand_mean = np.mean(self.data_points, axis=2)
        self.demand_std = np.std(self.data_points, axis=2)
        self.true_demand = June_27_data.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
            
        # Demand information used for solving optimizaiton problems
        self.demand_data = pd.read_csv("data/NYC/demand/fhv_records_06272019.csv")
        self.demand_data = self.demand_data[(self.demand_data['pu_zone'].isin(nodes) & (self.demand_data['do_zone'].isin(nodes)))].reset_index(drop=True)
        self.hist_avg_demand = June_27_data.loc[:, ['zone','bin','historical_average']].pivot(index='zone',columns='bin',values='historical_average').values
        
        # Problem Parameters
        self.β = 1
        self.γ = 1e2
        self.average_speed = 20
        self.maximum_waiting_time = 300    # seconds
        self.maximum_rebalancing_time = self.time_interval_length
        self.big_M = 1e5
        
        d = np.load("data/NYC/distance_matrix.npy")[nodes_index,:][:,nodes_index] # Zone centroid distances in miles
        self.d = np.repeat(d[:, :, np.newaxis], K, axis=2) # Repeat d to create a n x n x K matrix
        # Hourly travel time to 288 time intervals
        w_hourly = np.load("data/NYC/hourly_tt.npy")[nodes_index,:][:,nodes_index]
        a = np.repeat(w_hourly[:,:,0][:, :, np.newaxis], 12, axis=2)
        for i in range(1,24):
            b = np.repeat(w_hourly[:,:,i][:, :, np.newaxis], 12, axis=2)
            a = np.concatenate((a, b), axis=2)
        
        w = a * 3600
        w = w[:,:,self.start_bin:self.end_bin+1] # 7 AM to 9 AM travel time matrix
        
        self.a = (w > self.maximum_rebalancing_time)
        self.b = (w > self.maximum_waiting_time)
        
        P = np.load("data/NYC/p_matrix_occupied.npy")[nodes_index,:][:,nodes_index]
        Q = np.load("data/NYC/q_matrix_occupied.npy")[nodes_index,:][:,nodes_index]
        self.P = np.repeat(P[:,:,np.newaxis], K, axis=2)
        self.Q = np.repeat(Q[:,:,np.newaxis], K, axis=2)
        
        graph_lstm_mean = pd.read_csv("graph_lstm/0627_poisson_mean.csv")
        graph_lstm_mean = graph_lstm_mean[graph_lstm_mean['zone'].isin(nodes)]
        self.graph_lstm_mean = graph_lstm_mean.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
        graph_lstm_var = pd.read_csv("graph_lstm/0627_poisson_std.csv")
        graph_lstm_var = graph_lstm_var[graph_lstm_var['zone'].isin(nodes)]
        self.graph_lstm_var = graph_lstm_var.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values

        # Lower and upper bound from neural network
        graph_lstm_lb = pd.read_csv(f"graph_lstm/0627_poisson_{ci}_lb.csv")
        graph_lstm_lb = graph_lstm_lb[graph_lstm_lb['zone'].isin(nodes)]
        self.graph_lstm_lb = graph_lstm_lb.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
        graph_lstm_ub = pd.read_csv(f"graph_lstm/0627_poisson_{ci}_ub.csv")
        graph_lstm_ub = graph_lstm_ub[graph_lstm_ub['zone'].isin(nodes)]
        self.graph_lstm_ub = graph_lstm_ub.loc[:, ['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values