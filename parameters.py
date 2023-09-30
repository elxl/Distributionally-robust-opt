import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime


start_time = (7, 0) # Hour, Minute
end_time = (9, 0) # Hour, Minute

time_interval_length = 300 # Seconds
rebalancing_time_length = 1800 # 6 time intervals
matching_window = 30 # Seconds
data1 = pd.read_csv('data/NYC/road_network/distance.csv', header=None) / 1609.34
road_distance_matrix = data1.values
data2 = pd.read_csv('data/NYC/road_network/predecessor.csv', header=None)
predecessor = data2.values

start_bin = start_time[0] * 12
end_bin = end_time[0] * 12 - 1
start_timestamp = datetime(2019,6,27,start_time[0],0,0)
end_timestamp = datetime(2019,6,27,end_time[0],0,0)

# Demand
data = pd.read_csv("data/NYC/processed_data/normalized_data.csv")
data = data[(data['bin'] >= start_bin) & (data['bin'] <= end_bin)]
prev_data = data[(data['month'] !=6) | (data['day'] < 27)]
June_27_data = data[(data['month'] ==6) & (data['day'] == 27)]

gd = prev_data.groupby(['day','month'])
n = len(data['zone'].unique())
K = len(data['bin'].unique())
m = len(gd)
data_points = np.zeros((n, K, m))
index = 0
for _, df in gd:
    y_i = df.loc[:,['zone','bin','demand']].pivot(index='zone',columns='bin',values='demand').values
    data_points[:,:,index] = y_i
    index += 1
    
road_node_to_zone = pd.read_csv('data/NYC/road_node_to_zone.csv', header=None).values
zone_to_road_node_dict = defaultdict(list)
road_node_to_zone_dict = dict()
for i in range(road_node_to_zone.shape[0]):
    road_node_id = road_node_to_zone[i, 0]
    zone_id = road_node_to_zone[i, 1]
    road_node_to_zone_dict[road_node_id] = zone_id
    zone_to_road_node_dict[zone_id].append(road_node_id)
    
zone_centriod_node = pd.read_csv("data/NYC/centroid_ind_node.csv", header=None).values
centroid_to_node_dict = dict()
for i in range(zone_centriod_node.shape[0]):
    centroid_to_node_dict[zone_centriod_node[i,0]] = zone_centriod_node[i,1]

zone_index_id = pd.read_csv("data/NYC/zone_index_id.csv", header=None).values
zone_index_id_dict = dict()
for i in range(zone_index_id.shape[0]):
    zone_index_id_dict[zone_index_id[i,1]] = zone_index_id[i,0]
    
# Demand information used for solving optimizaiton problems
demand_data = pd.read_csv("data/NYC/demand/fhv_records_06272019.csv")

# Problem Parameters
β = 1
γ = 1e2
average_speed = 20
maximum_waiting_time = 300    # seconds
maximum_rebalancing_time = time_interval_length
big_M = 1e5

d = np.load("data/NYC/distance_matrix.npy") # Zone centroid distances in miles
d = np.repeat(d[:, :, np.newaxis], K, axis=2) # Repeat d to create a n x n x K matrix
# Hourly travel time to 288 time intervals
w_hourly = np.load("data/NYC/hourly_tt.npy")
a = np.repeat(w_hourly[:,:,0][:, :, np.newaxis], 12, axis=2)
for i in range(1,24):
    b = np.repeat(w_hourly[:,:,i][:, :, np.newaxis], 12, axis=2)
    a = np.concatenate((a, b), axis=2)

w = a * 3600;
w = w[:,:,start_bin:end_bin+1] # 7 AM to 9 AM travel time matrix

a = (w > maximum_rebalancing_time)
b = (w > maximum_waiting_time)

P = np.load("data/NYC/p_matrix_occupied.npy")
Q = np.load("data/NYC/q_matrix_occupied.npy")
P = np.repeat(P[:,:,np.newaxis], K, axis=2)
Q = np.repeat(Q[:,:,np.newaxis], K, axis=2)