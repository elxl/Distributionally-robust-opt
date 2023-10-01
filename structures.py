# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:01:02 2023

@author: 11481
"""

class Passenger:
    def __init__(self, nid, origin, destination, request_time, assign_time, wait_time, travel_time, arrival_time, leave_system):
        self.id = nid
        self.origin = origin
        self.destination = destination
        self.request_time = request_time
        self.assign_time = assign_time
        self.wait_time = wait_time
        self.travel_time = travel_time
        self.arrival_time = arrival_time
        self.leave_system = leave_system

class Vehicle:
    def __init__(self, nid, location, available_time, current_location, occupied, served_passenger, rebalancing_travel_distance,
                 pickup_travel_distance, occupied_travel_distance, rebalancing_trips):
        self.id = nid
        self.location = location
        self.available_time = available_time
        self.current_location = current_location
        self.occupied = occupied
        self.served_passenger = served_passenger
        self.rebalancing_travel_distance = rebalancing_travel_distance
        self.pickup_travel_distance = pickup_travel_distance
        self.occupied_travel_distance = occupied_travel_distance
        self.rebalancing_trips = rebalancing_trips