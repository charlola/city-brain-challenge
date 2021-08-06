# -*- coding: utf-8 -*-
import numpy as np
import citypb
from ray import tune
import os
from CBEngine_rllib.CBEngine_rllib import CBEngine_rllib as CBEngine_rllib_class
import argparse
from queue import Queue


class CBEngine_round3(CBEngine_rllib_class):
    """See CBEngine_rllib_class in /CBEngine_env/env/CBEngine_rllib/CBEngine_rllib.py

    Need to implement reward.

    implementation of observation is optional

    """

    def __init__(self, config):
        super(CBEngine_round3, self).__init__(config)
        self.observation_features = self.gym_dict['observation_features']
        self.custom_observation = self.gym_dict['custom_observation']
        self.observation_dimension = self.gym_dict['observation_dimension']
        self.adj_neighbors = self.gym_dict['adj_neighbors']
        self.agent_adjacency = {}
    def _get_observations(self):

        obs = {}
        features = self.observation_features
        lane_vehicle = self.eng.get_lane_vehicles()
        if (self.custom_observation == False):
            obs = super(CBEngine_round3, self)._get_observations()
            return obs
        else:
            self.adj_dict = self.create_intersection_adj_list(self.intersections, self.agents)
            self.intersection_dict = self.create_intersection_dict(self.intersections)
            adj = []
            for agent_id, roads in self.agent_signals.items():
                result_obs = []
                for feature in features:
                    if feature == 'lane_vehicle_num':
                        for lane in self.intersections[agent_id]['lanes']:
                            # -1 indicates empty roads in 'signal' of roadnet file
                            if lane == -1:
                                result_obs.append(-1)
                            else:
                                # -2 indicates there's no vehicle on this lane
                                if lane not in lane_vehicle.keys():
                                    result_obs.append(0)
                                else:
                                    # the vehicle number of this lane
                                    result_obs.append(len(lane_vehicle[lane]))
                    if feature == 'action_one_hot':
                        cur_phase = self.agent_curphase[agent_id]
                        phase_map = [
                            [-1, -1],
                            [0, 4],
                            [1, 5],
                            [2, 6],
                            [3, 7],
                            [0, 1],
                            [2, 3],
                            [4, 5],
                            [6, 7]
                        ]
                        one_hot_phase = [0] * 8
                        one_hot_phase[phase_map[cur_phase][0]] = 1
                        one_hot_phase[phase_map[cur_phase][1]] = 1
                        result_obs += one_hot_phase
                    if feature == 'neighbor_adj':
                        if agent_id not in self.agent_adjacency:
                            nn, visited, level, parent = self.breadth_first_search(agent_id)
                            order = {k: v for v, k in enumerate(nn)}
                            nn = list(set(nn).intersection(self.adj_dict.keys()))
                            nn.sort(key=order.get)
                            self.agent_adjacency[agent_id] = nn
                        adj = self.agent_adjacency[agent_id][:self.adj_neighbors]
                    if feature == 'neighbors' and adj:
                        result_obs = [result_obs]
                        adj = [adj]
                        for neighbor in adj[0][1:]:
                            neighbor_obs = []
                            for lane in self.intersections[neighbor]['lanes']:
                                # -1 indicates empty roads in 'signal' of roadnet file
                                if lane == -1:
                                    neighbor_obs.append(-1)
                                else:
                                    # -2 indicates there's no vehicle on this lane
                                    if lane not in lane_vehicle.keys():
                                        neighbor_obs.append(0)
                                    else:
                                        # the vehicle number of this lane
                                        neighbor_obs.append(len(lane_vehicle[lane]))

                            cur_phase = self.agent_curphase[neighbor]
                            phase_map = [
                                [-1, -1],
                                [0, 4],
                                [1, 5],
                                [2, 6],
                                [3, 7],
                                [0, 1],
                                [2, 3],
                                [4, 5],
                                [6, 7]
                            ]
                            one_hot_phase = [0] * 8
                            one_hot_phase[phase_map[cur_phase][0]] = 1
                            one_hot_phase[phase_map[cur_phase][1]] = 1
                            neighbor_obs += one_hot_phase
                            result_obs.append(neighbor_obs)
                            if neighbor not in self.agent_adjacency:
                                nn, visited, level, parent = self.breadth_first_search(neighbor)
                                order = {k: v for v, k in enumerate(nn)}
                                nn = list(set(nn).intersection(self.adj_dict.keys()))
                                nn.sort(key=order.get)
                                self.agent_adjacency[neighbor] = nn
                            neighbor_adj = self.agent_adjacency[neighbor][:self.adj_neighbors]
                            adj.append(neighbor_adj)
                obs[agent_id] = {'observation': np.array(result_obs),
                                 'adj': np.array(self.adjacency_index2matrix(adj)).squeeze(axis=0)}
            int_agents = list(obs.keys())
            for k in int_agents:
                obs[str(k)] = obs[k]
                obs.pop(k)

            return obs


    def _get_reward(self):

        rwds = {}

        ##################
        ## Example : pressure as reward.
        lane_vehicle = self.eng.get_lane_vehicles()
        for agent_id, roads in self.agent_signals.items():
            result_obs = []
            for lane in self.intersections[agent_id]['lanes']:
                # -1 indicates empty roads in 'signal' of roadnet file
                if (lane == -1):
                    result_obs.append(-1)
                else:
                    # -2 indicates there's no vehicle on this lane
                    if (lane not in lane_vehicle.keys()):
                        result_obs.append(0)
                    else:
                        # the vehicle number of this lane
                        result_obs.append(len(lane_vehicle[lane]))
            pressure = (np.sum(result_obs[12: 24]) - np.sum(result_obs[0: 12]))
            rwds[agent_id] = pressure
        ##################

        ##################
        ## Example : queue length as reward.
        # v_list = self.eng.get_vehicles()
        # for agent_id in self.agent_signals.keys():
        #     rwds[agent_id] = 0
        # for vehicle in v_list:
        #     vdict = self.eng.get_vehicle_info(vehicle)
        #     if(float(vdict['speed'][0])<0.5 and float(vdict['distance'][0]) > 1.0):
        #         if(int(vdict['road'][0]) in self.road2signal.keys()):
        #             agent_id = self.road2signal[int(vdict['road'][0])]
        #             rwds[agent_id]-=1
        # normalization for qlength reward
        # for agent_id in self.agent_signals.keys():
        #     rwds[agent_id] /= 10

        ##################

        ##################
        ## Default reward, which can't be used in rllib
        ## self.lane_vehicle_state is dict. keys are agent_id(int), values are sets which maintain the vehicles of each lanes.

        # def get_diff(pre,sub):
        #     in_num = 0
        #     out_num = 0
        #     for vehicle in pre:
        #         if(vehicle not in sub):
        #             out_num +=1
        #     for vehicle in sub:
        #         if(vehicle not in pre):
        #             in_num += 1
        #     return in_num,out_num
        #
        # lane_vehicle = self.eng.get_lane_vehicles()
        #
        # for agent_id, roads in self.agents.items():
        #     rwds[agent_id] = []
        #     for lane in self.intersections[agent_id]['lanes']:
        #         # -1 indicates empty roads in 'signal' of roadnet file
        #         if (lane == -1):
        #             rwds[agent_id].append(-1)
        #         else:
        #             if(lane not in lane_vehicle.keys()):
        #                 lane_vehicle[lane] = set()
        #             rwds[agent_id].append(get_diff(self.lane_vehicle_state[lane],lane_vehicle[lane]))
        #             self.lane_vehicle_state[lane] = lane_vehicle[lane]
        ##################
        # Change int keys to str keys because agent_id in actions must be str
        int_agents = list(rwds.keys())
        for k in int_agents:
            rwds[str(k)] = rwds[k]
            rwds.pop(k)
        return rwds

    def create_intersection_adj_list(self, intersections, agents):
        adj_dict = {}
        for idx, inter in intersections.items():
            if str(idx) in agents:
                adj_dict[idx] = [idx]
                for road in inter['start_roads'] or road in inter['end_roads']:
                    for neigh_idx, neigh in intersections.items():
                        if str(neigh_idx) in agents:
                            if road in neigh['end_roads'] or road in neigh['start_roads']:
                                adj_dict[idx].append(neigh_idx)
        return adj_dict

    def create_intersection_dict(self, intersections):
        dict = {}
        for idx, inter in intersections.items():
            dict[idx] = []
            for road in inter['start_roads'] or road in inter['end_roads']:
                for neigh_idx, neigh in intersections.items():
                    if road in neigh['end_roads'] or road in neigh['start_roads']:
                        dict[idx].append(neigh_idx)
        return dict

    def breadth_first_search(self, start_id):
        visited = {}
        level = {}
        parent = {}
        traversal_output = []
        queue = Queue()
        for node in self.intersection_dict.keys():
            visited[node] = False
            parent[node] = None
            level[node] = -1
        s = start_id
        visited[s] = True
        level[s] = 0
        queue.put(s)
        while not queue.empty():
            u = queue.get()
            traversal_output.append(u)
            for v in self.intersection_dict[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    level[v] = level[u] + 1
                    queue.put(v)
        return traversal_output, visited, level, parent

    def adjacency_index2matrix(self, adjacency_index):
        #for idx, adjacency in enumerate(adjacency_index):
        #    adjacency_index[idx] = np.array([np.array(i) for i in adjacency])
            # adjacency_index[idx] = np.sort(adjacency_index[idx])
        # adjacency_index = np.array([np.array(i) for i in adjacency_index])
        # adjacency_index_new = np.sort(adjacency_index)
        m = self.to_categorical(adjacency_index, num_classes=self.adj_neighbors)
        return m

    def to_categorical(self, y, num_classes, dtype='float32'):
        """ 1-hot encodes a tensor """
        in_cat = []
        for idx, agent in enumerate(y):
            y[idx] = [agent.index(n) for n in agent]
        b_cat = np.array(y, dtype='int')
        input_shape = b_cat.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        b_cat = b_cat.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = b_cat.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), b_cat] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        in_cat.append(categorical.tolist())
        return in_cat