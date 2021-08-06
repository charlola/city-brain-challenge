# -*- coding: utf-8 -*-
import numpy as np
import citypb
from ray import tune
import os
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=4, help="rllib num workers")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=5,
    help="Number of iterations to train.")


class CBEngine_rllib(MultiAgentEnv):
    """See MultiAgentEnv

    This environment will want a specific configuration:
        config: a dictionary with the environment configuration
            simulator_cfg_file :
                a str of the path of simulator.cfg
            gym_dict :
                a dictionary of gym configuration. It contains "observation_features" and "reward_feature"
            thread_num :
                a int of thread number
            metric_period :
                interval to log 'info_step *.json'

    """
    def __init__(self,config):
        super(CBEngine_rllib,self).__init__()
        self.simulator_cfg_file = config['simulator_cfg_file']
        self.gym_dict = config['gym_dict']
        self.observation_features = self.gym_dict['observation_features']
        self.thread_num = config['thread_num']
        self.metric_period = config['metric_period']
        self.vehicle_info_path = config['vehicle_info_path']
        self.__num_per_action = 10
        self.eng = citypb.Engine(self.simulator_cfg_file,self.thread_num)
        self.vehicles = {}
        # CFG FILE MUST HAVE SPACE
        with open(self.simulator_cfg_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n').split(' ')
                if(line[0] == 'start_time_epoch'):
                    self.now_step = int(line[-1])
                if(line[0] == 'max_time_epoch'):
                    self.max_step = int(line[-1])
                if(line[0] == 'road_file_addr'):
                    self.roadnet_file = line[-1]
                if(line[0] == 'report_log_rate'):
                    self.log_interval = int(line[-1])
                if(line[0] == 'report_log_addr'):
                    self.log_path = line[-1]
        # here agent is those intersections with signals
        self.intersections = {}
        self.roads = {}
        self.agent_signals = {}
        self.lane_vehicle_state = {}
        self.log_enable = 0
        self.warning_enable = 0
        self.ui_enable = 0
        self.info_enable = 0
        self.road2signal = {}
        self.agent_curphase = {}
        with open(self.roadnet_file,'r') as f:
            lines = f.readlines()
            cnt = 0
            pre_road = 0
            is_obverse = 0
            for line in lines:
                line = line.rstrip('\n').split(' ')
                if('' in line):
                    line.remove('')
                if(len(line) == 1):
                    if(cnt == 0):
                        self.agent_num = int(line[0])
                        cnt+=1
                    elif(cnt == 1):
                        self.road_num = int(line[0])*2
                        cnt +=1
                    elif(cnt == 2):
                        self.signal_num = int(line[0])
                        cnt+=1
                else:
                    if(cnt == 1):
                        self.intersections[int(line[2])] = {
                            'latitude':float(line[0]),
                            'longitude':float(line[1]),
                            'have_signal':int(line[3]),
                            'end_roads':[],
                            'start_roads':[]
                        }
                    elif(cnt == 2):
                        if(len(line)!=8):
                            road_id = pre_road[is_obverse]
                            self.roads[road_id]['lanes'] = {}
                            for i in range(self.roads[road_id]['num_lanes']):
                                self.roads[road_id]['lanes'][road_id*100+i] = list(map(int,line[i*3:i*3+3]))
                                self.lane_vehicle_state[road_id*100+i] = set()

                            is_obverse ^= 1
                        else:
                            self.roads[int(line[-2])]={
                                'start_inter':int(line[0]),
                                'end_inter':int(line[1]),
                                'length':float(line[2]),
                                'speed_limit':float(line[3]),
                                'num_lanes':int(line[4]),
                                'inverse_road':int(line[-1])
                            }
                            self.roads[int(line[-1])] = {
                                'start_inter': int(line[1]),
                                'end_inter': int(line[0]),
                                'length': float(line[2]),
                                'speed_limit': float(line[3]),
                                'num_lanes': int(line[5]),
                                'inverse_road':int(line[-2])
                            }
                            self.intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                            self.intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                            self.intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                            self.intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                            pre_road = (int(line[-2]),int(line[-1]))
                    else:
                        # 4 out-roads
                        signal_road_order = list(map(int,line[1:]))
                        now_agent = int(line[0])
                        in_roads = []
                        for road in signal_road_order:
                            if(road != -1):
                                in_roads.append(self.roads[road]['inverse_road'])
                                self.road2signal[self.roads[road]['inverse_road']] = now_agent
                            else:
                                in_roads.append(-1)
                        in_roads += signal_road_order
                        self.agent_signals[now_agent] = in_roads

                        # 4 in-roads
                        # self.agent_signals[int(line[0])] = self.intersections[int(line[0])]['end_roads']
                        # 4 in-roads plus 4 out-roads
                        # self.agent_signals[int(line[0])] += self.intersections[int(line[0])]['start_roads']
        for agent,agent_roads in self.agent_signals.items():
            self.intersections[agent]['lanes'] = []
            self.agent_curphase[agent] = 1
            for road in agent_roads:
                ## here we treat road -1 have 3 lanes
                if(road == -1):
                    for i in range(3):
                        self.intersections[agent]['lanes'].append(-1)
                else:
                    for lane in self.roads[road]['lanes'].keys():
                        self.intersections[agent]['lanes'].append(lane)
        ####################################
        # self.intersections
        #     - a dict
        #     - key is intersection_id (int), value is intersection_info
        #     - intersection_info : {
        #         'latitude': float value of latitude.
        #         'longitude': float value of longitude.
        #         'have_signal': 0 for no signal, 1 for signal.
        #         'end_roads': roads that end at this intersection.
        #         'start_roads': roads that start at this intersection.
        #         'lanes': optional. If this intersection is signalized, then it has 'lanes'. 24 dimension list with the same order as 'lane_vehicle_num' observation
        #     }

        # self.roads
        #     - a dict
        #     - key is road_id (int), value is road_info
        #     - road_info : {
        #         'start_inter': intersection this road starts with.
        #         'end_inter': intersection this road ends with.
        #         'length': length of this road.
        #         'speed_limit': speed limit of this road.
        #         'num_lanes': number of lanes of this road.
        #         'inverse_road': the inverse road of this road.
        #     }

        # self.agent_signals
        #     - a dict
        #     - key is agent (int), value is signal_info
        #     - signal_info is a list of 8 road_id. First 4 roads is in roads. Last 4 roads is out roads.

        # self.agent_curphase
        #     - a dict
        #     - key is agent_id (int), value is current phase
        ####################################

        ############ rllib api start here
        self.agents = list(map(str,self.agent_signals.keys()))
        self.n_obs = self.gym_dict['observation_dimension']

    def set_log(self,flg):
        self.log_enable = flg

    def set_warning(self,flg):
        self.warning_enable = flg

    def set_ui(self,flg):
        self.ui_enable = flg

    def set_info(self,flg):
        self.info_enable = flg

    def reset(self):
        del self.eng
        self.eng = citypb.Engine(self.simulator_cfg_file, self.thread_num)
        self.now_step = 0
        self.vehicles.clear()
        obs = self._get_observations()

        return obs

    def step(self, actions):
        # here action is a dict {agent_id:phase}
        # agent_id must be str

        for agent_id,phase in actions.items():
            result = self.eng.set_ttl_phase(int(agent_id),phase)
            if(result == -1 and self.warning_enable):
                print('Warnning: at step {} , agent {} switch to phase {} . Maybe empty road'.format(self.now_step,agent_id,phase))
        for cur in range(self.__num_per_action):
            self.eng.next_step()
            self.now_step+=1
            if((self.now_step +1)% self.log_interval == 0 and self.ui_enable==1):
                self.eng.log_info(os.path.join(self.log_path,'time{}.json'.format(self.now_step//self.log_interval)))

            # if((self.now_step+1) % self.log_interval ==0 and self.log_enable == 1):
            #     # replay file
            #     # vehicle info file
            #     vlist = self.eng.get_vehicles()
            #     for vehicle in vlist:
            #         if(vehicle not in self.vehicles.keys()):
            #             self.vehicles[vehicle] = {}
            #         for k,v in self.eng.get_vehicle_info(vehicle).items():
            #             self.vehicles[vehicle][k] = v
            #             self.vehicles[vehicle]['step'] = [self.now_step]
            if((self.now_step + 1) % self.metric_period == 0 and self.log_enable == 1):
                self.eng.log_vehicle_info(os.path.join(self.vehicle_info_path,'info_step {}.log'.format(self.now_step)))
                # with open(os.path.join(self.log_path,'info_step {}.log'.format(self.now_step)),'w+') as f:
                #     f.write("{}\n".format(self.eng.get_vehicle_count()))
                #     for vehicle in self.vehicles.keys():
                #         # if(self.vehicles[vehicle]['step'][0] <= self.now_step - self.metric_period):
                #         #     continue
                #         f.write("for vehicle {}\n".format(vehicle))
                #         for k,v in self.vehicles[vehicle].items():
                #             # f.write("{}:{}\n".format(k,v))
                #             if(k != 'step'):
                #                 f.write("{} :".format(k))
                #                 for val in v:
                #                     f.write(" {}".format(val))
                #                 f.write("\n")
                #         f.write('step :')
                #         for val in self.vehicles[vehicle]['step']:
                #             f.write(" {}".format(val))
                #         f.write("\n")
                #         f.write("-----------------\n")


        reward = self._get_reward()
        dones = self._get_dones()
        obs = self._get_observations()
        info = self._get_info()
        for agent_id,phase in actions.items():
            self.agent_curphase[int(agent_id)] = phase
        return obs, reward, dones , info


    def _get_info(self):
        info = {}
        if(self.info_enable == 0):
            return info
        else:
            v_list = self.eng.get_vehicles()
            for vehicle in v_list:
                info[vehicle] = self.eng.get_vehicle_info(vehicle)
            return info
    def _get_reward(self):
        raise NotImplementedError
        # in number
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
        # rwds = {}
        # # return every
        # lane_vehicle = self.eng.get_lane_vehicles()
        #
        # for agent_id, roads in self.agent_signals.items():
        #     rwds[agent_id] = 0
        #     result_reward = []
        #     for lane in self.intersections[agent_id]['lanes']:
        #         # -1 indicates empty roads in 'signal' of roadnet file
        #         if (lane == -1):
        #             result_reward.append(-1)
        #         else:
        #             if(lane not in lane_vehicle.keys()):
        #                 lane_vehicle[lane] = set()
        #             result_reward.append(get_diff(self.lane_vehicle_state[lane],lane_vehicle[lane]))
        #             self.lane_vehicle_state[lane] = lane_vehicle[lane]
        #     for i, res in enumerate(result_reward):
        #         if(isinstance(res,int) == False):
        #             rwds[agent_id] += res[0]


        # pressure



        # rwds = {}
        # if(self.reward_feature == 'pressure'):
        #     lane_vehicle = self.eng.get_lane_vehicles()
        #     for agent_id, roads in self.agent_signals.items():
        #         result_obs = []
        #         for lane in self.intersections[agent_id]['lanes']:
        #             # -1 indicates empty roads in 'signal' of roadnet file
        #             if (lane == -1):
        #                 result_obs.append(-1)
        #             else:
        #                 # -2 indicates there's no vehicle on this lane
        #                 if (lane not in lane_vehicle.keys()):
        #                     result_obs.append(0)
        #                 else:
        #                     # the vehicle number of this lane
        #                     result_obs.append(len(lane_vehicle[lane]))
        #         pressure = (np.sum(result_obs[12: 24]) - np.sum(result_obs[0: 12]))
        #         rwds[agent_id] = pressure
        # if(self.reward_feature == 'qlength'):
        #     v_list = self.eng.get_vehicles()
        #     for agent_id in self.agent_signals.keys():
        #         rwds[agent_id] = 0
        #     for vehicle in v_list:
        #         vdict = self.eng.get_vehicle_info(vehicle)
        #         if(float(vdict['speed'][0])<0.5 and float(vdict['distance'][0]) > 1.0):
        #             if(int(vdict['road'][0]) in self.road2signal.keys()):
        #                 agent_id = self.road2signal[int(vdict['road'][0])]
        #                 rwds[agent_id]-=1
        #     # normalization for qlength reward
        #     for agent_id in self.agent_signals.keys():
        #         rwds[agent_id] /= 10
        # int_agents = list(rwds.keys())
        # for k in int_agents:
        #     rwds[str(k)] = rwds[k]
        #     rwds.pop(k)
        # return rwds
    def _get_observations(self):
        # return self.eng.get_lane_vehicle_count()
        obs = {}
        lane_vehicle = self.eng.get_lane_vehicles()
        vehicle_speed = self.eng.get_vehicle_speed()

        features = self.observation_features

        # add 1 dimension to give current step for fixed time agent
        for agent_id, roads in self.agent_signals.items():
            result_obs = []
            for feature in features:
                if(feature == 'lane_speed'):
                    for lane in self.intersections[agent_id]['lanes']:
                        # -1 indicates empty roads in 'signal' of roadnet file
                        if(lane == -1):
                            result_obs.append(-1)
                        else:
                            # -2 indicates there's no vehicle on this lane
                            if(lane not in lane_vehicle.keys()):
                                result_obs.append(-2)
                            else:
                                # the average speed of this lane
                                speed_total = 0.0
                                for vehicle in lane_vehicle[lane]:
                                    speed_total += vehicle_speed[vehicle]
                                result_obs.append(speed_total / len(lane_vehicle[lane]))

                if(feature == 'lane_vehicle_num'):
                    for lane in self.intersections[agent_id]['lanes']:
                        # -1 indicates empty roads in 'signal' of roadnet file
                        if(lane == -1):
                            result_obs.append(-1)
                        else:
                            # -2 indicates there's no vehicle on this lane
                            if(lane not in lane_vehicle.keys()):
                                result_obs.append(0)
                            else:
                                # the vehicle number of this lane
                                result_obs.append(len(lane_vehicle[lane]))
                if(feature == 'classic'):
                    # first 8 lanes
                    for id, lane in enumerate(self.intersections[agent_id]['lanes']):
                        if(id > 11):
                            break
                        if(lane%100 == 2):
                            continue
                        if(lane == -1):
                            if(self.intersections[agent_id]['lanes'][id:id+3] == [-1,-1,-1]):
                                result_obs.append(0)
                                result_obs.append(0)
                        else:
                            if (lane not in lane_vehicle.keys()):
                                result_obs.append(0)
                            else:
                                # the vehicle number of this lane
                                result_obs.append(len(lane_vehicle[lane]))
                    # onehot phase
                    cur_phase = self.agent_curphase[agent_id]
                    phase_map = [
                        [-1,-1],
                        [0,4],
                        [1,5],
                        [2,6],
                        [3,7],
                        [0,1],
                        [2,3],
                        [4,5],
                        [6,7]
                    ]
                    one_hot_phase = [0]*8
                    one_hot_phase[phase_map[cur_phase][0]] = 1
                    one_hot_phase[phase_map[cur_phase][1]] = 1
                    result_obs += one_hot_phase
            obs[agent_id] = {'observation':result_obs}
        int_agents = list(obs.keys())
        for k in int_agents:
            obs[str(k)] = obs[k]
            obs.pop(k)
        return obs

    def _get_dones(self):
        #
        dones = {}
        for agent_id in self.agent_signals.keys():
            dones[str(agent_id)] = self.now_step >= self.max_step
        dones["__all__"] = self.now_step >= self.max_step
        return dones


if __name__ == "__main__":
    args = parser.parse_args()
    # order is important
    """
        simulator_cfg_file :
            a str of the path of simulator.cfg
        gym_dict :
            a dictionary of gym configuration. Now there's only 'observation_features', which is a list of str.
        thread_num :
            a int of thread number
    """
    env_config = {
        "simulator_cfg_file": 'cfg/simulator.cfg',
        "thread_num": 8,
        "gym_dict": {
            'observation_features':['classic'],
            'reward_feature':'qlength'
        },
        "metric_period": 3600
    }
    ACTION_SPACE = gym.spaces.Discrete(9)
    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-1e10, high=1e10, shape=(48,))
    })
    stop = {
        "training_iteration": args.stop_iters
    }
    tune_config = {
        "env":CBEngine_rllib,
        "env_config" : env_config,
        "multiagent": {
            "policies": {
                "default_policy": (None, OBSERVATION_SPACE, ACTION_SPACE, {},)
            }
        },

        "lr": 1e-4,
        "log_level": "WARN",
        "lambda": 0.95
    }

    tune.run("A3C",config = tune_config,stop = stop)






    # env = CBEngine_malib(env_config)
    # obs = env.reset()
    # while True:
    #     act_dict = {}
    #     for i, aid in enumerate(env.agents):
    #         act_dict[aid] = 1
    #     print('act_dict',act_dict)
    #     print('obs',obs)
    #     next_obs, rew, done, info = env.step(act_dict)
    #     print('rwd', rew)
    #     print('done', done)
    #     print('info', info)
    #     obs = next_obs
    #     if all(done.values()):
    #         break
    #     print()

