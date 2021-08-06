import numpy as np

from CBEngine_rllib.CBEngine_rllib import CBEngine_rllib as CBEngine_rllib_class


class CBEngine_qmix(CBEngine_rllib_class):

    def __init__(self, config):
        super(CBEngine_qmix, self).__init__(config)
        self.observation_features = self.gym_dict['observation_features']
        self.custom_observation = self.gym_dict['custom_observation']
        self.observation_dimension = self.gym_dict['observation_dimension']
        self.agent_N_mapping = {}
        self.agent_id_one_hot_mapping = {}

    def _get_observations(self):
        obs = super(CBEngine_qmix, self)._get_observations()
        # RLLib QMix expects obs with key 'obs' unlike 'observation'
        # provided by CBEngine env
        for agent_id, obs_dict in obs.items():
            obs[agent_id] = {'obs': obs_dict['observation']}

        if self.custom_observation == False:
            return obs
        else:
            # Custom observation for QMix
            # Obs from env + one hot encoding of agent_id

            if len(self.agent_N_mapping) == 0:
                for i, agent_id in enumerate(self.agents):
                    self.agent_N_mapping[agent_id] = i

                self.agent_id_one_hot_mapping = self.get_one_hot_encoding(
                    self.agent_N_mapping
                )
            for agent_id, agent_obs in obs.items():
                obs[agent_id]['obs'] = agent_obs['obs'] \
                                       + self.agent_id_one_hot_mapping[agent_id]

            return obs

    def _get_reward(self):

        rwds = {}

        ##################
        ## Pressure as reward.
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

        int_agents = list(rwds.keys())
        for k in int_agents:
            rwds[str(k)] = rwds[k]
            rwds.pop(k)
        return rwds

    def get_one_hot_encoding(self, agent_id_n_mapping):
        agent_one_hot_mapping = {}
        n = len(agent_id_n_mapping)
        one_hot_n = [int(i) for i in bin(n)[2:]]
        for agent_id, agent_n in agent_id_n_mapping.items():
            binary_agent_n = [
                int(i) for i in bin(agent_n)[2:]
            ]
            prefix_vec = [
                0 for i in range(len(one_hot_n) - len(binary_agent_n))
            ]
            agent_one_hot_id = prefix_vec + binary_agent_n
            assert len(agent_one_hot_id) == len(one_hot_n)
            agent_one_hot_mapping[agent_id] = agent_one_hot_id
        return agent_one_hot_mapping
