""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""

import os
import sys
import random
import logging
from collections import deque
import numpy as np
import gym
import torch
import torch.optim as optim
from presslight import Presslight
import mlflow

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

gym.logger.setLevel(gym.logger.ERROR)


class PresslightAgent():
    def __init__(self):

        # used in agent.py
        self.now_phase = {}
        self.last_change_step = {}
        self.agent_list = []

        # used in train_presslight.py
        self.memory = deque(maxlen=2000)
        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        # hyperparameters
        self.gamma = None  # discount rate
        self.epsilon = None  # exploration rate
        self.epsilon_min = None
        self.epsilon_decay = None
        self.learning_rate = None
        self.num_hidden_nodes = None
        self.num_hidden_layers = None
        self.model = None
        self.optimizer = None
        self.target_model = None

        # hardcode configs
        self.batch_size = 32
        self.ob_length = 24
        self.action_space = 8

        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 199)

    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list, 1)
        self.last_change_step = dict.fromkeys(self.agent_list, 0)

    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

    ################################

    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.

        actions = {}
        for agent_id in self.agent_list:
            action = self.get_action(observations_for_agent[agent_id]['lane'])
            actions[agent_id] = action
        return actions

    def act(self, obs):
        observations = obs['observations']
        info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = {}
        for key, val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_') + 1:]
            if observations_agent_id not in observations_for_agent.keys():
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            actions[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1

        return actions

    def get_action(self, ob):
        # The epsilon-greedy action selector.
        if np.random.rand() <= self.epsilon:
            return self.sample()
        ob = self._reshape_ob(ob)
        act_values = self.model(torch.as_tensor(ob, dtype=torch.float32))
        return np.argmax(act_values[0].detach().numpy())

    def sample(self):
        # Random samples
        return np.random.randint(0, self.action_space)

    def _build_model(self, num_hidden_nodes, num_hidden_layers):
        model = Presslight(self.ob_length, self.action_space, num_hidden_nodes, num_hidden_layers)
        # model.apply(self.init_weights)
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        model_state_dict = self.model.state_dict()
        self.target_model.load_state_dict(model_state_dict)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        # Update the Q network from the memory buffer.
        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs, = [np.stack(x) for x in np.array(minibatch).T]
        pred_act_values = self.target_model(torch.as_tensor(next_obs, dtype=torch.float32))
        target = torch.from_numpy(rewards) + torch.mul(self.gamma, torch.amax(pred_act_values, dim=1))
        target_f = self.model(torch.as_tensor(obs, dtype=torch.float32))
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        # original Keras statement: self.model.fit([obs], target_f, epochs=1, verbose=0)
        # single epoch training
        # zero the parameter gradients
        optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        loss = criterion(pred_act_values, target_f)
        loss.backward()
        optimizer.step()
        logger.info(f"Single epoch training, loss = {loss.item()}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load_model(self, dir="model/presslight", step=0):
        name = "presslight_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        logger.info("load from " + model_name)
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self, dir="model/presslight", step=0):
        name = "presslight_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        torch.save(self.model.state_dict(), model_name)
        logger.info("save model to " + model_name)

    def set_hyperparameters(self, config):
        for key in config:
            setattr(self, key, config[key])

        self.model = self._build_model(config['num_hidden_nodes'], config['num_hidden_layers'])
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.target_model = self._build_model(config['num_hidden_nodes'], config['num_hidden_layers'])
        self.update_target_network()


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = PresslightAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`
