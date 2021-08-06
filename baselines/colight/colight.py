
import numpy as np
import ray
from ray import tune
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
import torch.nn.functional as F

class Colight(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        #super(Colight, self).__init__()
        # neighbor have to be min(num_agents, num_neighbors) if neighbors should be adjusted for test purposes
        self.num_neighbors = model_config['custom_model_config']['num_neighbors']
        self.num_agents = model_config['custom_model_config']['num_agents']
        self.num_lanes = model_config['custom_model_config']['num_lanes']
        self.num_actions = action_space.n - 1




        # dimension oriented at official CoLight implementation
        self.dimension = model_config['custom_model_config']['mlp_layer'][-1]
        self.cnn_layer = model_config['custom_model_config']['cnn_layer']
        self.cnn_heads = model_config['custom_model_config']['cnn_heads'] * len(self.cnn_layer)
        self.mlp_layer = model_config['custom_model_config']['mlp_layer']

        self.mham_layers = nn.ModuleList()
        # MLP, feature, dimension
        self.mlp = MLP(self.num_lanes + self.num_actions, self.mlp_layer)
        # num of intersections, neighbor representation
        neighbor = torch.Tensor(self.num_agents, self.num_neighbors, self.num_agents)

        for CNN_layer_index, CNN_layer_size in enumerate(self.cnn_layer):
            mham = MHAM(self.num_agents, neighbor, self.num_actions, self.cnn_layer, self.num_neighbors, self.num_lanes,
                               self.dimension, CNN_layer_size[0], self.cnn_heads[CNN_layer_index],
                               CNN_layer_size[1])
            self.mham_layers.append(mham)

       # self.mham = MHAM(self.num_agents, neighbor, self.cnn_layer, num_lanes,
       #                  self.dimension, self.head_dim, self.num_heads, self.output_dim)
        self.out_hidden_layer = nn.Linear(self.cnn_layer[-1][1], self.num_actions)


    #def forward(self, nei, nei_actions, agent, actions):
    def forward(self, input_dict, state, seq_lens):
        adj = input_dict['obs']['adj']
        agent = input_dict['obs']['observation']
        batch_size = agent.shape[0]
        att_record = []
        #agent = torch.from_numpy(agent).float()
        x = self.mlp(agent)
        att_record_all_layers = []
        for i, mham in enumerate(self.mham_layers):
            x, att_record = mham(x, adj)
            att_record_all_layers.append(att_record)
        if len(self.cnn_layer) > 1:
            att_record_all_layers = torch.cat(att_record_all_layers, dim=1)
        else:
            att_record_all_layers = att_record_all_layers[0]
        att_record = torch.reshape(att_record_all_layers, (batch_size, len(self.cnn_layer), self.num_agents, self.cnn_heads[-1], self.num_neighbors))
        x = self.out_hidden_layer(x)
        x = x[:,0,:]
        return x, [] #att_record

# LambdaLayer for mimic Keras.Lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# see CoLight 4.1 (https://dl.acm.org/doi/10.1145/3357384.3357902)
class MLP(nn.Module):
    def __init__(self, input_shape, layer):
        super(MLP, self).__init__()
        layers = []
        for layer_index, layer_size in enumerate(layer):
            if layer_index == 0:
                layers.append(nn.Linear(input_shape, layer_size))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(layer[layer_index - 1], layer_size))
                layers.append(nn.ReLU())

        self.seq = nn.Sequential(*layers)

    def forward(self, ob):
        x = self.seq(ob)
        return x


# see CoLight 4.2 (https://dl.acm.org/doi/10.1145/3357384.3357902)
class MHAM(nn.Module):

    def __init__(self, num_agents, neighbor, action_space, cnn_layer, num_neighbors, input_shape=24, dimension=128, dv=16, nv=8, dout=128):
        super(MHAM, self).__init__()
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.dimension = dimension
        self.dv = dv
        self.nv = nv
        #self.neighbor = neighbor
        self.feature_length = input_shape
        self.dout = dout
        self.action_space = action_space

        # [agent,1,dim]->[agent,1,dv*nv], since representation of specific agent=1
        self.agent_head_hidden_layer = nn.Linear(self.dimension, self.dv*self.nv)
        self.agent_head_lambda_layer = LambdaLayer((lambda x: x.permute(0,1,4,2,3)))

       # self.neighbor_repr_3D = RepeatVector3D(num_agents)
        # [agent,neighbor,agent]x[agent,agent,dim]->[agent,neighbor,dim]
        #self.neighbor_repr_lambda_layer = LambdaLayer((lambda x: torch.einsum('ana, aad -> and', x[0], x[1])))
        self.neighbor_repr_lambda_layer = LambdaLayer((lambda x: torch.matmul(x[0], x[1])))

        # representation for all neighbors
        self.neighbor_repr_head_hidden_layer = nn.Linear(in_features=self.feature_length + self.action_space, out_features=dv*nv)
        self.neighbor_repr_head_lambda_layer = LambdaLayer((lambda x: x.permute(0,1,4,2,3)))

        # [batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
        self.attention_layer = LambdaLayer((lambda x: F.softmax(torch.einsum('bancd, baned -> bance', x[0], x[1]))))

        # self embedding
        self.neighbor_hidden_repr_head_hidden_layer = nn.Linear(self.feature_length + self.action_space, dv*nv)
        self.neighbor_hidden_repr_head_lambda_layer = LambdaLayer((lambda x: x.permute(0,1,4,2,3)))
        # mean values, preserving tensor shape
        self.out_lambda_layer = LambdaLayer((lambda x: torch.mean(torch.matmul(x[0], x[1]), 2)))
        self.out_hidden_layer = nn.Linear(dv, dout)

    def forward(self, agent, nei):
        batch_size = agent.size()[0]
        agent_repr = torch.reshape(agent, (batch_size, self.num_agents, 1, self.dimension))
        neighbor_repr = torch.reshape(agent, (batch_size, 1, self.num_agents, self.dimension))
        neighbor_repr = torch.tile(neighbor_repr, (1, self.num_agents,1,1))
        # nei = torch.FloatTensor(nei)
        #neighbor_repr = nei #self.neighbor_repr_lambda_layer([nei, neighbor_repr])
        neighbor_repr = self.neighbor_repr_lambda_layer([nei, neighbor_repr])

        agent_repr_head = self.agent_head_hidden_layer(agent_repr)
        agent_repr_head = F.relu(agent_repr_head)
        agent_repr_head = torch.reshape(agent_repr_head, (batch_size, self.num_agents, 1, self.dv, self.nv))

        agent_repr_head = self.agent_head_lambda_layer(agent_repr_head)
        neighbor_repr_head = self.neighbor_repr_head_hidden_layer(neighbor_repr)
        neighbor_repr_head = F.relu(neighbor_repr_head)
        # second num_agents could be replaced with num_neighbors if min(num_agents, num_neighbors)
        neighbor_repr_head = torch.reshape(neighbor_repr_head, (batch_size, self.num_agents, self.num_neighbors, self.dv, self.nv))
        neighbor_repr_head = self.neighbor_repr_head_lambda_layer(neighbor_repr_head)

       # agent_repr_head = agent_repr_head.reshape(-1, self.nv, 1, self.dv)
       # neighbor_repr_head = neighbor_repr_head.reshape(self.num_agents, self.nv, -1, self.dv)

        att = self.attention_layer([agent_repr_head, neighbor_repr_head])
        # second num_agents could be replaced with num_neighbors if min(num_agents, num_neighbors)
        att_record = torch.reshape(att, (batch_size, self.num_agents, self.nv, self.num_neighbors))

        neighbor_hidden_repr_head = self.neighbor_hidden_repr_head_hidden_layer(neighbor_repr)
        neighbor_hidden_repr_head = F.relu(neighbor_hidden_repr_head)
        neighbor_hidden_repr_head = torch.reshape(neighbor_hidden_repr_head, (batch_size, self.num_agents, self.num_neighbors, self.dv, self.nv))
        neighbor_hidden_repr_head = self.neighbor_hidden_repr_head_lambda_layer(neighbor_hidden_repr_head)
        out = self.out_lambda_layer([att, neighbor_hidden_repr_head])
        out = torch.reshape(out, (batch_size,self.num_agents, self.dv))
        out = self.out_hidden_layer(out)
        out = F.relu(out)
        return out, att_record


# Repeat vector x times
class RepeatVector3D(nn.Module):

    def __init__(self, times):
        super(RepeatVector3D, self).__init__()
        self.times = times

    def forward(self, x):
        x = torch.tile(torch.unsqueeze(x, 0), (1, self.times, 1, 1))
        return x



Colight = Colight