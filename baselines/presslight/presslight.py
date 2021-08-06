import torch.nn as nn
import os
import sys

path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)

# contains all of the intersections
class Presslight(nn.Module):

    def __init__(self, ob_length, action_space, num_hidden_nodes, num_hidden_layers):
        # Torch version
        super(Presslight, self).__init__()

        layers = [nn.Linear(in_features=ob_length, out_features=num_hidden_nodes)]

        # number of hidden layers specified in hp_config
        for _ in range(num_hidden_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(in_features=num_hidden_nodes, out_features=num_hidden_nodes))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(num_hidden_nodes, out_features=action_space))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
