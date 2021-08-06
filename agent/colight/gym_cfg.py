class gym_cfg():
    def __init__(self):
        '''
        'custom_observation': If 'True', use costom observation feature in CBEngine_round3.py of agent.zip. If 'False', use 'observation_features'

        'observation_features' : Same as round2. Add 'classic' observation feature, which has dimension of 16.

        'observation_dimension' : The dimension of observation. Need to be correct both custom observation and default observation.

        '''

        self.cfg = {
            'observation_features':['lane_vehicle_num','action_one_hot', 'neighbor_adj', 'neighbors'],
            'observation_dimension':32,
            'custom_observation' : True,
            # adj_neighbors = number of total agents
            'adj_neighbors': 22
        }