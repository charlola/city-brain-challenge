class gym_cfg():
    def __init__(self):
        """
        'custom_observation': If 'True', use custom observation feature in CBEngine_round3.py of agent.zip.
                    If 'False', use 'observation_features'

        'observation_features' : Same as round2. Add 'classic' observation feature,
                    which has dimension of 16.

        'observation_dimension' : The dimension of observation. Need to be correct both custom observation
                    and default observation. Remember to add size of one-hot-encoding vector for agent_id.
                    Size of one-hot-encoding vector for agent_id = 5 (warm_up), 10 (round2)
        """

        self.cfg = {
            'observation_features': ['lane_vehicle_num', 'classic'],
            # Add the length of one-hot encoded agent_id vector (ref above)
            'observation_dimension': 45,
            'custom_observation': True
        }
