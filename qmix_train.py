import argparse
import os
import sys
from pathlib import Path
import gym
import ray
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback

from agent.qmix.CBEngine_qmix import CBEngine_qmix as CBEngine_rllib_class

parser = argparse.ArgumentParser()


def process_roadnet(roadnet_file):
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id

    intersections = {}
    roads = {}
    agents = {}

    agent_num = 0
    road_num = 0
    signal_num = 0
    with open(roadnet_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        pre_road = 0
        is_obverse = 0
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if ('' in line):
                line.remove('')
            if (len(line) == 1):
                if (cnt == 0):
                    agent_num = int(line[0])
                    cnt += 1
                elif (cnt == 1):
                    road_num = int(line[0]) * 2
                    cnt += 1
                elif (cnt == 2):
                    signal_num = int(line[0])
                    cnt += 1
            else:
                if (cnt == 1):
                    intersections[int(line[2])] = {
                        'have_signal': int(line[3]),
                        'end_roads': [],
                        'start_roads': [],
                        'lanes':[]
                    }
                elif (cnt == 2):
                    if (len(line) != 8):
                        road_id = pre_road[is_obverse]
                        roads[road_id]['lanes'] = {}
                        for i in range(roads[road_id]['num_lanes']):
                            roads[road_id]['lanes'][road_id * 100 + i] = list(map(int, line[i * 3:i * 3 + 3]))
                        is_obverse ^= 1
                    else:
                        roads[int(line[-2])] = {
                            'start_inter': int(line[0]),
                            'end_inter': int(line[1]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[4]),
                            'inverse_road': int(line[-1])
                        }
                        roads[int(line[-1])] = {
                            'start_inter': int(line[1]),
                            'end_inter': int(line[0]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[5]),
                            'inverse_road': int(line[-2])
                        }
                        intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                        intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                        intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                        intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                        pre_road = (int(line[-2]), int(line[-1]))
                else:
                    # 4 out-roads
                    signal_road_order = list(map(int, line[1:]))
                    now_agent = int(line[0])
                    in_roads = []
                    for road in signal_road_order:
                        if (road != -1):
                            in_roads.append(roads[road]['inverse_road'])
                        else:
                            in_roads.append(-1)
                    in_roads += signal_road_order
                    agents[now_agent] = in_roads
    for agent, agent_roads in agents.items():
        intersections[agent]['lanes'] = []
        for road in agent_roads:
            ## here we treat road -1 have 3 lanes
            if (road == -1):
                for i in range(3):
                    intersections[agent]['lanes'].append(-1)
            else:
                for lane in roads[road]['lanes'].keys():
                    intersections[agent]['lanes'].append(lane)

    return intersections, roads, agents

if __name__ == "__main__":
    # some argument
    parser.add_argument(
        "--num_workers",
        type=int,
        default=30,
        help="rllib num workers"
    )
    parser.add_argument(
        "--multiflow",
        '-m',
        action="store_true",
        default = False,
        help="use multiple flow file in training"
    )
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=10,
        help="Number of iterations to train.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="QMIX",
        help="algorithm for rllib"
    )
    parser.add_argument(
        "--sim_cfg",
        type=str,
        default="/starter-kit/cfg/simulator_round3_flow0.cfg",
        help = "simulator file for CBEngine"
    )
    parser.add_argument(
        "--metric_period",
        type=int,
        default=3600,
        help = "simulator file for CBEngine"
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=8,
        help = "thread num for CBEngine"
    )
    parser.add_argument(
        "--gym_cfg_dir",
        type = str,
        default="agent",
        help = "gym_cfg (observation, reward) for CBEngine"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type = int,
        default = 5,
        help = "frequency of saving checkpoint"
    )

    parser.add_argument(
        "--foldername",
        type = str,
        default = 'train_result',
        help = 'The result of the training will be saved in ./model/$algorithm/$foldername/. Foldername can\'t have any space'
    )

    parser.add_argument(
        "--roadnet",
        type=str,
        default='/starter-kit/data/roadnet_warm_up.txt',
        help='Specify the roadnet file path'
    )

    # find the submission path to import gym_cfg
    args = parser.parse_args()
    for dirpath, dirnames, file_names in os.walk(args.gym_cfg_dir):
        for file_name in [f for f in file_names if f.endswith(".py")]:
            if file_name == "gym_cfg.py":
                cfg_path = dirpath
    sys.path.append(str(cfg_path))
    import gym_cfg as gym_cfg_submission
    gym_cfg_instance = gym_cfg_submission.gym_cfg()
    gym_dict = gym_cfg_instance.cfg
    simulator_cfg_files=[]

    # if set '--multiflow', then the CBEngine will utilize flows in 'simulator_cfg_files'
    if(args.multiflow):
        simulator_cfg_files = [
            '/starter-kit/cfg/simulator_round3_flow0.cfg'
            ]
    else:
        simulator_cfg_files = [args.sim_cfg]
    print('The cfg files of this training   ',format(simulator_cfg_files))
    class MultiFlowCBEngine(CBEngine_rllib_class):
        def __init__(self, env_config):
            env_config["simulator_cfg_file"] = simulator_cfg_files[0]
            super(MultiFlowCBEngine, self).__init__(config=env_config)


    # some configuration
    env_config = {
        "simulator_cfg_file": args.sim_cfg,
        "thread_num": args.thread_num,
        "gym_dict": gym_dict,
        "metric_period": args.metric_period,
        "vehicle_info_path": "/starter-kit/log/"
    }

    obs_size = gym_dict['observation_dimension']

    stop = {
        "training_iteration": args.stop_iters
    }

    roadnet_path = Path(args.roadnet)
    intersections, roads, agents = process_roadnet(roadnet_path)
    agent_group = {
        "group1": [str(agent_id) for agent_id, obsv in agents.items()]
    }

    OBSERVATION_SPACE = gym.spaces.Tuple(
        [
            gym.spaces.Dict({
                "obs": gym.spaces.Box(low=-1e10, high=1e10, shape=(obs_size,))
            }) for i in range(len(agents))
        ]
    )

    ACTION_SPACE = gym.spaces.Tuple(
        [gym.spaces.Discrete(9) for i in range(len(agents))]
    )

    ################################
    # modify this
    tune_config = {
        # env config
        "env_config": env_config,
        "num_cpus_per_worker": args.thread_num,
        "num_workers": args.num_workers,
        # "num_gpus": 1,

        # === QMix ===
        # Mixing network
        "mixer": "qmix",
        # Size of the mixing network embedding
        "mixing_embed_dim": 32,
        # Optimize over complete episodes by default
        "batch_mode": "complete_episodes",

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.
        },

        # Number of env steps to optimize for before returning
        "timesteps_per_iteration": 1000,
        "target_network_update_freq": 20,

        # === Replay buffer ===
        "buffer_size": 2000,

        # RMPProp Optimization
        "lr": 0.005,
        "optim_alpha": 0.99,
        "optim_eps": 0.00001,
        "learning_starts": 2000,
        "train_batch_size": 32,

        # === Model ===
        # Presslight as agent DQN
        # "model": {
        #     "fcnet_hiddens": [20],
        #     "fcnet_activation": "relu",
        # },
        # RNN as agent network
        "model": {
            "lstm_cell_size": 64,
            "max_seq_len": 999999,
        },

        # Only torch supported so far.
        "framework": "torch",
    }
    tune.register_env(
        "grouped_multiagent",
        lambda config: MultiFlowCBEngine(config).with_agent_groups(
            agent_group, obs_space=OBSERVATION_SPACE, act_space=ACTION_SPACE))
    tune_config = dict(tune_config, **{
        "env": "grouped_multiagent",
    })

    ########################################
    # ray.init(address="auto")          # Use for challenge submission
    # ray.init(local_mode=True)         # Use for local debugging
    local_path = './model'


    def name_creator(self=None):
        return args.foldername


    # train model
    ray.tune.run(args.algorithm,
                 config=tune_config,
                 local_dir=local_path,
                 stop=stop,
                 checkpoint_freq=args.checkpoint_freq,
                 trial_dirname_creator=name_creator,
                 callbacks=[
                     MLflowLoggerCallback(
                         tracking_uri="http://10.195.1.7:5000",
                         experiment_name="qmix-rllib-lstm-cc-warm_up-20000-iters",
                         save_artifact=True
                     )
                 ],
    )


