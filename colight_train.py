from copy import deepcopy
import gym
from agent.colight.CBEngine_round3 import CBEngine_round3
from baselines.colight.colight import Colight
import ray
import os
import numpy as np
import argparse
import sys
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.tune.integration.mlflow import MLflowLoggerCallback
parser = argparse.ArgumentParser()



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
        default="A3C",
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
        "--agents",
        type = int,
        required =True,
        help = 'The number of agents of the provided roadnet'
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
    class MultiFlowCBEngine(CBEngine_round3):
        def __init__(self, env_config):
            env_config["simulator_cfg_file"] = simulator_cfg_files[(env_config.worker_index - 1) % len(simulator_cfg_files)]
            super(MultiFlowCBEngine, self).__init__(config=env_config)


    # some configuration
    env_config = {
        "simulator_cfg_file": args.sim_cfg,
        "thread_num": args.thread_num,
        "gym_dict": gym_dict,
        "metric_period":args.metric_period,
        "vehicle_info_path":"/starter-kit/log/"
    }
    obs_size = gym_dict['observation_dimension']
    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-1e10, high=1e10, shape=(args.agents,obs_size,), dtype=np.float32),
        'adj': gym.spaces.Box(low=-1e10, high=1e10, shape=(args.agents,args.agents,args.agents), dtype=np.float32)
    })
    ACTION_SPACE = gym.spaces.Discrete(9)
    stop = {
        "training_iteration": args.stop_iters
    }
    ################################
    ModelCatalog.register_custom_model(
        "colight", Colight)
    if args.algorithm == "APEX":
        from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
        config = deepcopy(APEX_DEFAULT_CONFIG['model'])
        config.update({
            'custom_model': "colight",
            'custom_model_config': {
                'num_neighbors': args.agents,
                'num_agents': args.agents,
                'num_lanes': 24,
                'mlp_layer': [32, 32],
                'cnn_layer': [[32, 32], [32, 32]],
                'cnn_heads': [8],
            },
            'fcnet_hiddens': [8, 8],
        })
    elif args.algorithm == "DQN":
        from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG
        config = deepcopy(DEFAULT_CONFIG['model'])
        config.update({
            'custom_model': "colight",
            'custom_model_config': {
                'num_neighbors': args.agents,
                'num_agents': args.agents,
                'num_lanes': 24,
                'mlp_layer': [32, 32],
                'cnn_layer': [[32, 32], [32, 32]],
                'cnn_heads': [8],
            },
            'fcnet_hiddens': [8, 8],
        })

    # modify this
    tune_config = {
        # env config
        "framework": "torch",
        "env":MultiFlowCBEngine,
        "env_config" : env_config,
        "multiagent": {
            "policies": {
                "default_policy": (DQNTorchPolicy, OBSERVATION_SPACE, ACTION_SPACE, {},)
            }
        },
        "num_cpus_per_worker":args.thread_num,
        "num_workers":args.num_workers,
        "num_gpus": 1,
        "model": config,
        "n_step": 5



        # add your training config

    }
    ########################################
    #ray.init(address = "auto")
    #ray.init(local_mode=True)
    local_path = 'model'
    


    def name_creator(self=None):
        return args.foldername


    # train model
    ray.tune.run(args.algorithm, config=tune_config, local_dir=local_path, stop=stop,
                 checkpoint_freq=args.checkpoint_freq,trial_dirname_creator = name_creator,
                 callbacks=[MLflowLoggerCallback(
                     tracking_uri="http://10.195.1.7:5000",
                     experiment_name="colight_APEX_rllib",
                     save_artifact=True)]
                 )


