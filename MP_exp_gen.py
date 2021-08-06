import gym
import numpy as np
import os
import argparse

import ray._private.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from agent.CBEngine_round3 import CBEngine_round3
from agent.agent_MP import MPAgent
from agent import gym_cfg as gym_cfg_submission

parser = argparse.ArgumentParser()

if __name__ == "__main__":
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out"))
    print('target folder', os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out"))
    # You normally wouldn't want to manually create sample batches if a
    # simulator is available, but let's do it anyways for example purposes:

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
        default=False,
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
        help="simulator file for CBEngine"
    )
    parser.add_argument(
        "--metric_period",
        type=int,
        default=3600,
        help="simulator file for CBEngine"
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=8,
        help="thread num for CBEngine"
    )

    # find the submission path to import gym_cfg
    args = parser.parse_args()

    gym_cfg_instance = gym_cfg_submission.gym_cfg()
    gym_dict = gym_cfg_instance.cfg

    env_config = {
        "simulator_cfg_file": args.sim_cfg,
        "thread_num": args.thread_num,
        "gym_dict": gym_dict,
        "metric_period": args.metric_period,
        "vehicle_info_path": "/starter-kit/log/"
    }
    env = CBEngine_round3(env_config)
    agent = MPAgent()

    ACTION_SPACE = gym.spaces.Discrete(9)
    OBSERVATION_SPACE = gym.spaces.Box(low=-1e10, high=1e10, shape=(env.observation_dimension,))
    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(OBSERVATION_SPACE)(OBSERVATION_SPACE)
    print("The preprocessor is", prep)

    for eps_id in range(200):
        observations = env.reset()
        infos = {'step': 0}

        prev_action = np.zeros_like(ACTION_SPACE.sample())
        prev_reward = 0
        done = False
        t = 0
        while not done:
            all_info = {
                'observations': observations,
                'info': infos
            }
            action = agent.act(all_info)
            new_obs, rew, done, infos = env.step(action)

            for agent_id, current_obs, next_obs, reward in zip(observations.keys(), observations.values(), new_obs.values(), rew.values()):
                current_obs = current_obs["observation"]
                next_obs = next_obs["observation"]
                batch_builder.add_values(
                    agent_index=int(agent_id),
                    t=t,
                    eps_id=eps_id,
                    #obs=prep.transform(observations),
                    obs=prep.transform(current_obs),
                    # obs=observations,
                    # obs=current_obs,
                    actions=action,
                    action_prob=1.0,  # put the true action probability here
                    action_logp=0.0,
                    rewards=reward,
                    prev_actions=prev_action,
                    prev_rewards=prev_reward,
                    dones=done,
                    infos=infos,
                    new_obs=prep.transform(next_obs)
                    # new_obs=prep.transform(new_obs)
                    # new_obs = new_obs
                    # new_obs=next_obs
                )
            observations = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        writer.write(batch_builder.build_and_reset())
