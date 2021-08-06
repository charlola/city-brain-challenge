#!/bin/bash

cd starter-kit
pip install mlflow
pip install ray[rllib] -U
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python3 colight_train.py --sim_cfg /starter-kit/cfg/simulator_warm_up.cfg --gym_cfg_dir /starter-kit/agent/colight --algorithm APEX --stop-iters 30000 --foldername train_result --num_workers 16 --thread_num 4 --agents 22

