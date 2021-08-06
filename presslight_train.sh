#!/bin/sh
cd /starter-kit

python3 presslight_train.py \
  --gym_cfg_dir /starter-kit/agent \
  --sim_cfg /starter-kit/cfg/simulator_warm_up.cfg \
  --stop-iters 30000 \
  --foldername train_result \
  --num_workers 4 \
  --thread_num 4