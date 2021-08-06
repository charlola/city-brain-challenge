#!/bin/sh
cd starter-kit

python3 qmix_train.py \
  --sim_cfg /starter-kit/cfg/simulator_warm_up.cfg \
  --roadnet /starter-kit/data/roadnet_warm_up.txt \
  --stop-iters 20000 \
  --foldername train_result_qmix \
  --num_workers 3 \
  --thread_num 3 \
  --gym_cfg_dir agent/qmix