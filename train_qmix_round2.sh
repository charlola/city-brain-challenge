#!/bin/sh
cd starter-kit

python3 qmix_train.py \
  --sim_cfg /starter-kit/cfg/simulator_round2.cfg \
  --roadnet /starter-kit/data/roadnet_round2.txt \
  --stop-iters 20000 \
  --foldername train_result_qmix \
  --num_workers 3 \
  --thread_num 3 \
  --gym_cfg_dir agent/qmix