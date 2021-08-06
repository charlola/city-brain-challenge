#!/bin/bash

cd starter-kit 

python3 evaluate.py --input_dir agent --output_dir out --sim_cfg cfg/simulator.cfg --metric_period 200
