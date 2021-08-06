#!/bin/bash
cfg_array=('/starter-kit/cfg/simulator_warm_up.cfg')
algorithm="presslight"
foldername="presslight_result"
iteration_array=(10)
# Don't open lots of evaluating processes in parallel. It would cause the cloud server shutdown!!!!
for cfg in ${cfg_array[*]}
do
  for iteration in ${iteration_array[*]}
    do
      echo "=========================="
      echo "now test ${algorithm} ${cfg} iteration${iteration}"
      nohup python3 presslight_evaluate.py --sim_cfg ${cfg} --iteration ${iteration} --algorithm ${algorithm} --foldername ${foldername} --metric_period 200 --thread_num 4 > ./bash_result/${cfg:0-9}_iteration${iteration}_${foldername}.log 2>&1 &
    done
  wait
done

