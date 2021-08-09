# KDDCup 2021: City Brain Challenge - Praktikum Big Data Science, SoSe 2021
## Introduction

Challenge documentation: [KDDCup 2021 City-Brain-Challenge](https://kddcup2021-citybrainchallenge.readthedocs.io/en/latest/city-brain-challenge.html). 

Each directory contains a baseline and its respective files to conduct training. The training metrics and artefacts are logged to an MLFlow dashboard. The resulting checkpoints could be evaluated in order to obtain the total amount of served vehicles, the delay index and the average travel time.

MLFlow dashboard: [http://10.195.1.7:5000/#](http://10.195.1.7:5000/#)

## How to train the baselines
### For local training 
For local training, run the docker container:
```
docker run -it -v <path to code repository>:/starter-kit --shm-size=20gb citybrainchallenge/cbengine:0.1.3 bash
```
When the docker container is running and the shell is accessible, proceed to model specific instructions below. 


#### Presslight
To run Presslight training locally, in the bash shell inside docker container, execute:
```
$ cd starter-kit
$ ./presslight_train.sh
```
Set the required params according to your choice.  
**Note:** Change the values for **sim_cfg** and **roadnet** path based on which challenge stage (_warm_up_, _round2_, _round3_) you wish to train for. MLFlow logging can be activated in the presslight_train.py file. The checkpoints will be saved in model/presslight/ directory. 

Paper: http://personal.psu.edu/hzw77/publications/presslight-kdd19.pdf

#### QMix
To run QMix training locally, in the bash shell inside docker container, execute:
```
$ cd starter-kit
$ python3 qmix_train.py --sim_cfg /starter-kit/cfg/simulator_warm_up.cfg --roadnet /starter-kit/data/roadnet_warm_up.txt --stop-iters 20000 --foldername train_result_qmix_warm_up_20000_iters --num_workers 3 --thread_num 3
```
Set the required params according to your choice.   
**Note:** Change the values for **sim_cfg** and **roadnet** path based on which challenge stage (_warm_up_, _round2_, _round3_) you wish to train for.   
**Important:** Change the "observation_dimension" in agent/qmix/gym_cfg.py depending upon the challenge stage chosen, before training. Set it to 45 for _warm_up_ and 50 for _round2_.

Paper: https://arxiv.org/abs/1803.11485

#### Colight
To run Colight training locally, in the bash shell inside docker container, execute:
```
$ cd starter-kit
$ bash colight_train.sh
```
Set the required params according to your choice.   
**Note:** Change the values for **sim_cfg**, **roadnet** path based on which challenge stage (_warm_up_, _round2_, _round3_) and **agents** to the you wish to train for.   
**Important:** Change the "adj_neighbors" in agent/colight/gym_cfg.py depending upon the challenge stage chosen, before training. It has to match the number of agents. _warm_up_: 22, _round2_: 859, _round3_: 1004

Paper: http://personal.psu.edu/hzw77/publications/colight-cikm19.pdf

## How to evaluate training

For local evaluation, run the docker container:
```
docker run -it -v <path to code repository>:/starter-kit --shm-size=20gb citybrainchallenge/cbengine:0.1.3 bash
```
When the docker container is running and the shell is accessible, proceed to model specific instructions below. 

### Presslight

To run the evaluation, in the bash shell inside docker container, execute:
```
$ cd starter-kit
$ ./presslight_evaluate.sh
```

### Colight

To run the evaluation, in the bash shell inside docker container, execute:
```
$ cd starter-kit
$ bash colight_evaluate.sh
```

**Important:** Change the "iteration_array" to the checkpoints you want to evaluate, Change "num_agents" according to the configuration of the trained model.


## Generate Max Pressure Experience

To generate max pressure experience as batches, please execute the ```MP_exp_gen.py```. The resulting JSON-file will be in directory "demo-out".

## Team 
Niklas Strauß (supervisor)    
Charlotte Vaessen   
Faheem Zunjani  
Maximilian Gawlick  
