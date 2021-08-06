cd starter-kit

pip install 'ray[tune]' mlflow 'ray[default]'

python3 /starter-kit/baselines/presslight/train_presslight.py --input_dir /starter-kit/baselines/presslight --output_dir /starter-kit/out --sim_cfg /starter-kit/cfg/simulator.cfg --metric_period 200
