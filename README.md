# Real World Imitation

## Requirements

- python 3.7+
- mujoco 2.0 (for RL experiments)
- Ubuntu 18.04

## Installation Instructions

Create a virtual environment and install all required packages.
```
cd real_world_imitation
pip3 install virtualenv
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Install dependencies and package
pip3 install -r requirements.txt
pip3 install -e .
```

Set the environment variables that specify the root experiment and data directories. For example: 
```
mkdir ./experiments
mkdir ./data
export EXP_DIR=./experiments
export DATA_DIR=./data
```

## Training Policy

To pre-train the BC policy ensemble:
```
python real_world_imitation/train.py --path=real_world_imitation/configs/teleop_model/real_world/bc_ensemble --skip_first_val=1 --prefix=real_world_run1
```