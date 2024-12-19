# CompACT: Compliance Control via Action Chunking with Transformers

#### Project Website: https://omron-sinicx.github.io/CompACT/

This repo contains the implementation of CompACT, together with a simulated environment.
The simulation environment is implemented in the `robosuite` framework for a bimanual
wiping task, a digital twin of our real dual-arm robotic environment.

For the teleoperation system, check this repository: https://github.com/omron-sinicx/vive_tracking_ros

### Repo Structure

- `/config` Configuration files for dataset creation, training, and evaluation
- `/libs` Our implementation of CompACT, based on ACT and DETR
- `/scripts`
  - `train_policy_*.py` Train script for datasets in HDF5 format or LeRobot format
  - `/for_robosuite`
    - `create_dataset_via_device_*.py` Scripts to collect demonstrations in our robosuite environment either with HDF5 format or LeRobot format
    - `evaluate_robosuite.py` Script to evaluate CompACT policies in our robosuite environment

### Installation

    conda create -n comp-act python=3.8.10
    conda activate comp-act
    pip install -r requirements.txt
    pip install -e libs/act
    pip install -e libs/detr

We use Robosuite for our simulation environment. So clone and install our fork:

    git clone https://github.com/omron-sinicx/robosuite.git -b CompACT
    cd robosuite
    pip install -r requirements.txt
    pip install -r requirements-extras.txt
    pip install -e .

We strongly recommend using the lerobot format for space efficient data collection. So clone and install our fork:

    git clone https://github.com/omron-sinicx/lerobot.git -b python-3.8
    pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate comp-act
    cd <path to act repo>

### Robosuite experiments

We use `TwoArmWiping` task in the example below.

#### 1. Create a dataset for a_bot

a_bot is the robot you see on the right in the Robosuite environment. Note that this robot is "right" arm in the code.
To generate 10 episodes of wiping demonstration data of a_bot via gamepad, run:

    cd CompACT/dependencies/comp-act/scripts/
    python3 for_robosuite/create_dataset_via_device_lerobot.py \
        --config_file ../config/sim_bimanual_wiping_compliance.yaml \
        --arm right \
        --device keyboard \
        --save \
        --ft

After demonstrating each episode, press "p" / "START" in gamepad to save the demo and reset the environment to
proceed to the next demonstration.

#### 3. Train

To train:

    cd CompACT/dependencies/comp-act/scripts
    python3 train_policy_lerobot.py --task_name sim_bimanual_wiping_compliance

--task_name: Name of the config file related to this task

#### 4. Evaluate

To evaluate:
    cd CompACT/dependencies/comp-act/scripts
    python3 for_robosuite/evaluate_robosuite.py --rollout_dir path_to_rollout_directory

### Authors:

- Tatsuya Kamijo
- Cristian C. Beltran-Hernandez
- Masashi Hamaya

If you find this project useful,
consider citing it.

```
  @article{kamijo2024learning,
    title={Learning Variable Compliance Control From a Few Demonstrations for Bimanual Robot with Haptic Feedback Teleoperation System},
    author={Kamijo, Tatsuya and Beltran-Hernandez, Cristian C and Hamaya, Masashi},
    journal={arXiv preprint arXiv:2406.14990},
    year={2024}
  }
```
