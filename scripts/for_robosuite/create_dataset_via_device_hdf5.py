#!/usr/bin/env python

#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2024 OMRON SINIC X
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian C. Beltran-Hernandez, Tatsuya Kamijo

"""Create dataset for ACT via teleoperation.

***Choose user input option with the --device argument***

Usage:
        $ python create_dataset_via_device.py \
        --config_file /root/osx-ur/dependencies/act/config/sim_bimanual_wiping_compliance.yaml \
        --arm left \
        --device gamepad \
        --save \
        --ft \
        --other-arm-dataset-dir /root/osx-ur/dependencies/act/datasets/sim_both_wiping_fixed \
        --plot-ft \

    --environment: Robosuite environment to use. Default set to TwoArmWiping.
    --num-episode: Number of demonstrations you want to perform. Default set to 10.
    --dataset-dir: Directory to save the data. Default set to /root/dataset/abot_wiping.
    --device: Device to use for demonstration (keyboard/spacemouse/gamepad). Default set to keyboard.
    --ft: Whether to save F/T readings or not.
    --plot-ft: Simultaneously plot the force
    --control-hz: Control frequency
    --max-timesteps: Maximum number of timesteps. Default set to 200.
    --save: Whether you want to save the data or not.
    --other-arm-dataset-dir: Load and use the data from a previous demonstration for the other arm

    After demonstrating each episode, press "q" in keyboard / "START" in gamepad
    press "A" in gamepad to reset the episode without saving it
    to save the demo and reset the environment to
    proceed to the next demonstration.
"""

import argparse
from pathlib import Path
import shutil
import timeit
import numpy as np
import time
import os
import h5py
import matplotlib.pyplot as plt
import yaml

import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
import robosuite.utils.transform_utils as T
from utils import format_observations, get_action, visualize_ft, get_device, flatten_np_dict, flatten_dict, unflatten_dict


def load_data(dataset_dir, num_episodes):
    data = []
    for episode_idx in range(num_episodes):
        dataset_dir = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_dir, 'r+') as root:
            qpos = root['observations']['qpos'][:]
            ft = root['observations']['ft'][:] if 'ft' in root['observations'] else np.zeros(12)
            eef_pos = root['observations']['eef_pos'][:]
            eef_pos_quat = root['observations']['eef_pos_quat'][:]
            action = root['action'][:]

            data.append({
                'episode': episode_idx,
                'qpos': qpos,
                'eef_pos': eef_pos,
                'eef_pos_quat': eef_pos_quat,
                'ft': ft,
                'action': action,
            })
    return data


def get_stiffness(stiffness_representation, actions, arm):
    offset = 0 if arm == "right" else int(np.ceil(len(actions)/2))
    if stiffness_representation == 'cholesky':
        # Convert to SPD and return just the first value
        s = T.cholesky_vector_to_spd(actions[offset: offset + 6])
        return s[0, 0]
    else:
        # Return just the first value
        return actions[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Configuration file for dataset")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=.6, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=.4, help="How much to scale rotation user inputs")
    parser.add_argument("--save", action="store_true", help="Save the dataset")
    parser.add_argument("--ft", action="store_true", help="Save force-torque readings")
    parser.add_argument("--plot-ft", action="store_true", help="Plot force-torque readings")
    parser.add_argument("--other-arm-dataset-dir", action="store", type=str, default=None,
                        help="Directory to load the previously recorded trajectory from the other arm.")
    args = parser.parse_args()

    # Get controller config
    with open(args.config_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
        task_config = dataset_config['task_parameters']
        policy_config = dataset_config['policy_parameters']
        robosuite_config = dataset_config['robosuite']

    control_frequency = int(robosuite_config['control_freq'])
    max_timesteps = task_config['episode_len']
    camera_names = task_config["camera_names"]

    stiffness_representation = policy_config['stiffness_representation']  # cholesky, diag, or None

    copy_dataset = False
    add_noise = False

    print("Control frequency", control_frequency)
    print("Max steps", max_timesteps)
    print("Camera names", camera_names)

    # Create environment
    env = suite.make(
        **robosuite_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=task_config["camera_names"][0],
        ignore_done=True,
        use_camera_obs=True,
        camera_names=task_config["camera_names"],
        reward_shaping=True,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    print(f">>> Action dimension {env.action_dim} for env {robosuite_config['env_name']} <<<")

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Prepare teleoperation interface
    device = get_device(args)
    if args.device == 'keyboard':
        env.viewer.add_keypress_callback(device.on_press)

    if args.plot_ft:
        # init f/t visualizer
        fig, axs = plt.subplots(figsize=(5, 3))  # Two subplots for force and torque
        axs_twin = axs.twinx()
        fig.suptitle('Force and Torque Readings of the Right Arm', fontsize=14)
        plt.show(block=False)

    # Load data for the other arm
    if args.other_arm_dataset_dir:
        other_arm_data = load_data(args.other_arm_dataset_dir, task_config['num_episodes'])

    dataset_dir = Path(task_config['dataset_dir'])
    if not dataset_dir.exists():
        dataset_dir.mkdir(exist_ok=True, parents=True)

    destfile = dataset_dir / "config.yaml"
    shutil.copyfile(args.config_file, destfile)

    # Determine the next episode index by examining existing files
    existing_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    existing_indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    episode_idx = max(existing_indices) + 1 if existing_files else 0

    last_reset = 0
    # demo (episode) loop
    while episode_idx < task_config['num_episodes']:
        # Reset the environment
        obs = env.reset()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        times = []
        force_hist = []
        stiffness_hist = []

        # Initialize device control
        device.start_control()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (camera_height, camera_width, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (1path2,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict_per_demo = {
        }

        # Set active robot
        active_robot = env.robots[args.arm == "right"]

        last_stiffness_switch = 0

        stiffness_values = [100, 800]
        stiff_idx = 1
        base_stiffness = stiffness_values[stiff_idx]

        save_or_skip = False  # skip
        control_delta = robosuite_config['controller_configs']['control_delta'] or copy_dataset
        skip_frame = 5

        if args.other_arm_dataset_dir:
            other_arm_episode_len = other_arm_data.episode_data_index["to"][episode_idx].item() - other_arm_data.episode_data_index["from"][episode_idx].item()

        t = 0
        # action loop
        while True:
            start_time = timeit.default_timer()

            # Get the newest action
            device_action, grasp = input2action(
                device=device, robot=active_robot, active_arm=args.arm, env_configuration="single-arm-opposed", control_delta=control_delta
            )

            if args.device == "gamepad":
                device_state = device.get_controller_state()
                btn_state = device_state.get('buttons_state', False)
                if btn_state:
                    if btn_state[0] and not last_stiffness_switch:  # X button
                        stiff_idx = (stiff_idx + 1) % len(stiffness_values)
                        base_stiffness = stiffness_values[stiff_idx]
                        print("New stiffness:", base_stiffness)
                    if btn_state[2] and not last_reset:  # A button
                        save_or_skip = False  # skip
                        last_reset = btn_state[2]
                        break

                last_reset = btn_state[2]
                last_stiffness_switch = btn_state[0]  # X button

            # If action is none, then this a reset so we should break
            if device_action is None:
                save_or_skip = True
                break

            action_dict = get_action(env, device_action, control_delta, args.arm, add_noise, base_stiffness)

            # load previously taken action for the other robot
            if args.other_arm_dataset_dir and not copy_dataset and t < other_arm_episode_len:
                for k in action_dict:
                    n = len(other_arm_data[episode_idx][k][t])
                    if n % 2 != 0 and args.arm == "left":
                        action_dict[k] = other_arm_data[episode_idx][k][t]
                    elif n % 2 == 0:
                        if args.arm == "right":
                            action_dict[k][1] = other_arm_data[episode_idx][k][t][n//2:]
                        else:
                            action_dict[k][0] = other_arm_data[episode_idx][k][t][:n//2]

            # Option to just copy one dataset to another
            if args.other_arm_dataset_dir and copy_dataset:
                other_idx = other_arm_data.episode_data_index["from"][episode_idx].item() + t
                if t < other_arm_episode_len:
                    for k in action_dict:
                        if len(other_arm_data[episode_idx][k][t]) % 2 == 0:
                            action_dict[k] = np.split(other_arm_data[episode_idx][k][t], 2)
                        else:
                            action_dict[k] = [other_arm_data[episode_idx][k][t]]
                else:
                    save_or_skip = True
                    break

            # Step through the simulation and render
            obs, reward, done, info = env.step(flatten_np_dict(action_dict))
            # print("action", np.round(action, 3))
            env.render()

            observation = format_observations(env, obs, camera_names)
            observation = flatten_dict(unflatten_dict(observation, sep='.'), sep='/')
            observation = {'/'+k: v for k, v in observation.items()}

            for key in observation:
                if key not in data_dict_per_demo:
                    data_dict_per_demo[key] = []
                data_dict_per_demo[key].append(observation[key])

            computation_time = timeit.default_timer() - start_time
            print('Episode:', episode_idx, 'Step:', t, 'duration:', round(computation_time, 2), end="\r")
            t += 1

        # continue to save this episode
        if not args.save or not save_or_skip:
            continue

        # fill the rest of the data to make each episode have the same length
        for name, array in data_dict_per_demo.items():
            try:
                elements_to_add = max_timesteps - len(array)
                if elements_to_add > 0:
                    last_element = [array[-1]] * elements_to_add
                    array.extend(last_element)
                    data_dict_per_demo[name] = array
            except:
                print("ERROR", name, array)

        # save as hdf5
        t0 = time.time()

        dataset_dir = os.path.join(task_config['dataset_dir'], f'episode_{episode_idx}')
        with h5py.File(dataset_dir + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            root.attrs['max_timesteps'] = max_timesteps
            root.attrs['camera_names'] = camera_names
            root.attrs['control_frequency'] = control_frequency
            root.attrs['stiffness_type'] = stiffness_representation
            root.attrs['action_type'] = "absolute_pose"  # delta_pose

            for k, v in data_dict_per_demo.items():
                root.create_dataset(k, shape=np.array(v).shape)

            for name, array in data_dict_per_demo.items():
                root[name][...] = array[:max_timesteps]
        print(f'Saving: {time.time() - t0:.1f} secs\n', end="\r")
        episode_idx += 1

    if args.plot_ft:
        plt.close()

    print(f'Saved to {task_config["dataset_dir"]}')
