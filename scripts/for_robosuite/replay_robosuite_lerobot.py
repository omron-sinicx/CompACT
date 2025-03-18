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

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import imageio.v2 as iio
import yaml

from act.utils import load_hf_dataset, tensor_dict_to_np
import robosuite as suite
import robosuite.utils.transform_utils as T

from robosuite import macros

from robosuite.wrappers.visualization_wrapper import VisualizationWrapper

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"


def main(args):
    # command line parameters
    dataset_dir = args['dataset_dir']
    config_file = os.path.join(dataset_dir, "config.yaml")
    # Get controller config
    with open(config_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
        robosuite_config = dataset_config['robosuite']
        task_config = dataset_config['task_parameters']
        policy_config = dataset_config['policy_parameters']

    replay_bc(robosuite_config, task_config, policy_config,  dataset_dir=dataset_dir, args=args)


def replay_bc(config, task_config, policy_config, dataset_dir, args):
    # load environment

    # use a default value if not defined
    camera_names = task_config['camera_names']

    is_render = not args['save_video']
    debug = False
    plot_ft = args['plot_ft']

    if plot_ft:
        # init f/t visualizer
        fig, axs = plt.subplots(figsize=(5, 3))  # Two subplots for force and torque
        fig.suptitle('Force/Torque Readings of the Right Arm', fontsize=14)
        plt.show(block=False)

    # Create environment
    env = suite.make(
        **config,
        has_renderer=is_render,
        has_offscreen_renderer=True,
        render_camera=camera_names[0],
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names[0],
        reward_shaping=True,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # create a video writer with iio
    if args['save_video']:
        writer = iio.get_writer(args['video_path'], format='FFMPEG', mode='I', fps=50)

    # Setup printing options for numbers
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    all_force = []

    obs = env.reset()

    if is_render:
        env.render()

    times = []
    force_hist = []

    print('loading data..')
    ep_idx = args['episode_idx']
    dataset = load_hf_dataset(dataset_dir, ignore_videos=True)
    ep_start_idx = dataset.episode_data_index['from'][ep_idx].item()
    episode_len = dataset.episode_data_index['to'][ep_idx].item() - ep_start_idx
    for t in range(episode_len):
        action = {}
        for key in policy_config['training_data']['output_shapes']:
            action[key] = dataset[ep_start_idx + t][key]
        ds_obs = {}
        for key in policy_config['training_data']['input_shapes']:
            ds_obs[key] = dataset[ep_start_idx + t][key]
        ds_obs = tensor_dict_to_np(ds_obs)

        obs, _, _, _ = env.step(tensor_dict_to_np(action))

        if debug:
            print(f"\n t: {t}")
            print(f"{action=}")
            print(f"act aa {action[15:18]} | {action[-3:]}")
            print(f"act o6 {T.axis_angle2ortho6(action[15:18])} | {T.axis_angle2ortho6(action[-3:])}")
            exit(0)

        # Offset the first reading
        if t == 0:
            init_ft = np.concatenate([
                obs['robot0_eef_force_torque'],
                obs['robot1_eef_force_torque'],
            ])

        cur_ft = np.concatenate([
            obs['robot0_eef_force_torque'],
            obs['robot1_eef_force_torque'],
        ])
        obs_ft = cur_ft - init_ft

        if plot_ft:
            force_hist.append(obs_ft.ravel())
            times.append(t)
            if t % 5 == 0:
                visualize_ft(times, force_hist, fig, axs, "right")

        eef_diff0 = np.linalg.norm(obs['robot0_eef_pos'] - ds_obs['observation.eef.position'][:3])
        eef_diff1 = np.linalg.norm(obs['robot1_eef_pos'] - ds_obs['observation.eef.position'][3:])
        print(f'step: {t:03d} eef_diff0 {round(eef_diff0, 4):0.4f} eef_diff1 {round(eef_diff1, 4):0.4f}', end="\r")
        if is_render:
            env.render()

        if args['save_video']:
            # dump a frame from every K frames
            # if i % args.skip_frame == 0:
            frame = obs[f"{camera_names[0]}_image"]
            writer.append_data(frame)
            # print("Saving frame #{}".format(i))

    if args['save_ft']:
        all_force.append(force_hist)

        force_path = os.path.join(dataset_dir, "replay_contact_force.npy")
        np.save(force_path, all_force)


def visualize_ft(times, forces_hist, fig, axs: plt.Axes, arm):
    labels = ['Fx', 'Fy', 'Fz']
    colors = ['r', 'g', 'b']
    # Update the force subplot
    offset = 0 if arm == "left" else 6

    axs.clear()

    for i in range(3):
        axs.plot(times, np.array(forces_hist)[:, i+offset], label=labels[i], color=colors[i])

    axs.set_ylim(-15, 45)
    axs.legend(loc='upper left')
    axs.grid()

    fig.tight_layout()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', action='store', type=str, help='Absolute path to dataset', required=True)
    parser.add_argument('--episode-idx', action='store', type=int, help='Episode index', required=True)

    parser.add_argument("--plot-ft", action='store_true', help="Display Force/Torque")
    parser.add_argument("--save-video", action='store_true', help="Save a video of the replays")
    parser.add_argument("--video-path", type=str, default="demonstrations.mp4", help="Save a video of the replays")

    main(vars(parser.parse_args()))
