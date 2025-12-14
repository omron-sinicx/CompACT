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


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import robosuite.utils.transform_utils as T
import robosuite as suite
import torch
import yaml

from act.policy import ACTPolicy, CNNMLPPolicy
from act.utils import order_data, set_seed
from einops import rearrange
from act.utils import get_normalizers, reconstruct_dict

from utils import format_observations

torch.set_printoptions(precision=4, sci_mode=False, linewidth=1000)

"""Evaluation script for Comp-ACT in robosuite.

Usage:
        python3 evaluate_robosuite.py --rollout_dir

        --rollout_dir: Directory where the policy is stored
"""


def main(args):
    set_seed(1)

    # setup the environment
    config_filepath = os.path.join(args['rollout_dir'], "config.yaml")

    assert os.path.exists(config_filepath), f"Configuration file for rollout not found at '{config_filepath}'"

    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
        task_config = config['task_parameters']
        policy_config = config['policy_parameters']
        robosuite_config = config['robosuite']

    task_config["ckpt_dir"] = args['rollout_dir']

    # for robosuite
    suite_config = {
        **robosuite_config,
        "render_camera": task_config['camera_names'][0],
        "camera_names": task_config['camera_names']
    }

    ckpt_name = "policy_best.ckpt"

    eval_bc(task_config, policy_config, suite_config, ckpt_name)


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(obs, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs[f"{cam_name}_image"], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def visualize_ft(times, forces_hist, stiffness_vec, fig, axs: plt.Axes, axs_twin: plt.Axes, arm):
    labels = ['Fx', 'Fy', 'Fz']
    colors = ['r', 'g', 'b']
    # Update the force subplot
    offset = 6 if arm == "left" else 0

    axs.clear()

    for i in range(3):
        axs.plot(times, np.array(forces_hist)[:, i+offset], label=labels[i], color=colors[i], linewidth=1.0)

    axs.set_ylabel('force')
    axs.set_ylim(-30, 30)
    axs.legend(loc='upper left')
    # axs.grid()

    if len(stiffness_vec) > 0:
        axs_twin.clear()
        color = 'tab:olive'
        axs_twin.yaxis.set_label_position("right")
        axs_twin.plot(times, stiffness_vec, color=color, linewidth=1.0)
        axs_twin.set_ylabel('stiffness', color=color)
        axs_twin.tick_params(axis='y', color=color, labelcolor=color)
        axs_twin.set_ylim(0, 900)

    fig.tight_layout()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def get_obs(env, env_obs, camera_names, init_ft):
    obs = {}

    obs['observation.eef.position'] = np.concatenate([env_obs["robot0_eef_pos"], env_obs["robot1_eef_pos"]])
    obs['observation.eef.rotation_axis_angle'] = np.concatenate([T.quat2axisangle(env_obs["robot0_eef_quat"]), T.quat2axisangle(env_obs["robot1_eef_quat"])])
    obs['observation.eef.rotation_ortho6'] = np.concatenate([T.quat2ortho6(env_obs["robot0_eef_quat"]), T.quat2ortho6(env_obs["robot1_eef_quat"])])

    robot0_gripper_action = env.robots[0].gripper.current_action if np.any(env.robots[0].gripper.current_action) else [0.0]
    robot1_gripper_action = env.robots[1].gripper.current_action if np.any(env.robots[1].gripper.current_action) else [0.0]
    obs['observation.gripper'] = np.concatenate([robot0_gripper_action, robot1_gripper_action])

    obs['observation.qpos'] = np.concatenate([env.robots[0]._joint_positions, env.robots[1]._joint_positions])

    obs['images'] = get_image(env_obs, camera_names)

    cur_ft = np.concatenate([env_obs['robot0_eef_force_torque'],
                             env_obs['robot1_eef_force_torque']])
    obs_ft = cur_ft - init_ft

    obs['observation.ft'] = obs_ft

    obs = {k: torch.from_numpy(v).cuda().float().unsqueeze(0) if isinstance(v, np.ndarray) else v for k, v in obs.items()}
    return obs


def eval_bc(task_config, policy_config, robosuite_config, ckpt_name, debug=False):
    set_seed(1000)
    ckpt_dir = task_config["ckpt_dir"]
    action_dim = policy_config["action_dim"]
    policy_class = policy_config["policy_class"]
    onscreen_render = True
    policy_config = policy_config
    camera_names = task_config["camera_names"]
    max_timesteps = task_config["episode_len"]
    temporal_agg = policy_config["temporal_agg"]
    include_stiffness = robosuite_config['controller_configs']['type'] in ["OSC_POSE", "COMPLIANCE"]
    bimanual = policy_config["bimanual"]
    plot_ft = False
    skip_frame = 5

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")

    normalizers = get_normalizers(policy_config, ckpt_dir, device="cuda")

    env = suite.make(
        **robosuite_config,
        has_renderer=onscreen_render,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        hard_reset=False,
    )

    # init f/t visualizer
    if plot_ft:
        plt.ion()
        fig, axs = plt.subplots(figsize=(7, 4))  # Two subplots for force and torque
        axs_twin = axs.twinx() if include_stiffness else None
        fig.suptitle('Force and Torque Readings of the Right Arm', fontsize=14)
        plt.show(block=False)

    # Setup printing options for numbers
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1  # higher -> slower
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)
    num_rollouts = 30
    all_ft = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        obs = env.reset()

        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, action_dim]
            ).cuda()

        image_list = []  # for visualization

        times = []
        stiffness_vec = []
        forces_hist = []

        with torch.inference_mode():
            for t in range(max_timesteps):
                if onscreen_render:
                    env.render()

                # Offset the first reading
                if t == 0:
                    init_ft = np.concatenate([obs['robot0_eef_force_torque'], obs['robot1_eef_force_torque']])
                cur_ft = np.concatenate([obs['robot0_eef_force_torque'], obs['robot1_eef_force_torque']])
                ft = cur_ft - init_ft

                # Prepare observations
                observation = format_observations(env, obs, ft, camera_names)
                observation = {k: v.cuda().float().unsqueeze(0) for k, v in observation.items()}

                image_data = torch.stack([observation[f'observation.images.{cam_name}'] for cam_name in camera_names], dim=-4).permute([0, 1, 4, 2, 3]).cuda()
                image_list.append(image_data)

                normalized_obs = normalizers['input'](observation)

                state = torch.cat([normalized_obs[k] for k in policy_config['training_data']['observation_keys']], dim=-1)

                # query policy
                if policy_class == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(image=image_data, state=state, ft=normalized_obs['observation.ft'])

                    # if temporal ensemble, compute weighted average
                    if temporal_agg:
                        all_time_actions[[t], t: t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError

                # post-process action
                action_dict = reconstruct_dict(raw_action[0], policy_config['training_data']['output_shapes'], device="cuda")

                # un-normalize the predicted action
                action_dict = normalizers['output'](action_dict)

                # convert to axis angle for the controller if necessary
                if policy_config['orientation_representation'] == 'ortho6':
                    rot = action_dict[f'action.rotation_{policy_config["orientation_representation"]}'].squeeze(0).cpu().numpy()
                    action_rotation = torch.zeros(6).cuda()
                    action_rotation[:3] = torch.from_numpy(T.quat2axisangle(T.ortho62quat(rot[:6]))).cuda().float()
                    action_rotation[3:] = torch.from_numpy(T.quat2axisangle(T.ortho62quat(rot[6:]))).cuda().float()
                elif policy_config['orientation_representation'] == 'axis_angle':
                    action_rotation = action_dict[f'action.rotation_{policy_config["orientation_representation"]}']
                else:
                    raise ValueError(f"Invalid orientation representation {policy_config['orientation_representation']}")

                # reconstruct the action as expected by the environment
                env_action = order_data((
                    action_dict[f'action.stiffness_{policy_config["stiffness_representation"]}'],
                    action_dict['action.position'],
                    action_rotation,
                    action_dict['action.gripper'],
                ), bimanual).squeeze(0).cpu().numpy()

                obs, _, _, _ = env.step(env_action,)

                if plot_ft:
                    times.append(t)
                    if include_stiffness:
                        stiffness_vec.append(get_stiffness(policy_config['stiffness_representation'], env_action, arm="left"))
                    forces_hist.append(ft.ravel())
                    if t % skip_frame == 0:
                        visualize_ft(times, forces_hist, stiffness_vec, fig, axs, axs_twin, arm="left")
                        axs.set_xlim(xmin=-max_timesteps*0.01, xmax=max_timesteps*1.01)

            if plot_ft:
                fig_path = os.path.join(ckpt_dir, f"rollout_{rollout_id}.png")
                plt.savefig(fig_path)
                all_ft.append(forces_hist)

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    if plot_ft:
        plt.close()

    force_path = os.path.join(ckpt_dir, f"contact_force.npy")
    np.save(force_path, all_ft)


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
    parser.add_argument(
        "--rollout_dir", action="store", type=str, help="rollout_dir", required=True
    )
    main(vars(parser.parse_args()))
