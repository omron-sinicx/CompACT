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
# Author: Cristian Beltran


import timeit
import tqdm
from act.policy import ACTPolicy, CNNMLPPolicy
from act.utils import redirect_to_tqdm, set_seed  # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy

import torch
import numpy as np
import os
import argparse
import pickle
from einops import rearrange
import h5py
import yaml

from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat


def main(args):
    set_seed(1)

    # setup the environment
    config_filepath = os.path.join(args['rollout_dir'], "config.yaml")

    assert os.path.exists(config_filepath), f"Configuration file for rollout not found at '{config_filepath}'"

    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
        task_config = config['task_parameters']
        policy_config = config['policy_parameters']

    ckpt_names = [f"policy_best.ckpt"]

    dataset = load_data_bc(task_config['dataset_dir'], args['episode_idx'], policy_config['bimanual'])

    for ckpt_name in ckpt_names:
        eval_bc(task_config, policy_config, args['rollout_dir'], ckpt_name, dataset)

    print()
    exit()


def load_data_bc(dataset_dir, episode_idx, bimanual):
    data = []
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    with h5py.File(dataset_path, 'r+') as root:
        qpos = root['observations']['qpos'][:]
        ft = root['observations']['ft'][:] if 'ft' in root['observations'] else np.zeros_like(12 if bimanual else 6)
        eef_pos = root['observations']['eef_pos'][:]
        action = root['action'][:]

        image_dict = dict()
        for cam_name in root.attrs['camera_names']:
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][:]

        data = {
            'episode': episode_idx,
            'cameras': image_dict,
            'qpos': qpos,
            'eef_pos': eef_pos,
            'ft': ft,
            'action': action,
        }
    return data


def get_image(image_dict, camera_names, t):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(image_dict[cam_name][t], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image).float().cuda().unsqueeze(0) / 255.0
    return curr_image


def eval_bc(task_config, policy_config, ckpt_dir, ckpt_name, dataset):
    set_seed(1000)
    state_dim = policy_config['state_dim']
    policy_class = policy_config['policy_class']
    camera_names = task_config['camera_names']
    max_timesteps = task_config['episode_len']
    temporal_agg = policy_config['temporal_agg']
    num_queries = policy_config['chunk_size']
    action_space = policy_config['action_space']
    include_ft = policy_config['include_ft']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    print("checkpoint file", ckpt_path)
    assert os.path.exists(ckpt_path)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
        # Silly backward compatibility
        if "eef_or_q_pos_mean" in stats.keys():
            stats["obs_mean"] = stats["eef_or_q_pos_mean"]
            stats["obs_std"] = stats["eef_or_q_pos_std"]
        elif "qpos_mean" in stats.keys():
            stats["obs_mean"] = stats["qpos_mean"]
            stats["obs_std"] = stats["qpos_std"]

    def pre_process(s_qpos): return (s_qpos - stats["obs_mean"]) / stats["obs_std"]
    def post_process(a): return a * stats["action_std"] + stats["action_mean"]
    def pre_process_ft(ft): return ft * stats["ft_std"] + stats["ft_mean"]

    query_frequency = 50
    if temporal_agg:
        query_frequency = 10
        num_queries = num_queries

    max_timesteps = int(max_timesteps * 1)

    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, state_dim]
        ).cuda()

    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    inference_time = []
    obs_time = []

    with torch.inference_mode():
        for t in tqdm.tqdm(range(max_timesteps)):
            with redirect_to_tqdm():
                ##############
                obs_st = timeit.default_timer()
                print(f'{dataset['eef_pos'][t]=}')
                eef_pos = dataset['eef_pos'][t]
                eef_pos[3:6] = quat2axisangle(axisangle2quat(eef_pos[3:6]))
                print(f'<> {eef_pos=}')
                eef_pos = pre_process(eef_pos)
                print(f'{eef_pos=}')
                eef_pos = torch.from_numpy(eef_pos).float().cuda().unsqueeze(0)

                qpos = pre_process(dataset['qpos'][t])
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                curr_image = get_image(dataset['cameras'], camera_names, t)
                obs_time.append(timeit.default_timer()-obs_st)

                ft = pre_process_ft(dataset['ft'][t])
                ft = torch.from_numpy(ft).float().cuda().unsqueeze(0)
                if not include_ft:
                    ft = None
                ##############

                input('press any key to continue')

                # query policy
                ist = timeit.default_timer()
                if policy_class == "ACT":
                    if t % query_frequency == 0:
                        obs = eef_pos if action_space == 'cartesian' else qpos
                        all_actions = policy(image=curr_image, state=obs, ft=ft)

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
                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                inference_time.append(timeit.default_timer()-ist)

                # post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()

                action = post_process(raw_action)

                # print(f" policy action: {np.round(action,3)}")
                # print(f"dataset action: {np.round(dataset['action'][t], 3)}")
                # print("eef", dataset['eef_pos'][t])
                diff = action - dataset['action'][t]
                # diff = action[12:19] - dataset['action'][t][12:19]
                # assert np.allclose(diff, np.zeros_like(diff), atol=0.2), f"diff t{t}: {np.round(diff, 3).tolist()}"
                print(f"diff t{t}: {np.round(diff, 3).tolist()}")
                # break
        inference_time = np.array(inference_time)
        print(f"obs time avg:{round(np.average(obs_time),4)} std:{round(np.std(obs_time),4)}")
        print(f"inference time avg:{round(np.average(inference_time),4)} std:{round(np.std(inference_time),4)}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout_dir', action='store', type=str, help='Rollout directory', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='"episode index', default=0, required=False)
    main(vars(parser.parse_args()))
