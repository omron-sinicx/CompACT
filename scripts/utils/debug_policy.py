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


import pprint
import timeit
import tqdm
from act.policy import ACTPolicy, CNNMLPPolicy
from act.utils import order_data, redirect_to_tqdm, set_seed  # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy

import torch
import numpy as np
import os
import argparse
from einops import rearrange

import yaml

from osx_teleoperation.dataset_utils import get_normalizers, merge_dict_tensors, reconstruct_dict
from osx_teleoperation.debug_utils import load_hf_dataloader, print_comparison

torch.set_printoptions(precision=4, sci_mode=False, linewidth=1000)


def main(args):
    set_seed(1)

    # setup the environment
    config_filepath = os.path.join(args['rollout_dir'], "config.yaml")

    assert os.path.exists(config_filepath), f"Configuration file for rollout not found at '{config_filepath}'"

    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
        task_config = config['task_parameters']
        policy_config = config['policy_parameters']

    ckpt_names = [f"policy_iter_27000_seed_1.ckpt"]

    # dataset = load_data_bc(task_config['dataset_dir'], args['episode_idx'], policy_config['bimanual'])

    for ckpt_name in ckpt_names:
        eval_bc(task_config, policy_config, args['rollout_dir'], ckpt_name)

    print()
    exit()


def get_image(image_dict, camera_names, t):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(image_dict[cam_name][t], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image).float().cuda().unsqueeze(0) / 255.0
    return curr_image


def eval_bc(task_config, policy_config, ckpt_dir, ckpt_name):

    pprint.pprint(policy_config)
    set_seed(1000)
    state_dim = policy_config['state_dim']
    policy_class = policy_config['policy_class']
    camera_names = task_config['camera_names']
    max_timesteps = task_config['episode_len']
    temporal_agg = policy_config['temporal_agg']
    num_queries = policy_config['chunk_size']

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

    normalizers = get_normalizers(policy_config, ckpt_dir, device="cuda")

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

    batch_size = 10
    dataloader = load_hf_dataloader(task_config['dataset_dir'], batch_size=batch_size)

    with torch.inference_mode():
        for t, raw_batch in tqdm.tqdm(enumerate(dataloader), total=batch_size):
            with redirect_to_tqdm():
                ##############
                obs_st = timeit.default_timer()

                # Move to GPU
                batch = {k: v.to(torch.device("cuda"), non_blocking=True) for k, v in raw_batch.items()}

                # Get images before normalization as policy() does normalize images already
                curr_image = torch.stack([batch[f'observation.images.{cam_name}'] for cam_name in camera_names], dim=-4)

                # Normalization
                nbatch = normalizers['input'](batch)
                nbatch = normalizers['target'](nbatch)

                # Merge the observation components into a single array with shape (batch_size, merged_array)
                state_data = torch.cat([nbatch[k] for k in policy_config['training_data']['observation_keys']], dim=-1)

                ft_data = nbatch['observation.ft']

                # diff_image = torch.abs(curr_image_hf - curr_image)
                # display_images(curr_image.permute([0, 1, 4, 3, 2]).cpu().numpy()[0], titles=camera_names, figtitle="HF")
                # display_images(curr_image_hf.permute([0, 1, 4, 3, 2]).cpu().numpy()[0], titles=camera_names, figtitle="HDF5")
                # display_images(diff_image.permute([0, 1, 4, 3, 2]).cpu().numpy()[0], titles=camera_names, figtitle="Diff")

                obs_time.append(timeit.default_timer()-obs_st)

                ##############

                # query policy
                ist = timeit.default_timer()
                if policy_class == "ACT":
                    if t % query_frequency == 0:
                        curr_image = curr_image[0].unsqueeze(0)
                        state_data = state_data[0].unsqueeze(0)
                        ft_data = ft_data[0].unsqueeze(0)
                        # print(f"{curr_image.shape=}")
                        # print(f"{state_data.shape=}")
                        # print(f"{ft_data.shape=}")
                        all_actions = policy(image=curr_image, state=state_data, ft=ft_data)

                    # # if temporal ensemble, compute weighted average
                    # if temporal_agg:
                    #     print(f"{all_actions.shape=}")
                    #     all_time_actions[[t], t: t + num_queries] = all_actions
                    #     actions_for_curr_step = all_time_actions[:, t]
                    #     actions_populated = torch.all(
                    #         actions_for_curr_step != 0, axis=1
                    #     )
                    #     actions_for_curr_step = actions_for_curr_step[actions_populated]
                    #     k = 0.01
                    #     exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    #     exp_weights = exp_weights / exp_weights.sum()
                    #     exp_weights = (
                    #         torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    #     )
                    #     raw_action = (actions_for_curr_step * exp_weights).sum(
                    #         dim=0, keepdim=True
                    #     )
                    # else:
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError
                inference_time.append(timeit.default_timer()-ist)

                normalized_predicted_action = reconstruct_dict(raw_action[0], policy_config['training_data']['output_shapes'], device="cuda")

                # post-process actions
                predicted_action = normalizers['output'](normalized_predicted_action)

                for k in policy_config['training_data']['output_shapes'].keys():
                    print_comparison(k, predicted_action[k], batch[k][0])
                    # print_comparison(k, normalized_predicted_action[k], nbatch[k][0])

                exit(0)
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
