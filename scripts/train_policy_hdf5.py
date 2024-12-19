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
import pprint
import shutil
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import tqdm
import yaml

from act.policy import ACTPolicy, CNNMLPPolicy
from act.utils import load_data, redirect_to_tqdm
from act.utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from copy import deepcopy


def main(args):
    set_seed(1)

    # get task parameters
    is_sim = not args['real_robot']
    task_name = args['task_name']

    if is_sim:
        folder = "../config"
    else:
        import rospkg
        folder = rospkg.RosPack().get_path("osx_robot_control") + "/config/gym"

    # setup the environment
    config_filepath = os.path.join(folder, task_name+'.yaml')
    assert os.path.exists(config_filepath), f"Configuration file for task '{task_name}' not found at '{config_filepath}'"

    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
        task_config = config['task_parameters']
        policy_config = config['policy_parameters']

    print("Training parameters")
    pprint.pprint(task_config)
    pprint.pprint(policy_config)

    dataset_config = {
        'action_space': policy_config['action_space'],
        'include_ft': policy_config['include_ft'],
        'episode_len': task_config['episode_len'],
        'camera_names': task_config['camera_names'],
        'real_robot': not is_sim,
    }

    if args.get('pretrained_policy_path', None):
        config.update({'pretrained_policy_path': args['pretrained_policy_path']})

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir=task_config['dataset_dir'],
        num_episodes=task_config['num_episodes'],
        batch_size_train=policy_config['batch_size'],
        batch_size_val=policy_config['batch_size'],
        dataset_config=dataset_config
    )

    ckpt_dir = task_config['ckpt_dir']
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Save config file on rollout folder
    destfile = os.path.join(ckpt_dir, "config.yaml")
    shutil.copyfile(config_filepath, destfile)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def train_bc(train_dataloader, val_dataloader, config):
    task_config = config['task_parameters']
    policy_config = config['policy_parameters']

    num_epochs = policy_config["num_epochs"]
    ckpt_dir = task_config["ckpt_dir"]
    seed = policy_config["seed"]
    policy_class = policy_config["policy_class"]
    include_ft = policy_config["include_ft"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)

    if config.get('pretrained_policy_path', None):
        ckpt_path = config['pretrained_policy_path']
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print("pretrained policy:", loading_status)
        policy.cuda()
        policy.eval()
        print(f"Loaded: {ckpt_path}")
    else:
        policy.cuda()

    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm.tqdm(range(num_epochs)):
        with redirect_to_tqdm():
            # print(f"\nEpoch {epoch}")
            # validation
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy, include_ft)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            print(f"Val loss:   {epoch_val_loss:.5f}")
            summary_string = ""
            for k, v in epoch_summary.items():
                summary_string += f"{k}: {v.item():.3f} "
            print(summary_string)

            # training
            policy.train()
            optimizer.zero_grad()
            for batch_idx, data in enumerate(train_dataloader):
                forward_dict = forward_pass(data, policy, include_ft)
                # backward
                loss = forward_dict["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
            epoch_summary = compute_dict_mean(
                train_history[(batch_idx + 1) * epoch: (batch_idx + 1) * (epoch + 1)]
            )
            epoch_train_loss = epoch_summary["loss"]
            print(f"Train loss: {epoch_train_loss:.5f}")
            summary_string = ""
            for k, v in epoch_summary.items():
                summary_string += f"{k}: {v.item():.3f} "
            print(summary_string)

            if epoch % int(num_epochs/10) == 0:
                ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
                torch.save(policy.state_dict(), ckpt_path)
            if epoch % 100 == 0:
                plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        if key == "loss":
            plt.ylim([-0.1, 1])
        elif key == "kl":
            plt.ylim([-0.1, 1])
        plt.legend()
        plt.grid()
        plt.title(key)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    print(f"Saved plots to {ckpt_dir}")


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


def forward_pass(data, policy, include_ft):
    image_data, eef_or_q_pos_data, ft_data, action_data, is_pad = data

    image_data, eef_or_q_pos_data, action_data, is_pad = (
        image_data.cuda(),
        eef_or_q_pos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )

    return policy(image_data, state=eef_or_q_pos_data, actions=action_data, is_pad=is_pad, ft=ft_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_robot', action='store_true')
    parser.add_argument('--task_name', action='store', type=str, help='Task name', required=True)
    parser.add_argument('--pretrained_policy_path', action='store', type=str, help='Pretrained policy. Start training from it', required=False)
    main(vars(parser.parse_args()))
