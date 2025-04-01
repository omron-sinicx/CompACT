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


import datetime
from pathlib import Path
import timeit

from collections import deque
from lerobot.common.datasets.utils import flatten_dict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from copy import deepcopy
from act.utils import compute_dict_mean, get_cosine_schedule_with_warmup, set_seed, detach_dict, get_normalizers, get_random_batches  # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy
import yaml
import tqdm
import torch
import os
import numpy as np
import argparse
from safetensors.torch import save_file
import shutil

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

torch.set_printoptions(precision=4, sci_mode=False, linewidth=1000)


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
    print(task_config)
    # pprint.pprint(policy_config)

    if args.get('pretrained_policy_path', None):
        config.update({'pretrained_policy_path': args['pretrained_policy_path']})

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    ckpt_dir = Path(task_config["ckpt_dir"]) / f'HF_{current_datetime}'
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    train_dataloader, val_dataloader, normalizers = load_data(policy_config, task_config['dataset_dir'], ckpt_dir)

    # Save config file on rollout folder
    destfile = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(destfile):
        os.remove(destfile)
    shutil.copyfile(config_filepath, destfile)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, normalizers, ckpt_dir)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def load_data(config, dataset_dir, ckpt_dir):
    # Let's take one for this example
    device = torch.device("cuda")
    if isinstance(dataset_dir, list):
        root = Path(dataset_dir[0]).parent.resolve()
        dataset_class = MultiLeRobotDataset
        repo_id = [Path(ds).name for ds in dataset_dir]
    else:
        root = Path(dataset_dir).parent.resolve()
        dataset_class = LeRobotDataset
        repo_id = Path(dataset_dir).name

    dataset = dataset_class(repo_id, root=root)
    print("dataset info: \n", dataset)

    # # Set up the dataset.
    # dataset = LeRobotDataset(repo_id, root=root)

    normalizers = get_normalizers(config, ckpt_dir, device="cuda", stats=dataset.stats)

    stats_path = os.path.join(ckpt_dir, f'stats.safetensors')
    save_file(flatten_dict(dataset.stats), stats_path)

    # load `chunk_size` number of actions for each action component
    # e.g., loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
    delta_timestamps = {k: [t / dataset.fps for t in range(config['chunk_size'])] for k in config['training_data']['output_shapes']}

    num_workers = 8
    num_samples = dataset.num_samples
    validation_type = config.get('validation_type', 'split')
    print(f'Validation mode: {validation_type}')
    if validation_type == 'split':
        train_ratio = config.get('train_ratio', 0.9)
        ratio = int(num_samples*train_ratio)
        print(f'# samples {num_samples} | split ration {train_ratio} | train: {ratio} | val:{num_samples-ratio}')
        train_dataset = dataset_class(repo_id, root=root, split=f"train[:{ratio}]", delta_timestamps=delta_timestamps)
        val_dataset = dataset_class(repo_id, root=root, split=f"train[{ratio}:]", delta_timestamps=delta_timestamps)
    elif validation_type == 'cross_validation':
        train_dataset = dataset_class(repo_id, root=root, split=f"train", delta_timestamps=delta_timestamps)
        val_dataset = train_dataset
    else:
        raise ValueError(f"Unsupported validation type {validation_type}")

    # Create dataloader for offline training.
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
        persistent_workers=True
    )
    return train_dataloader, val_dataloader, normalizers


def train_bc(train_dataloader, val_dataloader, config, normalizers, ckpt_dir):
    policy_config = config['policy_parameters']
    max_checkpoints = policy_config.get('max_checkpoints', 2)

    seed = policy_config["seed"]
    policy_class = policy_config["policy_class"]

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
    if policy_config.get('lr_scheduler', None):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=policy_config['lr_warmup_steps'],
            num_training_steps=policy_config['num_training_steps'])

    # Initialize TensorBoard writer
    writer = SummaryWriter(ckpt_dir)

    min_val_loss = np.inf
    best_ckpt_info = None

    iter_num = 0  # avoid validation and saving at iteration 0
    total_iterations = policy_config["num_training_steps"]
    save_policy_interval = int(total_iterations/10)
    validation_iterations = policy_config["num_steps_to_validate"]
    validation_interval = policy_config["validation_interval"]
    train_loss_batch = []

    start_time = timeit.default_timer()

    # Use a deque to keep track of checkpoint filenames
    checkpoint_files = deque()

    pbar = tqdm.tqdm(total=total_iterations, desc="Training", position=0)
    while iter_num < total_iterations:
        for batch in train_dataloader:
            # train
            policy.train()
            optimizer.zero_grad()
            forward_dict = forward_pass(policy_config, batch, policy, normalizers)
            # backward
            loss = forward_dict["loss"]
            loss.backward()

            if policy_config['grad_clip_norm'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    policy_config['grad_clip_norm'],
                    error_if_nonfinite=False,
                )

            optimizer.step()
            optimizer.zero_grad()

            if policy_config['lr_scheduler']:
                lr_scheduler.step()

            for k, v in detach_dict(forward_dict).items():
                writer.add_scalar(f'Train/{k}', v, iter_num)
            writer.add_scalar(f'LR', optimizer.param_groups[0]['lr'], iter_num)
            train_loss_batch.append(loss.item())

            # Validation
            if iter_num % validation_interval == 0:
                dataloader_subset = get_random_batches(val_dataloader, validation_iterations)
                val_epoch_summary = validation(policy_config, policy, dataloader_subset, normalizers)

                for k, v in val_epoch_summary.items():
                    writer.add_scalar(f'Validation/{k}', v, iter_num)
                # Plot comparison
                writer.add_scalars('loss', {'train': np.mean(train_loss_batch), 'val': val_epoch_summary['loss']}, iter_num)
                train_loss_batch = []

                # Save best policy
                if val_epoch_summary['loss'] < min_val_loss:
                    min_val_loss = val_epoch_summary['loss']
                    best_ckpt_info = (iter_num, min_val_loss, deepcopy(policy.state_dict()))

                tqdm.tqdm.write(f"Val loss:   {val_epoch_summary['loss']:.5f} LR: {optimizer.param_groups[0]['lr']:.2E}")

            # Save policy at regular intervals
            if (iter_num + 1) % save_policy_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"policy_iter_{iter_num+1}_seed_{seed}.ckpt")
                torch.save(policy.state_dict(), ckpt_path)
                checkpoint_files.append(ckpt_path)

                tqdm.tqdm.write(f"Saved checkpoint at iteration {iter_num+1}")

                # Remove oldest checkpoint if we've exceeded max_checkpoints
                if len(checkpoint_files) > max_checkpoints:
                    oldest_file = checkpoint_files.popleft()
                    os.remove(oldest_file)
                    tqdm.tqdm.write(f"Removed oldest checkpoint: {oldest_file}")

            iter_num += 1
            # Update tqdm
            pbar.update(1)
            if iter_num > total_iterations:
                break
    # Close the progress bar
    pbar.close()

    print("Total Training Time:", round(timeit.default_timer() - start_time, 2))

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

    return best_ckpt_info


def validation(config, policy, val_dataloader, normalizers):
    # validation
    with torch.inference_mode():
        policy.eval()
        epoch_dicts = []
        for data in tqdm.tqdm(val_dataloader, total=len(val_dataloader), desc='validation loop', position=1, leave=False):
            forward_dict = forward_pass(config, data, policy, normalizers)
            epoch_dicts.append(detach_dict(forward_dict))

        return compute_dict_mean(epoch_dicts)


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


def forward_pass(config, batch, policy, normalizers):
    # Move data to GPU
    batch = {k: v.to(torch.device("cuda"), non_blocking=True) for k, v in batch.items()}

    # Get images before normalization as policy() does normalize images already
    image_data = torch.stack([batch[f'observation.images.{cam_name}'] for cam_name in config['camera_names']], dim=-4)

    # Normalize data
    normalized_batch = normalizers['input'](batch)
    normalized_batch = normalizers['target'](normalized_batch)

    # Merge the observation components into a single array with shape (batch_size, merged_array)
    state_data = torch.cat([normalized_batch[k] for k in config['training_data']['observation_keys']], dim=-1)

    # TODO(cambel): For now FT is separated from the rest of the observations
    ft_data = normalized_batch['observation.ft']

    # Merge the action components into a single array with shape (batch_size, merged_array)
    action_data = torch.cat([normalized_batch[k] for k in config['training_data']['output_shapes']], dim=-1)

    # Get action padding
    is_pad_key = list(config['training_data']['output_shapes'].keys())[0] + "_is_pad"
    is_pad = batch[is_pad_key]

    return policy(image_data, state=state_data, actions=action_data, is_pad=is_pad, ft=ft_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_robot', action='store_true')
    parser.add_argument('--task_name', action='store', type=str, help='Task name', required=True)
    parser.add_argument('--pretrained_policy_path', action='store', type=str, help='Pretrained policy. Start training from it', required=False)
    main(vars(parser.parse_args()))
