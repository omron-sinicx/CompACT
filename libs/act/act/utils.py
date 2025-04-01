import contextlib
import inspect
import h5py
import math
import numpy as np
import os
import random
import robosuite.utils.transform_utils as T
import torch
import tqdm

from copy import copy
from math import ceil
from pathlib import Path
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from safetensors.torch import load_file

from lerobot.common.datasets.utils import unflatten_dict
from lerobot.common.policies.normalize import Normalize, Unnormalize


ACTION_SPACES = ['joint', 'cartesian']


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, norm_stats, dataset_config):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.include_ft = dataset_config['include_ft']
        self.camera_names = dataset_config['camera_names']
        self.episode_len = dataset_config['episode_len']
        self.action_space = dataset_config['action_space']

        # Verify the action space is supported
        assert self.action_space in ACTION_SPACES, (
            "Error: Tried to instantiate EpisodicDataset for unsupported "
            "action space! Inputted action space: {}, Supported action spaces: {}".format(self.action_space, ACTION_SPACES)
        )

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            action_id = '/action'
            original_action_shape = root[action_id].shape
            if not self.episode_len:
                self.episode_len = original_action_shape[0] - 1
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(self.episode_len)

            # get observation at start_ts only
            obs = None
            if self.action_space == 'cartesian':
                obs = root['/observations/eef_pos'][start_ts]
            elif self.action_space == 'joint':
                obs = root['/observations/qpos'][start_ts]

            if self.include_ft:
                ft = root['/observations/ft'][start_ts]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            # get all actions after and including start_ts
            if is_sim:
                action = root[action_id][start_ts:]
                action_len = self.episode_len - start_ts
            else:
                action = root[action_id][max(0, start_ts - 1):]
                action_len = self.episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        obs_data = torch.from_numpy(obs).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        obs_data = (obs_data - self.norm_stats["obs_mean"]) / self.norm_stats["obs_std"]

        if self.include_ft:
            ft_data = torch.from_numpy(ft).float()
            ft_data = (ft_data - self.norm_stats["ft_mean"]) / self.norm_stats["ft_std"]
        else:
            ft_data = []

        return image_data, obs_data, ft_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, dataset_config):
    all_obs_data = []
    all_ft_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            action_id = '/action'

            if dataset_config['action_space'] == 'cartesian':
                obs = root['/observations/eef_pos'][()]
            elif dataset_config['action_space'] == 'joint':
                obs = root['/observations/qpos'][()]

            action = root[action_id][()]

            if dataset_config['include_ft']:
                ft = root['/observations/ft'][()]

        all_obs_data.append(torch.from_numpy(obs))
        all_action_data.append(torch.from_numpy(action))
        if dataset_config['include_ft']:
            all_ft_data.append(torch.from_numpy(ft))

    all_obs_data = torch.stack(all_obs_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    obs_mean = all_obs_data.mean(dim=[0, 1], keepdim=True)
    obs_std = all_obs_data.std(dim=[0, 1], keepdim=True)
    obs_std = torch.clip(obs_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "obs_mean": obs_mean.numpy().squeeze(), "obs_std": obs_std.numpy().squeeze(),
             "example_obs": obs}

    if dataset_config['include_ft']:
        all_ft_data = torch.stack(all_ft_data)
        # normalize ft data
        ft_mean = all_ft_data.mean(dim=[0, 1], keepdim=True)
        ft_std = all_ft_data.std(dim=[0, 1], keepdim=True)
        ft_std = torch.clip(ft_std, 1e-2, np.inf)  # clipping
        stats["ft_mean"] = ft_mean.numpy().squeeze()
        stats["ft_std"] = ft_std.numpy().squeeze()

    return stats


def load_data(dataset_dir, num_episodes, batch_size_train, batch_size_val, dataset_config):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # construct dataset and dataloader
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, dataset_config)
    train_dataset = EpisodicDataset(train_indices, dataset_dir, norm_stats, dataset_config)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, norm_stats, dataset_config)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, dataset_config['real_robot']


def load_hf_dataset(dataset_dir, ignore_videos=False):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = Path(dataset_dir)
    # Let's take one for this example
    repo_id = f"{dataset_dir.parents[0].name}/{dataset_dir.name}"
    root = dataset_dir.parents[1]

    # Set up the dataset.
    return LeRobotDataset(repo_id, root=root, ignore_videos=ignore_videos)

# env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

# helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@contextlib.contextmanager
def redirect_to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.tqdm.write(*args, **kwargs)
        except:
            old_print(*args, ** kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


def order_data(tuple_data: tuple, bimanual: bool) -> torch.Tensor:
    """
    Concatenates and optionally reorders tensor data based on whether the operation is bimanual or not.

    Args:
        tuple_data (tuple): A tuple of tensors to be concatenated. Each tensor should have the same shape
                            except for the last dimension.
        bimanual (bool): A flag indicating whether to perform bimanual reordering.

    Returns:
        torch.Tensor: A concatenated tensor. If bimanual is False, this is a simple concatenation
                      along the last dimension. If bimanual is True, the tensor is reordered such that
                      the first half of each input tensor is concatenated, followed by the second half
                      of each input tensor.

    Behavior:
        - Non-bimanual case (bimanual=False):
          Simply concatenates all input tensors along the last dimension.

        - Bimanual case (bimanual=True):
          1. Splits each input tensor into two halves along the last dimension.
          2. Concatenates all first halves in the order of input tensors.
          3. Concatenates all second halves in the order of input tensors.
          4. Concatenates the result of steps 2 and 3.

    Example:
        >>> t1 = torch.tensor([1, 2, 3, 4])
        >>> t2 = torch.tensor([5, 6, 7, 8])
        >>> order_data((t1, t2), bimanual=False)
        tensor([1, 2, 3, 4, 5, 6, 7, 8])
        >>> order_data((t1, t2), bimanual=True)
        tensor([1, 2, 5, 6, 3, 4, 7, 8])
    """
    if not bimanual:
        return torch.cat(tuple_data, dim=-1)

    halves = []
    for tensor in tuple_data:
        mid = ceil(tensor.shape[-1] / 2)
        halves.extend([tensor[..., :mid], tensor[..., mid:]])

    first_half = torch.cat(halves[::2], dim=-1)
    second_half = torch.cat(halves[1::2], dim=-1)

    return torch.cat([first_half, second_half], dim=-1)


def reconstruct_dict(merged_tensor, shape_dict,  device="cpu"):
    keys = list(shape_dict.keys())

    reconstructed_dict = {}
    start_idx = 0

    for key in keys:
        if key not in shape_dict:
            continue

        shape = shape_dict[key]
        end_idx = start_idx + shape[-1]

        if len(shape) == 1:
            reconstructed_tensor = merged_tensor[start_idx:end_idx]
        else:
            reconstructed_tensor = merged_tensor[..., start_idx:end_idx]

        reconstructed_dict[key] = reconstructed_tensor.reshape(shape).to(device)
        start_idx = end_idx

    return reconstructed_dict


def tensor_dict_to_np(dict):
    res = copy(dict)
    for k in dict:
        if isinstance(dict[k], torch.Tensor):
            res[k] = dict[k].cpu().numpy()
    return res


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_normalizers(policy_config, ckpt_dir, device, stats=None):
    """
        Initialize the normalize and unnormalize classes from the dataset stats
        if the stats are not provided then attempt to loaded them from the
        checkpoint directory
    """
    if stats is None:
        hf_stats_path = Path(os.path.join(ckpt_dir, f"stats.safetensors"))
        assert hf_stats_path.exists(), f"No dataset stats found! in {hf_stats_path}"
        stats = unflatten_dict(load_file(hf_stats_path, device=device))

    input_shapes = policy_config['training_data']['input_shapes']
    input_normalization_modes = policy_config['training_data']['input_normalization_modes']
    output_shapes = policy_config['training_data']['output_shapes']
    output_normalization_modes = policy_config['training_data']['output_normalization_modes']

    normalizers = {}
    normalizers['input'] = Normalize(input_shapes, input_normalization_modes, stats).cuda()
    normalizers['target'] = Normalize(output_shapes, output_normalization_modes, stats).cuda()
    normalizers['output'] = Unnormalize(output_shapes, output_normalization_modes, stats).cuda()

    return normalizers


def action_dict_to_robosuite(action_dict, orientation_representation, stiffness_representation, bimanual):
    action_dict = tensor_dict_to_np(action_dict)
    # convert to axis angle for the controller if necessary
    if orientation_representation == 'ortho6':
        rot = action_dict[f'action.rotation_{orientation_representation}']
        action_rotation = torch.zeros(6)
        action_rotation[:3] = torch.from_numpy(T.quat2axisangle(T.ortho62quat(rot[:6])))
        action_rotation[3:] = torch.from_numpy(T.quat2axisangle(T.ortho62quat(rot[6:])))
    elif orientation_representation == 'axis_angle':
        action_rotation = action_dict[f'action.rotation_{orientation_representation}']
    else:
        raise ValueError(f"Invalid orientation representation {orientation_representation}")

    # reconstruct the action as expected by the environment
    return order_data((
        action_dict[f'action.stiffness_{stiffness_representation}'],
        action_dict['action.position'],
        action_rotation,
        action_dict['action.gripper'],
    ), bimanual)


def get_random_batches(dataloader, num_batches):
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    dataset_size = len(dataset)

    # Calculate the total number of samples we need
    total_samples = num_batches * batch_size

    # Ensure we don't request more samples than available
    if total_samples > dataset_size:
        raise ValueError(f"Requested {total_samples} samples, but dataset only has {dataset_size}")

    # Get random indices
    indices = random.sample(range(dataset_size), total_samples)

    # Create a Subset of the dataset
    subset = Subset(dataset, indices)

    # Create a new DataLoader for this subset
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=dataloader.pin_memory, num_workers=dataloader.num_workers)

    return subset_loader


def get_random_batch(dataloader: DataLoader):
    # Get the dataset from the dataloader
    dataset = dataloader.dataset

    # Calculate the number of samples in a batch
    batch_size = dataloader.batch_size

    # Get a random starting index
    start_index = random.randint(0, len(dataset) - batch_size)

    # Create a Subset of the dataset for just this batch
    batch_subset = Subset(dataset, range(start_index, start_index + batch_size))

    # Create a new DataLoader for just this batch
    batch_loader = DataLoader(batch_subset, batch_size=batch_size, shuffle=False, pin_memory=dataloader.pin_memory, num_workers=dataloader.num_workers)

    # Get the single batch from this DataLoader
    return next(iter(batch_loader))