
from math import ceil
import os
from pathlib import Path
import pprint
from safetensors.torch import load_file, save_file

import torch
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import calculate_episode_data_index, flatten_dict, hf_transform_to_torch, load_hf_dataset
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from osx_robot_control.math_utils import cholesky2diag

from ur_control.transformations import ortho6_from_axis_angle

torch.set_printoptions(precision=4, sci_mode=False, linewidth=1000)


def split_eef_ortho6(eef_pos):
    position = eef_pos[:3]
    rotation = eef_pos[3:9]
    gripper = eef_pos[9]
    return position, rotation, gripper


def split_eef_axis_angle(eef_pos):
    position = eef_pos[:3]
    rotation = torch.from_numpy(ortho6_from_axis_angle(eef_pos[3:6]))
    gripper = eef_pos[6]
    return position, rotation, gripper


def split_action_diag_ortho6(action):
    stiffness = action[:6]
    position = action[6:9]
    rotation = action[9:15]
    gripper = action[15]
    return stiffness, position, rotation, gripper


def split_action_cholesky_axis_angle(action):
    stiffness = torch.cat((torch.from_numpy(cholesky2diag(action[:6])),
                           torch.from_numpy(cholesky2diag(action[6:12]))))
    position = action[12:15]
    rotation = torch.from_numpy(ortho6_from_axis_angle(action[15:18]))
    if len(action) == 19:
        gripper = action[18]
    else:
        gripper = None
    return stiffness, position, rotation, gripper


def split_eef_and_action(data):
    eef_key = 'observation.eef_pos'
    eef_dim = data[eef_key].shape[0]
    device = data[eef_key].device

    if 'action_diag_ortho6' in data:
        del data['action_diag_ortho6']
    if 'action_cholesky_ortho6' in data:
        del data['action_cholesky_ortho6']
    if 'observation.state' in data:
        del data['observation.state']
    if 'observation.eef_pos_ortho6' in data:
        del data['observation.eef_pos_ortho6']

    if bimanual:
        idx = ceil(eef_dim/2)
        position1, rotation1, gripper1 = split_eef_axis_angle(data[eef_key][:idx])
        position2, rotation2, gripper2 = split_eef_axis_angle(data[eef_key][idx:])
        data['observation.eef_pos.position'] = torch.concat((position1, position2))
        data['observation.eef_pos.rotation_ortho6'] = torch.concat((rotation1, rotation2))
        data['observation.eef_pos.gripper'] = torch.Tensor([gripper1, gripper2], device=device)
        data['observation.eef_pos.rotation_axis_angle'] = torch.concat((data[eef_key][3:6], data[eef_key][idx+3:idx+6]))
    else:
        position, rotation, gripper = split_eef_axis_angle(data[eef_key])
        data['observation.eef_pos.position'] = position
        data['observation.eef_pos.rotation_ortho6'] = rotation
        data['observation.eef_pos.gripper'] = torch.Tensor([gripper])
        data['observation.eef_pos.rotation_axis_angle'] = data[eef_key][3:6]

    action_key = 'action'
    action_dim = data[action_key].shape[0]
    if not sim:
        # 3 + 3 + 3 + 6 + 1 = 16 x robots
        split_action = split_action_diag_ortho6
    else:
        # 6 + 6 + 3 + 3 + 1 = 19 x robots
        split_action = split_action_cholesky_axis_angle

    if bimanual:
        idx = ceil(action_dim/2)
        stiffness1, position1, rotation1, gripper1 = split_action(data[action_key][:idx])
        stiffness2, position2, rotation2, gripper2 = split_action(data[action_key][idx:])
        if debug:
            print(f"{stiffness1=}")
            print(f"{position1=}")
            print(f"{rotation1=}")
            print(f"{gripper1=}")
            print(f"{stiffness2=}")
            print(f"{position2=}")
            print(f"{rotation2=}")
            print(f"{gripper2=}")
        data['action.stiffness_diag'] = torch.concat((stiffness1, stiffness2))
        data['action.position'] = torch.concat((position1, position2))
        data['action.rotation_ortho6'] = torch.concat((rotation1, rotation2))
        if gripper2:
            data['action.gripper'] = torch.Tensor([gripper1, gripper2], device=device)
        else:
            data['action.gripper'] = torch.Tensor([gripper1])

        idx = ceil(data['action'].shape[0]/2)
        data['action.rotation_axis_angle'] = torch.concat((data['action'][15:18], data['action'][idx+15:idx+18]))
        data['action.stiffness_cholesky'] = torch.concat((data['action'][:12], data['action'][idx:idx+12]))
    else:
        stiffness, position, rotation, gripper = split_action(data[action_key])
        data['action.stiffness_diag'] = stiffness
        data['action.position'] = position
        data['action.rotation_ortho6'] = rotation
        data['action.gripper'] = torch.Tensor([gripper])
        data['action.stiffness_cholesky'] = data['action'][:12]
        data['action.rotation_axis_angle'] = data['action'][15:18]

    if debug:
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                print(k, data[k].shape)
        pprint.pprint(data)
        exit(0)

    return data


def split_data(hf_dataset):
    # Split eef_pos and action in its components and compute the stats
    updated_dataset = hf_dataset.map(split_eef_and_action, remove_columns=hf_dataset.column_names)

    updated_dataset = updated_dataset.with_format(None)  # to remove transforms that cant be saved

    updated_dataset.save_to_disk(Path(root) / repo_id / "train2")

    episode_data_index = calculate_episode_data_index(updated_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": 50,
        "video": True}

    videos_dir = Path(root) / repo_id / "video"

    updated_dataset.set_transform(hf_transform_to_torch)
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=updated_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, 32, 8)

    meta_data_dir = Path(root) / repo_id / "meta_data2"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)


def overwrite_stats(dataset_path):
    hf_stats_path = Path(os.path.join(dataset_path, f"stats.safetensors"))
    assert hf_stats_path.exists(), "No dataset stats found!"
    hf_stats = load_file(hf_stats_path, device="cpu")

    # pprint.pprint(hf_stats)
    pprint.pprint(hf_stats['observation.eef_pos.rotation_ortho6/min'])
    pprint.pprint(hf_stats['observation.eef_pos.rotation_ortho6/max'])
    pprint.pprint(hf_stats['observation.eef_pos.rotation_axis_angle/min'])
    pprint.pprint(hf_stats['observation.eef_pos.rotation_axis_angle/max'])

    bimanual = hf_stats['observation.eef_pos.rotation_ortho6/min'].shape[0] > 6
    # override stats for rotation
    hf_stats['observation.eef_pos.rotation_ortho6/min'][:6] = torch.ones(6) * -1
    hf_stats['observation.eef_pos.rotation_ortho6/max'][:6] = torch.ones(6)
    hf_stats['action.rotation_ortho6/min'][:6] = torch.ones(6) * -1
    hf_stats['action.rotation_ortho6/max'][:6] = torch.ones(6)
    hf_stats['observation.eef_pos.rotation_axis_angle/min'][:3] = torch.ones(3) * torch.pi * -1
    hf_stats['observation.eef_pos.rotation_axis_angle/max'][:3] = torch.ones(3) * torch.pi
    if bimanual:
        hf_stats['observation.eef_pos.rotation_ortho6/min'][6:] = torch.ones(6) * -1
        hf_stats['observation.eef_pos.rotation_ortho6/max'][6:] = torch.ones(6)
        hf_stats['action.rotation_ortho6/min'][6:] = torch.ones(6) * -1
        hf_stats['action.rotation_ortho6/max'][6:] = torch.ones(6)
        hf_stats['observation.eef_pos.rotation_axis_angle/min'][3:] = torch.ones(3) * torch.pi * -1
        hf_stats['observation.eef_pos.rotation_axis_angle/max'][3:] = torch.ones(3) * torch.pi
    pprint.pprint(hf_stats['observation.eef_pos.rotation_ortho6/min'])
    pprint.pprint(hf_stats['observation.eef_pos.rotation_ortho6/max'])
    save_file(flatten_dict(hf_stats), hf_stats_path)


# GLOBAL VARIABLES!
sim = True
bimanual = True
debug = True


repo_id = "comp-act/sim_bimanual_wiping"
root = "/root/osx-ur/dependencies/datasets/"
hf_dataset = load_hf_dataset(repo_id=repo_id, root=root, version=CODEBASE_VERSION, split="train")
split_data(hf_dataset)
overwrite_stats(Path(root) / repo_id / "meta_data2")
