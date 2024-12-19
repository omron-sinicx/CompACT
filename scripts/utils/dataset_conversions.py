import glob
from math import ceil
import os
import traceback
import h5py
import numpy as np
import tqdm

from osx_robot_control.math_utils import cholesky_vector_to_spd
# from osx_teleoperation.debug_utils import print_action_diag_ortho6
from ur_control.transformations import ortho6_from_axis_angle

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def get_stiffness_diag(cholesky_vector):
    return np.diag(cholesky_vector_to_spd(cholesky_vector))


def convert_rotations_to_ortho6(dataset_path):
    with h5py.File(dataset_path, 'r+') as root:
        eef_pos = root['observations']['eef_pos'][:]

        bimanual = eef_pos.shape[1] > 7

        rotations = [eef_pos[:, 3:6]]
        if bimanual:
            idx = int(eef_pos.shape[1]/2)
            rotations.append(eef_pos[:, -4:-1])
        else:
            idx = 7

        rotations = [np.apply_along_axis(ortho6_from_axis_angle, 1, rot) for rot in rotations]

        # print(f"{rotations[1].shape=}")

        if 'observations/eef_pos_ortho6' not in root:
            eef_ortho6_dim = len(rotations)*3 + eef_pos.shape[1]
            root['observations'].create_dataset('eef_pos_ortho6', (root.attrs['max_timesteps'], eef_ortho6_dim))

        eef_ortho6 = np.hstack((eef_pos[:, :3], rotations[0], eef_pos[:, idx-1].reshape((-1, 1))))
        if len(rotations) > 1:
            eef_ortho6 = np.hstack((eef_ortho6, eef_pos[:, idx:idx+3], rotations[1], eef_pos[:, -1].reshape((-1, 1))))

        root['observations']['eef_pos_ortho6'][...] = eef_ortho6

        # print(f"{eef_pos.shape=} {eef_pos[0]}")
        # print(f"{eef_ortho6.shape=} {eef_ortho6[0]}")


def convert_actions(dataset_path):
    with h5py.File(dataset_path, 'r+') as root:
        if 'raw_action' in root:
            action = root['raw_action'][:]
        else:
            action = root['action'][:]

        # print(f"{root['action'][:][0]=}")

        bimanual = action.shape[1] > 19

        # Convert rotations to ortho6
        rotations = [action[:, 15:18]]
        if bimanual:
            idx = ceil(action.shape[1]/2)
            rotations.append(action[:, -4:-1])
        else:
            idx = 19

        rotations = [np.apply_along_axis(ortho6_from_axis_angle, 1, rot) for rot in rotations]

        # print(f"{rotations[1].shape=}")

        if 'action_cholesky_ortho6' not in root:
            action_cholesky_ortho6_dim = len(rotations)*3 + action.shape[1]
            root.create_dataset('action_cholesky_ortho6', (root.attrs['max_timesteps'], action_cholesky_ortho6_dim))

        action_cholesky_ortho6 = np.hstack((action[:, :15], rotations[0], action[:, idx-1].reshape((-1, 1))))
        if bimanual:
            if action.shape[1] % 2 != 0:
                action_cholesky_ortho6 = np.hstack((action_cholesky_ortho6, action[:, idx:idx+15], rotations[1]))
            else:
                action_cholesky_ortho6 = np.hstack((action_cholesky_ortho6, action[:, idx:idx+15], rotations[1], action[:, -1].reshape((-1, 1))))

        # print(f"{action_cholesky_ortho6_dim=}")
        # print(f"{action.shape=} {action[0]}")
        # print(f"{action_cholesky_ortho6.shape=} {action_cholesky_ortho6[0]}")
        root['action_cholesky_ortho6'][...] = action_cholesky_ortho6

        # convert stiffness to diagonal
        stiffness_cholesky = [action[:, :6], action[:, 6:12]]
        if bimanual:
            stiffness_cholesky.append(action[:, idx:idx+6])
            stiffness_cholesky.append(action[:, idx+6:idx+12])

        stiffness_diag = [np.apply_along_axis(get_stiffness_diag, 1, cholesky_vector) for cholesky_vector in stiffness_cholesky]

        if 'action_diag_ortho6' not in root:
            action_diag_ortho6_dim = len(rotations)*3 + action.shape[1] - len(stiffness_diag)*3
            root.create_dataset('action_diag_ortho6', (root.attrs['max_timesteps'], action_diag_ortho6_dim))

        action_diag_ortho6 = np.hstack((stiffness_diag[0], stiffness_diag[1], action[:, 12:15], rotations[0], action[:, idx-1].reshape((-1, 1))))

        if bimanual:
            if action.shape[1] % 2 != 0:
                action_diag_ortho6 = np.hstack((action_diag_ortho6, stiffness_diag[2], stiffness_diag[3], action[:, idx+12:idx+15], rotations[1]))
            else:
                action_diag_ortho6 = np.hstack((action_diag_ortho6, stiffness_diag[2], stiffness_diag[3], action[:, idx+12:idx+15], rotations[1], action[:, -1].reshape((-1, 1))))

        root['action_diag_ortho6'][...] = action_diag_ortho6
        # print_action_diag_ortho6(action_diag_ortho6)
        # exit(0)
        # print(f"{action.shape=} {action[0]}")
        # print(f"{action_diag_ortho6.shape=} \n{action_diag_ortho6[0][:16]} \n{action_diag_ortho6[0][16:]}")


def find_hdf5_files(directory):
    # Use os.path.join to create a platform-independent path
    search_pattern = os.path.join(directory, '**', '*.hdf5')

    # Use glob with recursive=True to search subdirectories
    hdf5_files = glob.glob(search_pattern, recursive=True)

    return hdf5_files


datasets_root = '/root/osx-ur/dependencies/act/datasets/bimanual_round_insertion_variable_v5'
files = find_hdf5_files(datasets_root)

for f in tqdm.tqdm(files):
    try:
        convert_rotations_to_ortho6(f)
        convert_actions(f)
    except Exception as e:
        print("Faulty file", f)
        print(traceback.format_exc())
        break
# 1. action stiffness to diag only
# 2. action/eef_pos to ortho6
