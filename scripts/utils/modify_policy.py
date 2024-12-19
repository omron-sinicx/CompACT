import glob
import h5py
import numpy as np
dataset_path = '/root/osx-ur/dependencies/act/datasets/episode_0.hdf5'
dataset_path = '/root/osx-ur/dependencies/act/datasets/bimanual_round_insertion_variable_v3/'

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def distance_to_angle(distance):
    max_gap = 0.085
    max_angle = 0.8028
    distance = np.clip(distance, 0, max_gap)
    angle = (max_gap - distance) * max_angle / max_gap
    return angle


# a_bot always 0.330
# b_bot 0.045 to 0.987

files = glob.glob(dataset_path + '*.hdf5')

for f in files:
    with h5py.File(f, 'r+') as root:
        eef_pos = root['observations']['eef_pos'][:]
        action = root['action'][:]
        if 'observations/raw_eef_pos' not in root:
            root['observations'].create_dataset('raw_eef_pos', (root.attrs['max_timesteps'], len(eef_pos[0])))
            root['observations/raw_eef_pos'][...] = eef_pos

        action_idx = int(action.shape[1]/2)-1
        eef_pos_idx = int(eef_pos.shape[1]/2)-1

        # print(f"{eef_pos[:, -1]=}")
        eef_pos[:, eef_pos_idx] = eef_pos[:, eef_pos_idx] * 0.0 + 0.330
        eef_pos[:, -1] = 1 - eef_pos[:, -1] * 0.085 / 0.8028
        root['observations']['eef_pos'][...] = eef_pos
        # new = np.apply_along_axis(distance_to_angle, 0, new)
        print(f"{eef_pos[:, eef_pos_idx]=}")
        print(f"{eef_pos[:, -1]=}")

        # print(f"{action[:, -1]=}")
        break
