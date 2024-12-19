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

import glob
import os
import h5py
import numpy as np
from ur_control.transformations import axis_angle_from_quaternion, quaternion_from_axis_angle

# Configure numpy printing options
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class RotationConsistencyChecker:
    """
    A class to check and ensure rotation consistency in robotics datasets.
    """

    def __init__(self, dataset_path):
        """
        Initialize the RotationConsistencyChecker.

        Args:
            dataset_path (str): Path to the dataset directory.
        """
        self.dataset_path = dataset_path

    def check_rotation_consistency(self):
        self.check_attribute_rotation_consistency('action', indices=[15, 16, 17])
        self.check_attribute_rotation_consistency('observations/eef_pos', indices=[3, 4, 5])

    def check_attribute_rotation_consistency(self, attr_name='action', indices=[15, 16, 17]):
        """
        Check rotation consistency across episodes and timesteps.
        """
        a_first_quaternions, b_first_quaternions = self._collect_quaternions(attr_name, indices)
        self._check_episode_consistency(a_first_quaternions, 'a')
        if b_first_quaternions:
            self._check_episode_consistency(b_first_quaternions, 'b')

    def _collect_quaternions(self, attr_name='action', indices=[15, 16, 17]):
        """
        Collect quaternions from all episodes.

        Returns:
            tuple: Lists of quaternions for arm A and B (if bimanual).
        """
        a_first_quaternions, b_first_quaternions = [], []
        episodes = glob.glob(os.path.join(self.dataset_path, '*.hdf5'))
        if len(episodes) == 0:
            raise IOError("No dataset episodes found")
        for filepath in episodes:  # Assuming max 100 episodes
            with h5py.File(filepath, 'r') as root:
                actions = root[attr_name][:]
                is_bimanual = actions.shape[1] > 19
                self._process_episode(actions, a_first_quaternions, b_first_quaternions, is_bimanual, indices)
        return a_first_quaternions, b_first_quaternions

    def _process_episode(self, attributes, a_first_quaternions, b_first_quaternions, is_bimanual, indices):
        """
        Process attributes for a single episode.

        Args:
            attributes (np.array): Array of attributes.
            a_first_quaternions (list): List to store arm A quaternions.
            b_first_quaternions (list): List to store arm B quaternions.
            is_bimanual (bool): Whether the attributes are for a bimanual setup.
        """
        last_a_rot = last_b_rot = None
        for attr in attributes:
            if last_a_rot is None:
                last_a_rot = quaternion_from_axis_angle(attr[indices])
                a_first_quaternions.append(last_a_rot)
                if is_bimanual:
                    last_b_rot = quaternion_from_axis_angle(attr[-4:-1])
                    b_first_quaternions.append(last_b_rot)
                continue

            a_rot = quaternion_from_axis_angle(attr[indices])
            self._assert_consistent_sign(a_rot, last_a_rot, 'a')
            last_a_rot = a_rot.copy()

            if is_bimanual:
                b_rot = quaternion_from_axis_angle(attr[-4:-1])
                self._assert_consistent_sign(b_rot, last_b_rot, 'b')
                last_b_rot = b_rot.copy()

    @staticmethod
    def _assert_consistent_sign(current_rot, last_rot, arm_name):
        """
        Assert that the sign of the largest component is consistent.

        Args:
            current_rot (np.array): Current rotation quaternion.
            last_rot (np.array): Previous rotation quaternion.
            arm_name (str): Name of the arm ('a' or 'b').
        """
        idx = np.argmax(np.abs(current_rot))

        if np.sign(current_rot[idx]) != np.sign(last_rot[idx]):
            raise ValueError(f"Flipped! {arm_name}: {current_rot} {last_rot}")

    @staticmethod
    def _check_episode_consistency(first_quaternion, arm_name):
        """
        Check rotation consistency between episodes.

        Args:
            first_quaternion (list): List of the first quaternion of each episode.
            arm_name (str): Name of the arm ('a' or 'b').
        """
        for i in range(len(first_quaternion) - 1):
            idx = np.argmax(np.abs(first_quaternion[i]))
            if np.sign(first_quaternion[i][idx]) != np.sign(first_quaternion[i+1][idx]):
                ValueError(f"Flipped! {arm_name} {i}: {first_quaternion[i]} {first_quaternion[i+1]}")


class RotationConverter:
    """
    A class to convert and fix rotations in robotics datasets.
    """

    def __init__(self, dataset_path):
        """
        Initialize the RotationConverter.

        Args:
            dataset_path (str): Path to the dataset directory.
        """
        self.dataset_path = dataset_path

    def fix_rotations(self):
        """
        Fix rotations in all episodes of the dataset.
        """
        episodes = glob.glob(os.path.join(self.dataset_path, '*.hdf5'))
        assert len(episodes) > 0, "No dataset episodes found"
        for episode_filepath in episodes:
            self._fix_episode_rotations(episode_filepath)

    def _fix_episode_rotations(self, filepath):
        """
        Fix rotations for a single episode.

        Args:
            filepath (str): Path to the episode file.
        """
        with h5py.File(filepath, 'a') as root:
            actions = root['action'][:]
            ee_pos = root['observations/eef_pos'][:]
            is_bimanual = actions.shape[1] > 19
            fixed_actions = self._convert_actions(actions, is_bimanual)
            fixed_ee_poses = self._convert_ee_pos(ee_pos, is_bimanual)

            if 'observations/raw_eef_pos' not in root:
                root['observations'].create_dataset('raw_eef_pos', (root.attrs['max_timesteps'], len(ee_pos[0])))
                root['observations/raw_eef_pos'][...] = ee_pos
            if 'raw_action' not in root:
                root.create_dataset('raw_action', (root.attrs['max_timesteps'], len(actions[0])))
                root['raw_action'][...] = actions
            root['action'][...] = fixed_actions
            root['observations/eef_pos'][...] = fixed_ee_poses

    @staticmethod
    def _convert_actions(actions, is_bimanual):
        """
        Convert actions by fixing rotations.

        Args:
            actions (np.array): Array of actions.
            is_bimanual (bool): Whether the actions are for a bimanual setup.

        Returns:
            list: Fixed actions.
        """
        fixed_actions = []
        for action in actions:
            a_q = quaternion_from_axis_angle(action[15:18])
            fixed_action = np.copy(action)
            fixed_action[15:18] = axis_angle_from_quaternion(a_q)

            if is_bimanual:
                b_q = quaternion_from_axis_angle(action[-4:-1])
                fixed_action[-4:-1] = axis_angle_from_quaternion(b_q)

            fixed_actions.append(fixed_action)
        return fixed_actions

    @staticmethod
    def _convert_ee_pos(ee_poses, is_bimanual):
        """
        Convert ee_poses by fixing rotations.

        Args:
            ee_poses (np.array): Array of ee_poses.
            is_bimanual (bool): Whether the ee_poses are for a bimanual setup.

        Returns:
            list: Fixed ee_poses.
        """
        fixed_ee_poses = []
        for ee_pos in ee_poses:
            a_q = quaternion_from_axis_angle(ee_pos[3:6])
            fixed_action = np.copy(ee_pos)
            fixed_action[3:6] = axis_angle_from_quaternion(a_q)

            if is_bimanual:
                b_q = quaternion_from_axis_angle(ee_pos[-4:-1])
                fixed_action[-4:-1] = axis_angle_from_quaternion(b_q)

            fixed_ee_poses.append(fixed_action)
        return fixed_ee_poses


def main():
    dataset_path = '/root/osx-ur/dependencies/act/datasets/bimanual_round_insertion_variable_v4'
    dataset_path = '/root/osx-ur/dependencies/act/datasets/bimanual_round_insertion_variable_v3'
    dataset_path = '/root/osx-ur/dependencies/act/datasets/b_bot_wiping_v5'
    # dataset_path = '/root/osx-ur/dependencies/act/datasets/sim_bimanual_wiping'

    try:
        checker = RotationConsistencyChecker(dataset_path)
        checker.check_rotation_consistency()
    except ValueError:
        confirmation = input("Inconsistency detected! Press Y to fix dataset: ")
        if confirmation.lower() == 'y':
            converter = RotationConverter(dataset_path)
            converter.fix_rotations()
        else:
            print("Dataset not modified.")
            return

        try:
            checker = RotationConsistencyChecker(dataset_path)
            checker.check_rotation_consistency()
        except ValueError:
            print("Dataset still has issues!!")
            return

        print("Dataset updated and verified!")
        return

    print("Dataset has no issues.")


if __name__ == "__main__":
    main()
