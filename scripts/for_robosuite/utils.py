import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
import robosuite.utils.transform_utils as T


def format_observations(env: ManipulationEnv, obs, camera_names):

    # store the data as dictionaries
    obs_formatted_dict = {}

    # Gripper actions
    obs_formatted_dict['observation.gripper'] = []
    for robot in env.robots:
        if len(robot.gripper.current_action) > 0:
            obs_formatted_dict['observation.gripper'].append(robot.gripper.current_action)  # TODO: Make this part of the env's obs

    # Joint position and Velocity
    gripper1 = env.robots[1].gripper.current_action.tolist()
    obs_formatted_dict['observation.qpos'] = np.concatenate([obs['robot0_joint_pos'], env.robots[0].gripper.current_action, obs['robot1_joint_pos'], gripper1])
    obs_formatted_dict['observation.qvel'] = np.concatenate([obs['robot0_joint_vel'], env.robots[0].gripper.current_action, obs['robot1_joint_vel'], gripper1])

    # Force/Torque
    obs_formatted_dict['observation.ft'] = np.concatenate([obs['robot0_eef_force_torque'], obs['robot1_eef_force_torque']])

    # End-Effector's Cartesian pose and velocity
    obs_formatted_dict['observation.eef.position'] = [obs['robot0_eef_pos'], obs['robot1_eef_pos']]
    obs_formatted_dict['observation.eef.linear_velocity'] = [obs['robot0_eef_vel_lin'], obs['robot1_eef_vel_lin']]
    obs_formatted_dict['observation.eef.angular_velocity'] = [obs['robot0_eef_vel_ang'], obs['robot1_eef_vel_ang']]
    obs_formatted_dict['observation.eef.rotation_ortho6'] = [T.quat2ortho6(obs['robot0_eef_quat']), T.quat2ortho6(obs['robot1_eef_quat'])]

    for k in obs_formatted_dict:
        obs_formatted_dict[k] = np.array(obs_formatted_dict[k]).flatten()

    # Cameras
    for cam_name in camera_names:
        obs_formatted_dict[f'observation.images.{cam_name}'] = obs[f'{cam_name}_image']

    return obs_formatted_dict


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def dict_np_to_torch(d):
    res = {}
    for k in d:
        res[k] = torch.from_numpy(d[k])
    return res


def visualize_ft(times, forces_hist, stiffness_hist, fig, axs: plt.Axes, axs_twin: plt.Axes, arm):
    labels = ['Fx', 'Fy', 'Fz']
    colors = ['r', 'g', 'b']
    # Update the force subplot
    offset = 6 if arm == "left" else 0

    axs.clear()

    for i in range(3):
        axs.plot(times, np.array(forces_hist)[:, i+offset], label=labels[i], color=colors[i], alpha=0.8, linewidth=1)

    axs.set_ylabel('force')
    axs.set_ylim(-50, 50)
    axs.legend(loc='upper left')
    axs.grid()

    axs_twin.clear()
    color = 'tab:olive'
    axs_twin.yaxis.set_label_position("right")
    axs_twin.plot(times, stiffness_hist, color=color)
    axs_twin.set_ylabel('stiffness', color=color)
    axs_twin.tick_params(axis='y', color=color)
    axs_twin.set_ylim(0, 1000)

    fig.tight_layout()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def get_action(env, device_action, control_delta, arm, add_noise, base_stiffness):
    position_noise = np.zeros(3) if not add_noise else np.random.uniform(low=-0.0005, high=0.0005, size=3)
    ori_noise = np.zeros(3) if not add_noise else np.random.uniform(low=-np.deg2rad(0.05), high=np.deg2rad(0.05), size=3)
    noise = np.concatenate([position_noise, ori_noise])

    if not control_delta:
        robot0_eef_pos = env.robots[0].controller.ee_pos
        robot0_eef_rot = T.mat2ortho6(env.robots[0].controller.ee_ori_mat)
        robot1_eef_pos = env.robots[1].controller.ee_pos
        robot1_eef_rot = T.mat2ortho6(env.robots[1].controller.ee_ori_mat)
    else:
        robot0_eef_pos = np.zeros(3)
        robot0_eef_rot = np.zeros(3)
        robot1_eef_pos = np.zeros(3)
        robot1_eef_rot = np.zeros(3)
    robot0_gripper, robot1_gripper = 0.0, 0.0

    # Set teleoperation command
    if arm == "left":
        robot0_eef_pos = device_action[:3] + noise[:3]
        robot0_eef_rot = device_action[3:6] + noise[3:]
        if not control_delta:
            robot0_eef_rot = T.axis_angle2ortho6(robot0_eef_rot)
        robot0_gripper = device_action[-1]
    if arm == "right":
        robot1_eef_pos = device_action[:3] + noise[:3]
        robot1_eef_rot = device_action[3:6] + noise[3:]
        if not control_delta:
            robot1_eef_rot = T.axis_angle2ortho6(robot1_eef_rot)
        robot1_gripper = device_action[-1]

    # diagonal stiffness position(3) rotation(3)
    stiffness_diag = base_stiffness*np.ones(6)
    # full stiffness (cholesky) position(6) rotation(6)
    stiffness_choleskly = np.concatenate([T.spd_to_cholesky_vector(np.eye(3)*base_stiffness), T.spd_to_cholesky_vector(np.eye(3)*base_stiffness)])

    action_dict = {}
    action_dict['action.position'] = [robot0_eef_pos, robot1_eef_pos]
    if control_delta:
        action_dict['action.rotation_delta'] = [robot0_eef_rot, robot1_eef_rot]  # 3D for deltas, 6D for absolute rotation
    else:
        action_dict['action.rotation_ortho6'] = [robot0_eef_rot, robot1_eef_rot]  # 3D for deltas, 6D for absolute rotation
    action_dict['action.stiffness_diag'] = [stiffness_diag, stiffness_diag]
    action_dict['action.stiffness_cholesky'] = [stiffness_choleskly, stiffness_choleskly]
    if len(env.robots[1].gripper.current_action) > 1:
        action_dict['action.gripper'] = [np.array([robot0_gripper]), np.array([robot1_gripper])]
    else:
        action_dict['action.gripper'] = [np.array([robot0_gripper])]
    return action_dict


def get_device(args):
    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "gamepad":
        from robosuite.devices import GamePad
        device = GamePad(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard', 'gamepad', or 'spacemouse'.")

    return device


def flatten_np_dict(dict):
    res = copy.copy(dict)
    for key in dict:
        res[key] = np.array(dict[key]).flatten()
    return res
