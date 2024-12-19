#!/usr/bin/env python

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

"""Create dataset for ACT via teleoperation.

***Choose user input option with the --device argument***

Usage:
        $ python create_dataset_via_device_lerobot.py \
        --config_file /root/osx-ur/dependencies/act/config/sim_bimanual_wiping_compliance.yaml \
        --arm left \
        --device gamepad \
        --save \
        --ft \
        --other-arm-dataset-dir /root/osx-ur/dependencies/act/datasets/sim_both_wiping_fixed \
        --plot-ft \

    --environment: Robosuite environment to use. Default set to TwoArmWiping.
    --num-episode: Number of demonstrations you want to perform. Default set to 10.
    --dataset-dir: Directory to save the data. Default set to /root/dataset/abot_wiping.
    --device: Device to use for demonstration (keyboard/spacemouse/gamepad). Default set to keyboard.
    --ft: Whether to save F/T readings or not.
    --plot-ft: Simultaneously plot the force
    --control-hz: Control frequency
    --max-timesteps: Maximum number of timesteps. Default set to 200.
    --save: Whether you want to save the data or not.
    --other-arm-dataset-dir: Load and use the data from a previous demonstration for the other arm

    After demonstrating each episode, press "q" in keyboard / "START" in gamepad
    press "A" in gamepad to reset the episode without saving it
    to save the demo and reset the environment to
    proceed to the next demonstration.
"""

import av  # Import first to avoid issue with OpenCV
import argparse
import concurrent.futures
import json
import shutil
import timeit
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch
import tqdm
import yaml

from pathlib import Path
from PIL import Image

from act.utils import load_hf_dataset
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset

from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
import robosuite.utils.transform_utils as T
import robosuite as suite

from utils import format_observations, get_action, visualize_ft, get_device, flatten_np_dict


def save_image(img_tensor, key, frame_index, episode_index, videos_dir):
    assert isinstance(img_tensor, np.ndarray)
    img = Image.fromarray(img_tensor)
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def main(args):

    #### Hard-coded parameters ####
    num_image_writers_per_camera = 4
    run_compute_stats = True

    # Get controller config
    with open(args.config_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
        task_config = dataset_config['task_parameters']
        robosuite_config = dataset_config['robosuite']

    local_dir = Path(task_config['dataset_dir']).resolve()
    if local_dir.exists() and task_config.get('force_override', False):
        shutil.rmtree(local_dir)

    episodes_dir = local_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Logic to resume data recording
    rec_info_dir = episodes_dir / "data_recording_info.json"
    if rec_info_dir.exists():
        with open(rec_info_dir) as f:
            rec_info = json.load(f)
        episode_idx = rec_info["last_episode_index"] + 1
    else:
        episode_idx = 0

    control_frequency = int(robosuite_config['control_freq'])
    sleep_time = 1. / control_frequency
    max_timesteps = task_config['episode_len']
    camera_names = task_config["camera_names"]

    copy_dataset = args.copy_dataset
    if copy_dataset:
        assert args.other_arm_dataset_dir, "Required other_arm_dataset_dir to copy dataset"
    add_noise = False

    print("Control frequency", control_frequency)
    print("Max steps", max_timesteps)
    print("Camera names", task_config["camera_names"])

    # Create environment
    env = suite.make(
        **robosuite_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=task_config["camera_names"][0],
        ignore_done=True,
        use_camera_obs=True,
        camera_names=task_config["camera_names"],
        reward_shaping=True,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    print(f">>> Action dimension {env.action_dim} for env {robosuite_config['env_name']} <<<")

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Prepare teleoperation interface
    device = get_device(args)
    if args.device == 'keyboard':
        env.viewer.add_keypress_callback(device.on_press)

    if args.plot_ft:
        # init f/t visualizer
        fig, axs = plt.subplots(figsize=(6, 4))  # Two subplots for force and torque
        axs_twin = axs.twinx()
        fig.suptitle('Force Readings of the Right Arm')
        plt.show(block=False)

    # Load data for the other arm
    if args.other_arm_dataset_dir:
        other_arm_data = load_hf_dataset(args.other_arm_dataset_dir, ignore_videos=True)

    if not os.path.exists(task_config['dataset_dir']):
        os.mkdir(task_config['dataset_dir'])

    destfile = os.path.join(task_config['dataset_dir'], "config.yaml")
    shutil.copyfile(args.config_file, destfile)

    num_episodes = task_config['num_episodes']
    last_reset = 0

    futures = []
    num_image_writers = num_image_writers_per_camera * len(camera_names)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_image_writers) as executor:
        # demo (episode) loop
        while episode_idx < num_episodes:
            # Reset the environment
            obs = env.reset()

            # env.render()

            times = []
            force_hist = []
            stiffness_hist = []

            # Initialize device control
            device.start_control()

            episode_dict = {}
            frame_index = 0

            # Set active robot
            active_robot = env.robots[args.arm == "right"]

            last_stiffness_switch = 0

            stiffness_values = [100, 800]
            stiff_idx = 1
            base_stiffness = stiffness_values[stiff_idx]

            save_or_skip = False  # skip
            control_delta = robosuite_config['controller_configs']['control_delta']
            skip_frame = 5

            if args.other_arm_dataset_dir:
                other_arm_episode_len = other_arm_data.episode_data_index["to"][episode_idx].item() - other_arm_data.episode_data_index["from"][episode_idx].item()

            t = 0
            # action loop
            while True:
                start_time = timeit.default_timer()

                # Get the newest action
                device_action, _ = input2action(
                    device=device, robot=active_robot, active_arm=args.arm, env_configuration="single-arm-opposed", control_delta=control_delta
                )

                if args.device == "gamepad":
                    device_state = device.get_controller_state()
                    btn_state = device_state.get('buttons_state', False)
                    if btn_state:
                        if btn_state[0] and not last_stiffness_switch:  # X button
                            stiff_idx = (stiff_idx + 1) % len(stiffness_values)
                            base_stiffness = stiffness_values[stiff_idx]
                            print("New stiffness:", base_stiffness)
                        if btn_state[2] and not last_reset:  # A button
                            save_or_skip = False  # skip
                            last_reset = btn_state[2]
                            break

                    last_reset = btn_state[2]
                    last_stiffness_switch = btn_state[0]  # X button

                # If action is none, then this a reset so we should break
                if device_action is None:
                    save_or_skip = True
                    break

                action_dict = get_action(env, device_action, control_delta, args.arm, add_noise, base_stiffness)

                # load previously taken action for the other robot
                if args.other_arm_dataset_dir and not copy_dataset and t < other_arm_episode_len:
                    other_idx = other_arm_data.episode_data_index["from"][episode_idx].item() + t

                    for k in action_dict:
                        n = len(other_arm_data[other_idx][k])
                        if n % 2 != 0 and args.arm == "left":
                            action_dict[k] = other_arm_data[other_idx][k].numpy()
                        elif n % 2 == 0:
                            if args.arm == "right":
                                action_dict[k][1] = other_arm_data[other_idx][k][n//2:].numpy()
                            else:
                                action_dict[k][0] = other_arm_data[other_idx][k][:n//2].numpy()

                # Option to just copy one dataset to another
                if args.other_arm_dataset_dir and copy_dataset:
                    other_idx = other_arm_data.episode_data_index["from"][episode_idx].item() + t
                    if t < other_arm_episode_len:
                        for k in action_dict:
                            if len(other_arm_data[other_idx][k]) % 2 == 0:
                                action_dict[k] = np.split(other_arm_data[other_idx][k].numpy(), 2)
                            else:
                                action_dict[k] = [other_arm_data[other_idx][k].numpy()]
                    else:
                        save_or_skip = True
                        break

                # Step through the simulation and render
                obs, _, _, _ = env.step(flatten_np_dict(action_dict))
                observation = format_observations(env, obs, camera_names)

                # print("action", np.round(action, 3))
                env.render()

                image_keys = [key for key in observation if "image" in key]
                not_image_keys = [key for key in observation if "image" not in key]

                for key in image_keys:
                    futures += [
                        executor.submit(
                            save_image, observation[key], key, frame_index, episode_idx, videos_dir
                        )
                    ]

                observation = {k: torch.from_numpy(v) for k, v in observation.items()}

                for key in not_image_keys:
                    if key not in episode_dict:
                        episode_dict[key] = []
                    episode_dict[key].append(observation[key])

                for key in action_dict:
                    if key not in episode_dict:
                        episode_dict[key] = []
                    episode_dict[key].append(torch.from_numpy(np.array(action_dict[key]).flatten()))

                frame_index += 1

                computation_time = timeit.default_timer() - start_time
                print(f'Episode: {episode_idx:03d} Step: {t} duration: {computation_time: 0.3f}', end="\r")
                t += 1
                # sleep about the same as the control frequency
                if computation_time < sleep_time:
                    time.sleep(sleep_time - computation_time)

            # continue to save this episode
            if not args.save or not save_or_skip:
                # wait for images to be saved and then delete them as a skip has been invoked
                for _ in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Writing images"
                ):
                    pass
                for key in image_keys:
                    path = videos_dir / f"{key}_episode_{episode_idx:06d}"
                    shutil.rmtree(path)
                continue

            # save as HuggingFace format
            t0 = time.time()

            # During env reset we save the data and encode the videos
            num_frames = frame_index

            for key in image_keys:
                fname = f"{key}_episode_{episode_idx:06d}.mp4"
                video_dir = local_dir / "videos" / fname
                if video_dir.exists():
                    video_dir.unlink()
                # Store the reference to the video frame, even tho the videos are not yet encoded
                episode_dict[key] = []
                for i in range(num_frames):
                    episode_dict[key].append({"path": f"videos/{fname}", "timestamp": i / control_frequency})

            for key in not_image_keys:
                episode_dict[key] = torch.stack(episode_dict[key])

            for key in action_dict:
                episode_dict[key] = torch.stack(episode_dict[key])

            episode_dict["episode_index"] = torch.tensor([episode_idx] * num_frames)
            episode_dict["frame_index"] = torch.arange(0, num_frames, 1)
            episode_dict["timestamp"] = torch.arange(0, num_frames, 1) / control_frequency

            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True
            episode_dict["next.done"] = done

            ep_dir = episodes_dir / f"episode_{episode_idx}.pth"
            print("Saving episode dictionary...")
            torch.save(episode_dict, ep_dir)

            print(f'Saving time: {time.time() - t0:.1f} secs\n', end="\r")

            rec_info = {
                "last_episode_index": episode_idx,
            }
            with open(rec_info_dir, "w") as f:
                json.dump(rec_info, f)

            is_last_episode = (episode_idx == (num_episodes - 1))

            episode_idx += 1

            if is_last_episode:
                print("Done recording")
                print("Waiting for threads writing the images on disk to terminate...")
                for _ in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Writing images"
                ):
                    pass
                break

    if args.plot_ft:
        plt.close()

    image_keys = []

    print("Concatenating episodes")
    ep_dicts = []
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        ep_dir = episodes_dir / f"episode_{episode_idx}.pth"
        ep_dict = torch.load(ep_dir, weights_only=True)
        image_keys = [key for key in ep_dict if "image" in key]
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    print("Encoding episodes")
    # Use ffmpeg to convert frames stored as png into mp4 videos
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        for key in image_keys:
            tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_idx:06d}"
            fname = f"{key}_episode_{episode_idx:06d}.mp4"
            video_dir = local_dir / "videos" / fname
            if video_dir.exists():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
            # since video encoding with ffmpeg is already using multithreading.
            encode_video_frames(tmp_imgs_dir, video_dir, control_frequency, vcodec='libx264', overwrite=True)
            shutil.rmtree(tmp_imgs_dir)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = to_hf_dataset(data_dict, True)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": control_frequency,
        "video": True,
    }
    info["encoding"] = get_default_encoding()
    info["encoding"]["vcodec"] = 'libx264'

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=local_dir.name,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

    if run_compute_stats:
        print("Computing dataset statistics")
        stats = compute_stats(lerobot_dataset)
        lerobot_dataset.stats = stats
    else:
        stats = {}
        print("Skipping computation of the dataset statistics")

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    meta_data_dir = local_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    print(f'Saved to {task_config["dataset_dir"]}')
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Configuration file for dataset")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=.6, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=.4, help="How much to scale rotation user inputs")
    parser.add_argument("--save", action="store_true", help="Save the dataset")
    parser.add_argument("--ft", action="store_true", help="Save force-torque readings")
    parser.add_argument("--copy-dataset", action="store_true", help="Replay `other-arm` dataset and save into a new dataset")
    parser.add_argument("--plot-ft", action="store_true", help="Plot force-torque readings")
    parser.add_argument("--other-arm-dataset-dir", action="store", type=str, default=None,
                        help="Directory to load the previously recorded trajectory from the other arm.")
    args = parser.parse_args()

    main(args)
