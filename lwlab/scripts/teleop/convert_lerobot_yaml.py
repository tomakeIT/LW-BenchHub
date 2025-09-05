import os
import argparse
import yaml
from pathlib import Path

import h5py
import numpy as np
import tqdm
import cv2
import ast
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def convert_isaaclab_to_lerobot(args, config):
    # Load configuration: features, robot_type, and default task
    features = config["features"]
    robot_type = config["robot_type"]
    default_task = config.get("default_task", "UnknownTask")
    # Create LeRobot dataset
    repo_id = args.tgt_repo_id or f"{Path(args.src_hdf5).stem}"
    root = Path(args.src_video_dir) / repo_id

    if root.exists():
        raise ValueError(f"Target dataset directory {root} already exists. Please delete it or specify another path.")

    AGENTVIEW_MAIN = f"{Path(args.src_hdf5).stem}.mp4"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(root),
        fps=25,
        robot_type=robot_type,
        features=features,
    )

    # Load hdf5 file
    with h5py.File(args.src_hdf5, "r") as f:
        demo_names = list(f["data"].keys())
        episode_count = len(demo_names)
        print(f"Found {len(demo_names)} demos: {demo_names}")
        demo_names.sort(key=lambda x: int(x.split("_")[-1]))
        for i in tqdm.tqdm([episode_count - 1], desc="Convert last demo"):
            demo_name = demo_names[i]
            demo_group = f["data"][demo_name]

            robot_joint_pos = np.array(demo_group["obs/joint_pos"])
            actions = np.array(demo_group["joint_targets/joint_pos_target"])
            T = robot_joint_pos.shape[0]
            print(f"Demo {demo_name}: {T} frames, joint state shape: {robot_joint_pos.shape}, action shape: {actions.shape}")

            # Prepare video path
            if args.src_video_dir:
                video_paths = {
                    "observation.images.agentview_main": Path(args.src_video_dir) / AGENTVIEW_MAIN,
                }
            else:
                video_paths = {
                    "observation.images.agentview_main": "",
                }

            # Extract task description from ep_meta
            task = default_task
            # Frame-wise write
            cap_main = cv2.VideoCapture(str(video_paths["observation.images.agentview_main"]))

            for i in tqdm.tqdm(range(T), desc="Processing frames"):
                ret_main, img_main = cap_main.read()
                img_main = cv2.cvtColor(img_main, cv2.COLOR_BGR2RGB)
                num_elements = actions[i].shape[0] // 2
                frame = {
                    "observation.images.agentview_main": img_main,
                    "observation.state": np.array(robot_joint_pos[i]).astype(np.float32),
                    "action": np.array(actions[i][-num_elements:], dtype=np.float32),
                    # "action": np.array(actions[i], dtype=np.float32),
                }
                dataset.add_frame(frame, task=task)
            dataset.save_episode()
            cap_main.release()

    print(f"Conversion completed. Dataset saved to: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_hdf5", type=str, required=True, help="Path to lwlab-style HDF5 file")
    parser.add_argument("--src_video_dir", type=str, required=True, help="Directory containing the video files")
    parser.add_argument("--tgt_repo_id", type=str, default=None, help="LeRobot dataset repo_id (folder name)")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config_yaml, "r") as f:
        config = yaml.safe_load(f)
    config['features']['observation.state']['shape'] = ast.literal_eval(config['features']['observation.state']['shape'])
    config['features']['action']['shape'] = ast.literal_eval(config['features']['action']['shape'])
    convert_isaaclab_to_lerobot(args, config)
