#!/usr/bin/env python3
import ast
import os
import re
import sys
import yaml
import subprocess
import tempfile
import time
import shutil
import traceback
import csv
import random
from pathlib import Path

import cv2
from termcolor import colored

current_dir = Path(__file__).parent.parent.parent

from lwlab import CONFIGS_PATH  # noqa: E402

try:
    from robocasa_upload.joylo_uploader import JoyLoUploader
except ImportError:
    print("Warning: robocasa_upload.joylo_uploader not found")
    JoyLoUploader = None

# Pre-compile regex pattern for better performance
TIMESTAMP_VIDEO_PATTERN = re.compile(r'^\d{8}_\d{4}\.mp4$')


def load_layout_task_mapping(csv_file_path):
    """
    Load the mapping of robot-task combinations to layout configurations from a CSV file

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        dict: A dictionary mapping (robot, task) tuples to lists of layout configurations,
              each configuration containing:
            - robot: Robot type
            - layout: Layout ID
            - init_robot_base_pos: Robot initial position
            - init_robot_base_ori: Robot initial orientation
            - object_init_offset: Object initialization offset
    """
    robot_task_to_layouts = {}

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                robot = row['robot']
                task = row['task']
                layout = row['layout']
                try:
                    object_offset = ast.literal_eval(row['object_init_offset'])

                    robot_pos = None
                    robot_ori = None

                    if row['init_robot_base_pos'].strip() != 'None':
                        robot_pos = ast.literal_eval(row['init_robot_base_pos'])

                    if row['init_robot_base_ori'].strip() != 'None':
                        robot_ori = ast.literal_eval(row['init_robot_base_ori'])

                except (ValueError, SyntaxError) as e:
                    print(colored(f"Warning: Failed to parse parameters {row}: {e}", "yellow"))
                    continue

                layout_config = {
                    'robot': robot,
                    'layout': layout,
                    'object_init_offset': object_offset
                }

                if robot_pos is not None:
                    layout_config['init_robot_base_pos'] = robot_pos
                if robot_ori is not None:
                    layout_config['init_robot_base_ori'] = robot_ori

                robot_task_key = (robot, task)
                if robot_task_key not in robot_task_to_layouts:
                    robot_task_to_layouts[robot_task_key] = []
                robot_task_to_layouts[robot_task_key].append(layout_config)

    except FileNotFoundError:
        print(colored(f"warning: file {csv_file_path} is not exist", "yellow"))
        return {}
    except Exception as e:
        print(colored(f"error in loading file: {e}", "red"))
        return {}

    return robot_task_to_layouts


def get_random_layout_for_task(task_name, robot_type=None, csv_file_path=None):
    """
    Get a random layout configuration for a given robot-task combination.

    Args:
        task_name (str): Task name
        robot_type (str): Robot type
        csv_file_path (str): Path to the CSV file, if None then use the default path

    Returns:
        dict: A random layout configuration, including:
            - layout: Layout ID
            - object_init_offset: Object initialization offset
            - init_robot_base_pos: Robot initial position (if available for the robot type)
            - init_robot_base_ori: Robot initial orientation (if available for the robot type)
        If not found, return the default configuration
    """
    if csv_file_path is None:
        csv_file_path = Path(__file__).parent / 'layout_task_mapping.csv'

    if robot_type is None:
        print(colored("warning: No robot type specified, layout configuration cannot be selected", "red"))
        return None

    robot_task_to_layouts = load_layout_task_mapping(csv_file_path)
    robot_task_key = (robot_type, task_name)

    if robot_task_key in robot_task_to_layouts:
        layout_configs = robot_task_to_layouts[robot_task_key]
        selected_config = random.choice(layout_configs)
        available_layouts = [config['layout'] for config in layout_configs]

        print(colored(f"robot '{robot_type}' + task '{task_name}' Available layouts: {available_layouts}", "cyan"))
        print(colored(f"Randomly select layout: {selected_config['layout']}", "green"))
        print(colored(f"Object initialization offset: {selected_config['object_init_offset']}", "cyan"))

        if 'init_robot_base_pos' in selected_config:
            print(colored(f"robot Initial position: {selected_config['init_robot_base_pos']}", "cyan"))
        else:
            print(colored(f"robot '{robot_type}' ", "yellow"))

        if 'init_robot_base_ori' in selected_config:
            print(colored(f"robot初始方向: {selected_config['init_robot_base_ori']}", "cyan"))
        else:
            print(colored(f"robot '{robot_type}' No initial positional parameters are used", "yellow"))

        return_config = {
            'layout': selected_config['layout'],
            'object_init_offset': selected_config['object_init_offset']
        }

        if 'init_robot_base_pos' in selected_config:
            return_config['init_robot_base_pos'] = selected_config['init_robot_base_pos']
        if 'init_robot_base_ori' in selected_config:
            return_config['init_robot_base_ori'] = selected_config['init_robot_base_ori']

        return return_config
    else:
        print(colored(f"Warning: Robot not found '{robot_type}' + task '{task_name}' Corresponding layout, use the default layout", "yellow"))
        default_config = {
            'layout': 'robocasakitchen',
            'object_init_offset': [0.0, 0.0],
        }

        return default_config


def get_video_duration(video_path):
    """
    Get video duration in seconds

    Args:
        video_path (str): Path to the video file

    Returns:
        float: Video duration in seconds, or 0.0 if failed to read
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Unable to open video file: {video_path}")
            return 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        cap.release()

        if fps > 0:
            duration = frame_count / fps
            return round(duration, 2)
        else:
            return 0.0

    except Exception as e:
        print(f"Failed to obtain video duration: {e}")
        return 0.0


class TeleopDaemon:

    def __init__(self, params, on_exit_callback=None, uploader=None, task_name=None,
                 instance_id=None):
        self.params = params
        self.on_exit_callback = on_exit_callback
        self.process = None
        self.config_path = None
        self.temp_files = []
        self.is_running = False
        self.exit_handled = False
        self.uploader = uploader
        self.task_name = task_name
        self.instance_id = instance_id

    def create_config(self):
        default_config_path = CONFIGS_PATH / 'data_collection' / 'teleop' / 'teleop_base.yml'
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)

        cleaned_params = {}
        for key, value in self.params.items():
            if isinstance(value, Path):
                cleaned_params[key] = str(value)
            else:
                cleaned_params[key] = value

        config.update(cleaned_params)
        temp_fd, temp_yaml_path = tempfile.mkstemp(suffix='.yml', prefix='teleop_')
        temp_config_name = Path(temp_yaml_path).stem

        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(config, f,
                      default_flow_style=False,
                      indent=2,
                      sort_keys=False,
                      allow_unicode=True)

        config_dir = CONFIGS_PATH / 'data_collection' / 'teleop'
        final_config_path = config_dir / f'{temp_config_name}.yml'

        shutil.copy2(temp_yaml_path, final_config_path)

        self.temp_files = [temp_yaml_path, final_config_path]
        self.config_path = final_config_path

        return temp_config_name, config

    def start(self):
        """Start the teleop_main process"""
        temp_config_name, config = self.create_config()

        teleop_main_path = Path(__file__).parent / 'teleop_main.py'
        if self.params.get('enable_camera'):
            cmd = [
                sys.executable,
                str(teleop_main_path),
                f'--task_config={temp_config_name}',
                '--enable_camera',
            ]
        else:
            cmd = [
                sys.executable,
                str(teleop_main_path),
                f'--task_config={temp_config_name}',
            ]

        print("Start the teleoperation system...")
        print(f"settings: {self.config_path}")
        print(f"task: {config.get('task', 'Unknown')}")
        print(f"robot: {config.get('robot', 'Unknown')}")
        print(f"device: {config.get('teleop_device', 'Unknown')}")
        if config.get('record'):
            print(f"record to: {config.get('dataset_file', 'Unknown')}")
        print(f"command: {' '.join(cmd)}")
        print("-" * 50)

        self.process = subprocess.Popen(cmd, cwd=current_dir)
        self.is_running = True

        return self.process.pid

    def _handle_exit(self, return_code=None):
        """Handle the exit of the process"""
        if self.exit_handled:
            return

        self.exit_handled = True

        dataset_file = None
        video_files = None
        has_dataset = False
        has_videos = False

        if self.params.get('record'):
            dataset_file = './lwlab/datasets/dataset.hdf5'
            if dataset_file and os.path.exists(dataset_file):
                print(colored(f"Discovery data files: {dataset_file}", "green"))
                file_size = os.path.getsize(dataset_file) / (1024 * 1024)
                print(f"file size: {file_size:.2f} MB")
                has_dataset = True

        if self.params.get('save_video'):
            layout = self.params.get('layout', '')
            robot = self.params.get('robot', '')
            task = self.params.get('task', '')
            video_dir = f'./lwlab/datasets/{layout}_{robot}_{task}'
            if os.path.exists(video_dir):
                video_files_list = list(Path(video_dir).glob('*.mp4'))
                if video_files_list:
                    print(f"Discovery data files: {len(video_files_list)} 个")
                    for vf in video_files_list[:3]:
                        print(colored(f"Discovery data files: {vf}", "green"))
                    if len(video_files_list) > 3:
                        print(colored(f"... has {len(video_files_list) - 3} 个文件", "cyan"))

                    timestamp_videos = [vf for vf in video_files_list
                                        if TIMESTAMP_VIDEO_PATTERN.match(vf.name)]

                    if timestamp_videos:
                        timestamp_videos.sort(key=lambda x: x.name, reverse=True)
                        latest_video = timestamp_videos[0]
                        print(colored(f"Select the video file with the latest timestamp: {latest_video}", "green"))
                    else:
                        latest_video = video_files_list[0]
                        print(colored(f"No timestamp format video found, use: {latest_video}", "yellow"))

                    # Commented out unnecessary file size calculation for better performance
                    # total_size = sum(f.stat().st_size for f in video_files_list) / (1024 * 1024)
                    # print(f"视频总大小: {total_size:.2f} MB")
                    video_files = str(latest_video)
                    has_videos = True

        if not has_dataset and not has_videos:
            print(colored("No data files or video files were found, so the task was reset directly....", "yellow"))
            self._reset_task()
        else:
            print(colored("\nFound the upload configuration and started the upload process...", "green"))
            upload_success = self._upload_data(dataset_file, video_files)
            if not upload_success:
                print(colored("...Upload failed, reset task", "yellow"))
                self._reset_task()

        self._cleanup()

    def _reset_task(self):
        """Reset task using unified reset function"""
        reset_task(self.uploader)

    def _upload_data(self, recording_path, video_files):
        """upload data

        Returns:
            bool: True if upload was successful, False otherwise
        """
        print(colored("Uploader configuration detected", "green"))
        user_input = input(colored("Do you want to upload this record? (y/n): ", "cyan"))
        while True:
            if user_input.lower() == 'n':
                print(colored("skip upload", "yellow"))

                return False
            elif user_input.lower() == 'y':
                try:
                    print(colored("start upload...", "green"))

                    running_args = {
                        "task_name": self.task_name,
                        "instance_id": self.instance_id
                    }

                    video_duration = 0.0
                    if video_files and os.path.exists(video_files):
                        video_duration = get_video_duration(video_files)
                        print(colored(f"video len : {video_duration} s", "cyan"))

                    metadata = {
                        "is_success": True,
                        "recording_path": str(recording_path),
                        "params": self.params,
                        "checkpoints": [],
                        "video_duration": video_duration
                    }

                    result = self.uploader.upload(
                        self.task_name,
                        Path(recording_path),
                        Path(video_files) if video_files is not None else None,
                        2,  # 2: lwlab, 1: joylo
                        running_args,
                        metadata,
                    )

                    print(colored("upload success!", "green"))
                    print(f"upload result: {result}")
                    return True

                except Exception as e:
                    print(colored(f"upload failed: {e}", "red"))
                    print(traceback.format_exc())
                    return False

            else:
                print(colored("Invalid input, please enter y or n:", "red"))
                user_input = input()

    def _cleanup(self):
        """Clean up temporary files"""
        print(colored("\nCleaning up temporary files...", "green"))
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"delete: {temp_file}")
            except Exception as e:
                print(f"can not delete {temp_file}: {e}")

    def stop(self):
        """Stop the teleop process"""
        if self.process and self.is_running:
            print(colored("Stopping teleoperation process...", "red"))
            self.process.terminate()
            try:
                return_code = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(colored("Forcefully terminate a process...", "red"))
                self.process.kill()
                return_code = -1

            self.is_running = False
            self._handle_exit(return_code)

    def is_alive(self):
        """Check if the process is still running"""
        return self.is_running and self.process and self.process.poll() is None


def reset_task(uploader, context=""):
    """
    reset task

    Args:
        uploader: uploader instance
        context (str): context info for log
    """
    try:
        if context:
            print(colored(f"{context}，Resetting task...", "yellow"))
        else:
            print(colored("Resetting task...", "yellow"))
        uploader.update_task(mode="reset")
        print(colored("task reset success", "green"))
    except Exception as e:
        print(colored(f"task reset failed: {e}", "red"))


def main():
    """Main function with comprehensive error handling"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default="test HKU I2I")
    parser.add_argument('--batch_name', type=str, default="0818-33")
    parser.add_argument('--enable_camera', type=bool, default=True)
    parser.add_argument('--robot', type=str, default='DoublePiper-Abs')
    args = parser.parse_args()

    uploader = None
    task_name = None
    daemon = None

    try:
        uploader = JoyLoUploader(project_name=args.project_name)

        try:
            task_info = uploader.pull_task(batch_name=args.batch_name)
            if task_info is None:
                raise ValueError(f"Unable to obtain task information：batch_name='{args.batch_name}' May not exist or may have been completed")
            args.task_name = task_info["env_name"]
            task_name = args.task_name
            args.instance_id = 1
        except Exception as e:
            print(colored(f"Error: Failed to get task - {str(e)}", "red"))
            raise

        output_dir = Path('./datasets')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.recording_path = str(output_dir / f"{args.task_name}.hdf5")

        layout_config = get_random_layout_for_task(args.task_name, args.robot)
        if layout_config is None:
            raise ValueError(f"Unable to get layout configuration：robot='{args.robot}', task='{args.task_name}'")

        test_params = {
            'teleop_device': 'vr-controller',
            'robot': args.robot,
            'task': args.task_name,
            'record': True,
            'dataset_file': './datasets/dataset.hdf5',
            'save_video': False,
            'video_save_dir': './datasets',
            'layout': layout_config['layout'],
            'num_demos': 1,
            'enable_cameras': args.enable_camera,
            'object_init_offset': layout_config['object_init_offset'],
            'first_person_view': True,
            'enable_multiple_viewports': True,
        }

        if layout_config and 'init_robot_base_pos' in layout_config:
            test_params['init_robot_base_pos'] = layout_config['init_robot_base_pos']
        if layout_config and 'init_robot_base_ori' in layout_config:
            test_params['init_robot_base_ori'] = layout_config['init_robot_base_ori']

        print(colored("Test start teleoperation daemon...", "green"))
        print(f"params: {test_params}")

        daemon = TeleopDaemon(test_params, None, uploader, args.task_name, args.instance_id)
        daemon.start()

        print(f"start，PID: {daemon.process.pid}")

        try:
            while daemon.is_alive():
                time.sleep(3)

            return_code = daemon.process.returncode if daemon.process else -1
            print("\n" + "=" * 50)
            print(colored(f"teleop stop，return code: {return_code}", "red"))
            daemon._handle_exit(return_code)

        except KeyboardInterrupt:
            print(colored("\nreceived stop signal...", "red"))
            daemon.stop()
            print(colored("process stop", "green"))

    except Exception as e:
        print(colored(f"error in process: {e}", "red"))
        print(traceback.format_exc())

        if task_name:
            reset_task(uploader, "error before start")

        if daemon and daemon.is_alive():
            try:
                daemon.stop()
            except Exception as stop_e:
                print(colored(f"error in stop process: {stop_e}", "red"))

        sys.exit(1)


if __name__ == "__main__":
    main()
