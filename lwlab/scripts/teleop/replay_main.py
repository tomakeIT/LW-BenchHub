# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse
from pathlib import Path
import mediapy as media
import tqdm
from itertools import count
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from isaaclab.app import AppLauncher

from lwlab.utils.isaaclab_utils import get_robot_joint_target_from_scene

from lwlab.utils.log_utils import get_default_logger, get_logger

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--width", type=int, default=1920, help="Width of the rendered image.")
parser.add_argument("--height", type=int, default=1080, help="Height of the rendered image.")
parser.add_argument("--without_image", action="store_true", default=True, help="without image")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument("--robot_scale", type=float, default=1.0, help="robot scale")
parser.add_argument("--first_person_view", action="store_true", default=False, help="first person view")
parser.add_argument("--replay_mode", type=str, default="action", help="replay mode(action, joint_target, or state)")
parser.add_argument("--layout", type=str, default=None, help="layout name")
parser.add_argument("--replay_all_clips", action="store_true", help="replay all clips, otherwise only replay the last clips")
parser.add_argument("--record", action="store_true", default=False, help="record the replayed actions")
parser.add_argument("--demo", type=int, default=-1, help="demo num in hdf5.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

if args_cli.replay_mode not in ("action", "joint_target", "state", "raw_action", "raw_input"):
    raise ValueError(f"Invalid replay mode: {args_cli.replay_mode}, can only be 'action', 'joint_target', 'state', 'raw_action', or 'raw_input'")

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

if not args_cli.without_image:
    import cv2

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import torch

if not args_cli.headless:
    from lwlab.core.devices.keyboard.se3_keyboard import Se3Keyboard
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

obj_state_logger = get_logger("obj_state_logger")

if not args_cli.headless:
    from lwlab.utils.ui_utils import (
        hide_ui_windows
    )

    hide_ui_windows(simulation_app)

# Global GUI controller instance
gui_controller = None


def play_cb():
    if gui_controller:
        gui_controller.is_paused = False
        gui_controller._update_button_states()


def pause_cb():
    if gui_controller:
        gui_controller.is_paused = True
        gui_controller._update_button_states()


class VideoProcessor:
    """Independent video processing thread for ordered frame handling"""

    def __init__(self, replay_mp4_path, video_height, video_width, args_cli):
        self.replay_mp4_path = replay_mp4_path
        self.video_height = video_height
        self.video_width = video_width
        self.args_cli = args_cli
        self.frame_queue = queue.Queue(maxsize=100)
        self.running = True
        self.v = None
        self.thread = threading.Thread(target=self._process_frames_worker, daemon=True)
        self.thread.start()

    def add_frame(self, obs, camera_names):
        """Add a frame to the processing queue"""
        if not self.running:
            return
        self.frame_queue.put_nowait((obs, camera_names))

    def _process_frames_worker(self):
        """Worker thread that processes frames in order"""
        self.v = media.VideoWriter(path=self.replay_mp4_path, shape=(self.video_height, self.video_width), fps=30)
        self.v.__enter__()

        frame_count = 0
        try:
            while self.running:
                if not self.frame_queue.empty():
                    obs, camera_names = self.frame_queue.get_nowait()
                    self._process_single_frame(obs, camera_names)
                    frame_count += 1
                    self.frame_queue.task_done()
                else:
                    import time
                    time.sleep(0.01)

            # Process remaining frames after shutdown signal
            while not self.frame_queue.empty():
                obs, camera_names = self.frame_queue.get_nowait()
                self._process_single_frame(obs, camera_names)
                frame_count += 1
                self.frame_queue.task_done()

        except Exception as e:
            print(f"Video processing error: {e}")
        finally:
            if self.v:
                self.v.__exit__(None, None, None)

    def _process_single_frame(self, obs, camera_names):
        """Process a single frame"""
        camera_images = [obs['policy'][name].cpu().numpy() for name in camera_names]
        if not camera_images:
            return

        camera_images = [img.squeeze(0) for img in camera_images]
        num_cameras = len(camera_images)

        if num_cameras > 4:
            cameras_per_row = (num_cameras + 1) // 2
            first_row = camera_images[:cameras_per_row]
            first_row_final = np.concatenate(first_row, axis=1)
            second_row = camera_images[cameras_per_row:]
            if second_row:
                second_row_final = np.concatenate(second_row, axis=1)
                full_image = np.concatenate([first_row_final, second_row_final], axis=0)
            else:
                full_image = first_row_final
        else:
            full_image = np.concatenate(camera_images, axis=1)

        self.v.add_image(full_image)
        if not self.args_cli.without_image:
            cv2.imshow("replay", full_image[..., ::-1])
            cv2.waitKey(1)

    def shutdown(self):
        """Shutdown the video processor"""
        self.running = False
        import time
        start_time = time.time()
        while not self.frame_queue.empty() and time.time() - start_time < 10.0:
            time.sleep(0.1)

        try:
            self.frame_queue.join()
        except Exception:
            pass

        self.thread.join(timeout=3.0)

    def get_video_path(self):
        """Get the video file path"""
        return self.replay_mp4_path


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index].squeeze(0)
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name].squeeze(0)
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


class ReplayGUIController:
    """GUI controller for replay functionality with comprehensive controls."""

    def __init__(self, env=None, replay_mode="state"):
        """Initialize the replay GUI controller.

        Args:
            env: The Isaac Lab environment instance
            replay_mode: Replay mode ("state", "action", or "joint_target")
        """
        import omni.ui as ui

        self.env = env
        self.replay_mode = replay_mode
        self.current_frame = 0
        self.max_frames = 0
        self.frame_jump_target = -1
        self.step_mode = False
        self.step_backward = False  # Flag to indicate backward stepping
        self.is_paused = True  # Start paused
        self.episode_data = None
        self.reset_requested = False  # Flag to indicate reset requested

        # UI Components
        self.progress_window = None
        self.current_frame_label = None
        self.frame_input = None
        self.frame_progress = None
        self.frame_slider = None
        self.frame_range_label = None
        self.play_pause_btn = None
        self.next_btn = None

        # Shared data model for ProgressBar and IntSlider synchronization
        self.frame_model = None

        # Create the GUI
        self._create_gui()

    def _create_gui(self):
        """Create the GUI components."""
        import omni.ui as ui

        # Create shared data model for ProgressBar and IntSlider synchronization
        self.frame_model = ui.SimpleIntModel(0, min=0, max=0)

        # Adjust window size based on mode
        if self.replay_mode == "state":
            window_width = 800
            window_height = 150
        else:
            window_width = 400
            window_height = 100

        self.progress_window = ui.Window(
            f"Replay Control ({self.replay_mode})",
            width=window_width,
            height=window_height,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_RESIZE
        )

        with self.progress_window.frame:
            with ui.VStack(spacing=5):
                if self.replay_mode == "state":
                    # Full GUI for state mode
                    # Frame information row
                    with ui.HStack(spacing=10):
                        ui.Label("CurrentFrame:", width=80)
                        self.current_frame_label = ui.Label("0", width=80)
                        ui.Spacer(width=20)
                        ui.Label("Jump to:", width=60)
                        self.frame_input = ui.IntField(width=80, on_value_changed_fn=self._on_frame_jump)
                        ui.Button("Jump", width=60, clicked_fn=self._on_jump_button_click)

                    # Control buttons row
                    with ui.HStack(spacing=5):
                        ui.Button("Reset", width=60, clicked_fn=self._on_reset, tooltip="Reset to Begin")
                        ui.Button("Prev", width=60, clicked_fn=self._on_frame_prev, tooltip="Previous Frame")
                        self.play_pause_btn = ui.Button("Play", width=60, clicked_fn=self._on_play_pause, tooltip="Toggle replay state")
                        self.next_btn = ui.Button("Next", width=60, clicked_fn=self._on_frame_next, tooltip="Next Frame (only when paused)")
                        self.frame_range_label = ui.Label("/ 0", width=60)

                    # Progress bar and slider row (synchronized)
                    with ui.HStack(spacing=10):
                        ui.Label("Progress:", width=60)
                        # ProgressBar using shared model
                        self.frame_progress = ui.ProgressBar(
                            model=self.frame_model,
                            width=200,
                            height=20,
                            tooltip="Current frame progress"
                        )
                        # IntSlider using the same shared model for synchronization
                        self.frame_slider = ui.IntSlider(
                            model=self.frame_model,
                            width=200,
                            min=0,
                            max=0,
                            tooltip="Drag to set target frame",
                            on_value_changed_fn=self._on_slider_changed
                        )
                else:
                    # Simplified GUI for action/joint_target mode
                    # Frame information row
                    with ui.HStack(spacing=10):
                        ui.Label("Frame:", width=60)
                        self.current_frame_label = ui.Label("0", width=80)
                        self.frame_range_label = ui.Label("/ 0", width=60)

                    # Control buttons row (only basic controls)
                    with ui.HStack(spacing=5):
                        ui.Button("Reset", width=60, clicked_fn=self._on_reset, tooltip="Reset to Begin")
                        self.play_pause_btn = ui.Button("Play", width=60, clicked_fn=self._on_play_pause, tooltip="Toggle replay state")
                        self.next_btn = ui.Button("Next", width=60, clicked_fn=self._on_frame_next, tooltip="Next Frame (only when paused)")

                    # Simple progress bar
                    with ui.HStack(spacing=10):
                        ui.Label("Progress:", width=60)
                        self.frame_progress = ui.ProgressBar(
                            model=self.frame_model,
                            width=200,
                            height=20,
                            tooltip="Current frame progress"
                        )

                # Additional controls row (only for state mode)
                if self.replay_mode == "state":
                    with ui.HStack(spacing=5):
                        ui.Button("Save Frame", width=80, clicked_fn=self._on_save_frame, tooltip="Save current frame as image")
                        ui.Button("Load Episode", width=80, clicked_fn=self._on_load_episode, tooltip="Load episode data")
                        ui.Button("Export Video", width=80, clicked_fn=self._on_export_video, tooltip="Export current episode as video")

        # Initialize button states
        self._update_button_states()

    def _update_button_states(self):
        """Update button enabled states based on current mode."""
        if self.next_btn:
            self.next_btn.enabled = self.is_paused
        if self.play_pause_btn:
            self.play_pause_btn.text = "Pause" if not self.is_paused else "Play"

    def set_episode_data(self, episode_data):
        """Set the episode data for replay.

        Args:
            episode_data: EpisodeData object containing the recorded data
        """
        print(f"DEBUG: set_episode_data called with episode_data={episode_data}")
        self.episode_data = episode_data
        self._update_max_frames()

    def _update_max_frames(self):
        """Set the maximum number of frames based on episode data."""
        print(f"DEBUG: _update_max_frames - replay_mode={self.replay_mode}")
        if self.episode_data:
            # Use actions length for all replay modes since state and action counts are always equal
            if "actions" in self.episode_data._data:
                self.max_frames = len(self.episode_data._data["actions"])
                print(f"Set maximum number of frames (all modes): {self.max_frames}")
            else:
                self.max_frames = 0
                print("Cannot get actions data, keep maximum number of frames as 0")
                return

            # Update shared model with new max value
            if self.frame_model:
                self.frame_model.set_max(self.max_frames - 1)

            # Update slider max value
            if self.frame_slider:
                self.frame_slider.model.set_max(self.max_frames - 1)

            # Update range label
            if self.frame_range_label:
                self.frame_range_label.text = f"/ {self.max_frames - 1}"
        else:
            self.max_frames = 0
            print("Cannot get episode data, keep maximum number of frames as 0")

    def _update_frame_display(self):
        """Update the frame number display."""
        if self.current_frame_label:
            self.current_frame_label.text = str(self.current_frame)

        # Update shared model - this will automatically update both ProgressBar and IntSlider
        if self.frame_model:
            self.frame_model.set_value(self.current_frame)

        if self.frame_input:
            self.frame_input.model.set_value(self.current_frame)
        if self.env and hasattr(self.env, 'sim'):
            self.env.sim.render()

    def get_action_by_index(self, action_index):
        """Get action by index from current episode data.

        Args:
            action_index: Index of the action to retrieve

        Returns:
            Action data or None if not found
        """
        if not self.episode_data or "actions" not in self.episode_data._data:
            return None

        actions_data = self.episode_data._data["actions"]
        if action_index >= len(actions_data):
            return None

        return actions_data[action_index]

    def get_joint_target_by_index(self, joint_target_index):
        """Get joint target by index from current episode data.

        Args:
            joint_target_index: Index of the joint target to retrieve

        Returns:
            Joint target data or None if not found
        """
        if not self.episode_data or "joint_targets" not in self.episode_data._data:
            return None

        joint_targets_data = self.episode_data._data["joint_targets"]["joint_pos_target"]
        if joint_target_index >= len(joint_targets_data):
            return None

        return joint_targets_data[joint_target_index]

    def _on_frame_prev(self):
        """Handle previous frame button click (step mode, state replay only)."""
        if not self.is_paused:
            print("Step mode is only available when paused. Please pause first.")
            return

        # Only work in state replay mode
        if self.replay_mode != "state":
            print("Previous frame stepping is only available in state replay mode.")
            return

        if self.current_frame > 0:
            self.step_mode = True
            self.step_backward = True  # Set backward flag
            self.is_paused = False
            self._update_button_states()
            print(f"Execute previous frame in step mode, then pause")

    def _on_frame_next(self):
        """Handle next frame button click (step mode)."""
        if not self.is_paused:
            print("Step mode is only available when paused. Please pause first.")
            return

        print(f"DEBUG: _on_frame_next - max_frames={self.max_frames}, current_frame={self.current_frame}")
        if self.max_frames > 0 and self.current_frame < self.max_frames - 1:
            self.step_mode = True
            self.step_backward = False  # Clear backward flag for forward stepping
            self.is_paused = False
            self._update_button_states()
            print(f"Execute next frame in step mode, then pause")

    def _on_frame_jump(self, value):
        """Handle frame jump input change.

        Args:
            value: Target frame value
        """
        self.frame_jump_target = int(value)
        print(f"Set jump target frame: {self.frame_jump_target}")

    def _on_slider_changed(self, value):
        """Handle slider value change.

        Args:
            value: New frame value from slider
        """
        if 0 <= value < self.max_frames:
            self.current_frame = int(value)
            print(f"Slider changed to frame: {self.current_frame}")
            # Update other displays
            if self.current_frame_label:
                self.current_frame_label.text = str(self.current_frame)
            if self.frame_input:
                self.frame_input.model.set_value(self.current_frame)

    def _on_jump_button_click(self):
        """Handle jump button click."""
        if self.max_frames > 0 and 0 <= self.frame_jump_target < self.max_frames:
            self.current_frame = self.frame_jump_target
            print(f"Execute jump to frame: {self.frame_jump_target}")
            self._update_frame_display()

    def _on_play_pause(self):
        """Handle play/pause button click."""
        self.is_paused = not self.is_paused
        self.step_mode = False
        self.step_backward = False  # Clear backward flag
        self._update_button_states()
        print(f"Play state: {'pause' if self.is_paused else 'play'}")

    def _on_reset(self):
        """Handle reset button click."""
        # self.current_frame = 0
        # self.is_paused = True
        # self.step_mode = False
        # self._update_button_states()
        # print("Reset replay")
        # self._update_frame_display()

        # Simplified reset - only reset GUI state, avoid touching episode data
        print("Simplified reset - only GUI state reset")

        # Use global flag for thread synchronization
        self.reset_requested = True
        print(f"Scene reset requested - using global flag, reset_requested={self.reset_requested}")
        print(f"DEBUG: Reset method - reset_requested={self.reset_requested}, id(self)={id(self)}")

        # Note: Removed env.sim.render() call to avoid interfering with replay loop
        print("Reset complete - restart replay for fresh scene state")

    def _on_save_frame(self):
        """Handle save frame button click."""
        if self.env and hasattr(self.env, 'sim'):
            # Save current frame as image
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{self.current_frame:06d}_{timestamp}.png"

            # Create screenshots directory if it doesn't exist
            screenshots_dir = "screenshots"
            os.makedirs(screenshots_dir, exist_ok=True)

            filepath = os.path.join(screenshots_dir, filename)
            self.env.sim.render()
            # Note: Actual image saving would require additional implementation
            print(f"Frame {self.current_frame} saved to {filepath}")

    def _on_load_episode(self):
        """Handle load episode button click."""
        print("Load episode functionality - implement file dialog here")
        # This would typically open a file dialog to select an episode file

    def _on_export_video(self):
        """Handle export video button click."""
        if self.episode_data and self.max_frames > 0:
            print(f"Export video functionality - would export {self.max_frames} frames")
            # This would typically start a video export process
        else:
            print("No episode data available for export")

    def set_current_frame(self, frame):
        """Set the current frame programmatically.

        Args:
            frame: Frame number to set
        """
        if 0 <= frame < self.max_frames:
            self.current_frame = frame
            self._update_frame_display()
        else:
            print(f"DEBUG: set_current_frame failed - frame={frame} not in range [0, {self.max_frames})")

    def get_current_frame(self):
        """Get the current frame number.

        Returns:
            Current frame number
        """
        return self.current_frame

    def is_playing(self):
        """Check if currently playing.

        Returns:
            True if playing, False if paused
        """
        return not self.is_paused

    def is_step_mode(self):
        """Check if in step mode.

        Returns:
            True if in step mode, False otherwise
        """
        return self.step_mode

    def should_advance_frame(self):
        """Check if should advance to next frame.

        Returns:
            True if should advance, False otherwise
        """
        return not self.is_paused and (self.current_frame < self.max_frames - 1)

    def advance_frame(self):
        """Advance to next frame if conditions are met."""
        if self.should_advance_frame():
            self.current_frame += 1
            self._update_frame_display()

            # If in step mode, pause after advancing
            if self.step_mode:
                self.is_paused = True
                self.step_mode = False
                self._update_button_states()
                print("Step mode: paused after advancing one frame")

            return True
        return False

    def close(self):
        """Close the GUI window."""
        if self.progress_window:
            self.progress_window.visible = False
            self.progress_window = None


def main():
    """Replay episodes loaded from a file."""
    from isaaclab.envs import ManagerBasedRLEnv

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        if not args_cli.replay_all_clips:
            episode_indices_to_replay = [episode_count - 1]
        else:
            episode_indices_to_replay = list(range(episode_count))

    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    replayed_episode_count = 0

    episode_names_to_replay = []
    for idx in episode_indices_to_replay:
        episode_names_to_replay.append(episode_names[idx])

    num_envs = args_cli.num_envs

    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])
    if "LW_API_ENDPOINT" in env_args.keys():
        os.environ["LW_API_ENDPOINT"] = env_args["LW_API_ENDPOINT"]
    usd_simplify = env_args["usd_simplify"] if 'usd_simplify' in env_args else False
    if "-" in env_args["env_name"] and not env_args["env_name"].startswith("Robocasa"):
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        task_name = args_cli.task
    else:  # robocasa
        from lwlab.utils.env import ExecuteMode
        task_name = env_args["task_name"] if args_cli.task is None else args_cli.task
        if task_name == "PutButterInBasket":
            if "PutButterInBasket2" in env_args["env_name"]:
                task_name = "PutButterInBasket2"
        robot_name = env_args["robot_name"]
        if robot_name == "double_piper_abs":
            robot_name = "DoublePiper-Abs"
        if robot_name == "double_piper_rel":
            robot_name = "DoublePiper-Rel"
        if "robocasalibero" in env_args["usd_path"]:
            scene_name = "robocasalibero"
        else:
            scene_name = "robocasakitchen"
        env_cfg = parse_env_cfg(
            task_name=task_name,
            robot_name=robot_name,
            scene_name=f"{scene_name}-{env_args['layout_id']}-{env_args['style_id']}" if args_cli.layout is None else args_cli.layout,
            robot_scale=args_cli.robot_scale,
            device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True,
            replay_cfgs={"hdf5_path": args_cli.dataset_file, "ep_meta": env_args, "render_resolution": (args_cli.width, args_cli.height), "ep_names": episode_names_to_replay},
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.REPLAY_ACTION if args_cli.replay_mode == "action" else ExecuteMode.REPLAY_JOINT_TARGETS if args_cli.replay_mode == "joint_target" else ExecuteMode.REPLAY_STATE,
            usd_simplify=usd_simplify,
        )
        env_name = f"Robocasa-{task_name}-{robot_name}-v0"
        gym.register(
            id=env_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                # "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
            },
            disable_env_checker=True,
        )

    # Disable all recorders and terminations
    if args_cli.record:
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.dataset_file)
        env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0] + f"_{args_cli.replay_mode}_replay_record.hdf5"

    delattr(env_cfg.terminations, "time_out")

    # create environment from loaded config
    env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    from multiprocessing import shared_memory
    from lwlab.core.devices import VRController
    image_size = (720, 1280)
    shm = shared_memory.SharedMemory(
        create=True,
        size=image_size[0] * image_size[1] * 3 * np.dtype(np.uint8).itemsize,
    )
    teleop_interface = VRController(env,
                                    img_shape=image_size,
                                    shm_name=shm.name,)

    # Initialize GUI controller
    global gui_controller
    if not args_cli.headless:
        try:
            gui_controller = ReplayGUIController(env, args_cli.replay_mode)
            print("GUI controller initialized successfully")
        except Exception as e:
            print(f"Failed to initialize GUI controller: {e}")
            gui_controller = None

    if app_launcher._enable_cameras and not args_cli.headless:
        # teleop_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
        # teleop_interface.add_callback("N", play_cb)
        # teleop_interface.add_callback("B", pause_cb)
        print('Press "B" to pause and "N" to resume the replayed actions.')
    print(args_cli.dataset_file)

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    elif args_cli.replay_mode in ("action", "joint_target", "raw_action", "raw_input"):
        idle_action = torch.zeros(env.action_space.shape)
    elif args_cli.replay_mode == "state":
        idle_action = None  # State mode doesn't use actions
    else:
        raise ValueError(f"Invalid replay mode: {args_cli.replay_mode}, can only be 'action', 'joint_target', or 'state'")

    import carb
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
    # reset before starting
    env.reset()
    if app_launcher._enable_cameras and not args_cli.headless:
        teleop_interface.reset()
    font = ImageFont.load_default()
    if hasattr(font, 'font_variant'):
        font = font.font_variant(size=20)

    # prepare video writer and json file
    success_info = {}
    has_success = False

    # read ee_poses from hdf5
    import h5py
    from lwlab.utils.teleop_utils import get_action_by_frame, get_joint_target_by_frame, get_raw_action_by_frame, get_raw_input_by_frame, replay_from_raw_input

    # simulate environment -- run everything in inference mode
    num_cameras = sum(env.cfg.task_type in c["tags"] for c in env.cfg.observation_cameras.values())
    if num_cameras > 4:
        # two rows layout: height is twice the original, width is the maximum width of each row
        cameras_per_row = (num_cameras + 1) // 2
        video_height = args_cli.height * 2
        video_width = max(cameras_per_row, num_cameras - cameras_per_row) * args_cli.width
    else:
        # single row layout: original calculation
        video_height = args_cli.height
        video_width = num_cameras * args_cli.width

    # Initialize async image processor
    video_processor = None

    for i in episode_indices_to_replay:
        episode_indices_to_replay_tmp = [i]
        if not args_cli.replay_all_clips:
            save_dir = Path(args_cli.dataset_file).parent
        else:
            save_dir = Path(args_cli.dataset_file).parent / 'replay_results' / episode_names[i]
        save_dir.mkdir(parents=True, exist_ok=True)
        replay_mp4_path = save_dir / f"isaac_replay_action_{args_cli.replay_mode}.mp4"
        replay_json_path = replay_mp4_path.with_suffix('.json')
        # For state mode, we don't need to read ee_poses file as we generate it
        # For action/joint_target modes, we'll read it inside the loop when needed
        gt_ee_poses_f = None

        # Initialize video processor (manages VideoWriter internally)
        if app_launcher._enable_cameras:
            video_processor = VideoProcessor(replay_mp4_path, video_height, video_width, args_cli)
        else:
            video_processor = None

        with (
            contextlib.suppress(KeyboardInterrupt),
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_pose_divergence.hdf5", "w") as pose_divergence_f,
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_obj_pose_divergence.hdf5", "w") as obj_divergence_f,
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_obj_force.hdf5", "w") as obj_force_f
        ):
            pose_divergence_f.create_group("data")
            obj_divergence_f.create_group("data")
            obj_force_f.create_group("data")
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}

            # Common episode loading logic for both state and action replay modes
            def load_next_episode(env_id):
                """Load next episode for the given environment ID"""
                next_episode_index = None
                while episode_indices_to_replay_tmp:
                    next_episode_index = episode_indices_to_replay_tmp.pop(0)
                    if next_episode_index < episode_count:
                        break
                    next_episode_index = None

                if next_episode_index is not None:
                    nonlocal replayed_episode_count
                    replayed_episode_count += 1
                    episode_name = episode_names[next_episode_index]
                    print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode {episode_name} to env_{env_id}")
                    episode_data = dataset_file_handler.load_episode(
                        episode_names[next_episode_index], env.device
                    )
                    env_episode_data_map[env_id] = episode_data
                    return episode_data, episode_name
                return None, None

            if args_cli.replay_mode == "state":
                # State mode: replay by resetting to each state directly
                for env_id in range(num_envs):
                    episode_data, episode_name = load_next_episode(env_id)
                    if episode_data is None:
                        break

                    if "states" not in episode_data._data:
                        break

                    # We'll manage the count ourselves, no need to get total states upfront
                    # The loop will continue until we can't get a state by index

                    # Set initial state for the new episode
                    initial_state = episode_data.get_initial_state()
                    env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=False)

                    ee_poses = []
                    step_count = 0

                    # Set episode data for GUI controller
                    if gui_controller:
                        gui_controller.set_episode_data(episode_data)

                    # Create progress bar for state replay
                    pbar = tqdm.tqdm(desc=f"Replaying states {episode_name}")

                    # Use index-based loop with self-managed counting
                    while True:
                        # Check for step mode (both forward and backward)
                        if gui_controller and gui_controller.is_step_mode():
                            # Check if this is backward stepping
                            if gui_controller.step_backward:
                                # Backward stepping: reset to previous state using index
                                target_frame = step_count - 1
                                print(f"Backward step: resetting to frame {target_frame}")

                                # Get state by index using the utility function
                                try:
                                    from lwlab.utils.teleop_utils import get_state_by_frame
                                    target_state = get_state_by_frame(episode_data, target_frame)

                                    if target_state is not None:
                                        env.reset_to(target_state, torch.tensor([env_id], device=env.device), is_relative=True if env_args["robot_name"].endswith("Rel") else False)

                                        # Update current state
                                        step_count = target_frame

                                        # Update GUI
                                        if gui_controller:
                                            gui_controller.set_current_frame(step_count)

                                        # Handle step mode - pause after stepping backward
                                        gui_controller.is_paused = True
                                        gui_controller.step_mode = False
                                        gui_controller.step_backward = False
                                        gui_controller._update_button_states()
                                        print("Step mode: paused after stepping one frame backward")
                                        continue
                                    else:
                                        print(f"Warning: Could not get state for frame {target_frame}")
                                        gui_controller.step_mode = False
                                        gui_controller.step_backward = False
                                        gui_controller._update_button_states()
                                        continue
                                except Exception as e:
                                    print(f"Error during backward stepping: {e}")
                                    gui_controller.step_mode = False
                                    gui_controller.step_backward = False
                                    gui_controller._update_button_states()
                                    continue
                            else:
                                # Forward stepping: continue with normal flow
                                pass

                        # Check for scene reset request
                        if gui_controller and gui_controller.reset_requested:
                            try:
                                initial_state = episode_data.get_initial_state()
                                if initial_state is not None:
                                    env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=False)
                                    step_count = 0
                                    # Update GUI display after reset
                                    gui_controller.set_current_frame(step_count)
                                    gui_controller.reset_requested = False
                                    print("Scene reset completed in replay loop")
                                else:
                                    print("Warning: No initial state available for reset")
                                    gui_controller.reset_requested = False
                            except Exception as e:
                                print(f"Error during scene reset: {e}")
                                gui_controller.reset_requested = False
                            continue

                        # Handle pause functionality
                        if gui_controller and gui_controller.is_paused:
                            env.sim.render()
                            continue

                        # Handle frame jumping
                        if gui_controller and gui_controller.frame_jump_target != -1 and gui_controller.current_frame != gui_controller.frame_jump_target:
                            print(f"Need to jump to frame: {gui_controller.frame_jump_target}")
                            # Implement frame jumping logic here

                        if args_cli.first_person_view:
                            try:
                                from lwlab.utils.env import set_camera_follow_pose
                                set_camera_follow_pose(env, env_cfg.viewport_cfg["offset"], env_cfg.viewport_cfg["lookat"])
                            except ImportError:
                                print("set_camera_follow_pose function not available")

                        # Get state by index instead of using get_next_state
                        from lwlab.utils.teleop_utils import get_state_by_frame
                        current_state = get_state_by_frame(episode_data, step_count)

                        if current_state is not None:
                            obs, _ = env.reset_to(current_state, torch.tensor([env_id], device=env.device), is_relative=True if env_args["robot_name"].endswith("Rel") else False)
                            env.sim.render()
                            ee_poses.append(obs['policy']['ee_pose'].cpu().numpy())

                            # Update progress bar only when actually processing a state
                            pbar.update(1)

                            if app_launcher._enable_cameras and video_processor:
                                camera_names = [n for n, c in env.cfg.observation_cameras.items() if env.cfg.task_type in c["tags"]]
                                video_processor.add_frame(obs, camera_names)

                            # Handle step mode - pause after advancing one frame
                            if gui_controller and gui_controller.is_step_mode():
                                # Check if this was a forward step (not backward)
                                was_forward_step = not gui_controller.step_backward

                                gui_controller.is_paused = True
                                gui_controller.step_mode = False
                                gui_controller.step_backward = False  # Clear backward flag
                                gui_controller._update_button_states()
                                print("Step mode: paused after advancing one frame")

                                # For forward step mode, increment step_count
                                if was_forward_step:
                                    step_count += 1
                            else:
                                # Only increment step_count in normal playback mode
                                step_count += 1

                            # Update GUI display after step_count is updated
                            if gui_controller:
                                gui_controller.set_current_frame(step_count)
                            else:
                                # Fallback for when GUI is not available
                                print(f"Frame: {step_count}")
                        else:
                            print(f"Reached end of states at frame {step_count}")
                            break

                    # Close progress bar
                    pbar.close()

                    # Save ee poses for state mode
                    if ee_poses:
                        ee_poses = np.concatenate(ee_poses, axis=0)
                        # Save ee poses to hdf5 file (always create new file for state mode)
                        ee_poses_path = save_dir / "replay_state_ee_poses.hdf5"
                        with h5py.File(ee_poses_path, "w") as ee_poses_f:
                            ee_poses_f.create_group("data")
                            ee_poses_f.create_dataset(f"data/{episode_name}/ee_poses", data=ee_poses)

                    success_info[episode_name] = {"success": True}
            else:
                # Use index-based access instead of get_next methods
                # Action/joint_target mode: original logic
                for env_id in range(num_envs):
                    episode_data, episode_name = load_next_episode(env_id)
                    if episode_data is None:
                        break

                    # Set episode data for GUI controller
                    if gui_controller:
                        gui_controller.set_episode_data(episode_data)

                    # Set initial state for the new episode
                    initial_state = episode_data.get_initial_state()
                    env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=False)

                    step_count = 0  # Initialize step counter for action replay

                    ee_poses = []
                    joint_pos_list = []
                    joint_target_list = []
                    gt_joint_target_list = []
                    obj_states = {}
                    obj_force_states = {}

                    gt_joint_pos = episode_data._data["states"]["articulation"]["robot"]["joint_position"]

                    # Create progress bar for action replay
                    pbar = tqdm.tqdm(desc=f"Replaying actions {args_cli.replay_mode}")

                    while step_count < gui_controller.max_frames:
                        # Check for scene reset request using global flag
                        if gui_controller and gui_controller.reset_requested:
                            try:
                                # Find the current episode data for reset
                                current_episode_data = None

                                if env_id in env_episode_data_map:
                                    current_episode_data = env_episode_data_map[env_id]

                                if current_episode_data:
                                    initial_state = current_episode_data.get_initial_state()
                                    if initial_state is not None:
                                        env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=False)
                                        step_count = 0
                                        gui_controller.set_current_frame(step_count)
                                        gui_controller.reset_requested = False  # Clear global flag
                                        print("Scene reset completed in action replay loop")
                                    else:
                                        print("Warning: No initial state available for reset")
                                        gui_controller.reset_requested = False
                                else:
                                    print("Warning: No episode data available for reset")
                                    gui_controller.reset_requested = False
                            except Exception as e:
                                print(f"Error during scene reset: {e}")
                                gui_controller.reset_requested = False
                            continue

                        # Handle pause functionality before getting actions
                        if gui_controller and gui_controller.is_paused:
                            env.sim.render()
                            continue

                        # initialize actions with idle action so those without next action will not move
                        actions = idle_action
                        if args_cli.replay_mode == "action":
                            actions[env_id] = get_action_by_frame(env_episode_data_map[env_id], step_count)
                        elif args_cli.replay_mode == "joint_target":
                            actions[env_id] = get_joint_target_by_frame(env_episode_data_map[env_id], step_count)["joint_pos_target"]
                        elif args_cli.replay_mode == "raw_action":
                            raw_action = get_raw_action_by_frame(env_episode_data_map[env_id], step_count)
                            actions[env_id] = env.cfg.preprocess_device_action(raw_action, teleop_interface)
                        elif args_cli.replay_mode == "raw_input":
                            raw_input = get_raw_input_by_frame(env_episode_data_map[env_id], step_count)
                            raw_action = replay_from_raw_input(raw_input, teleop_interface)
                            if raw_action is None:
                                print(f"Attention action is None")
                                continue
                            if raw_action['reset'] is True:
                                print(f"Attention reset is True")
                                continue
                            actions[env_id] = env.cfg.preprocess_device_action(raw_action, teleop_interface)

                        if actions[env_id] is None:
                            print(f"Warning: No action found for step {step_count}")
                            break

                        obs, _, ter, _, _ = env.step(actions)

                        print(f"DEBUG: step_count: {step_count}")

                        # Increment step count after successful step
                        step_count += 1
                        pbar.update(1)

                        # Update GUI display after incrementing step count
                        if gui_controller:
                            gui_controller.set_current_frame(step_count)
                        else:
                            # Fallback for when GUI is not available
                            print(f"Frame: {step_count}")

                        # Handle step mode - pause after advancing one frame
                        if gui_controller and gui_controller.is_step_mode():
                            gui_controller.is_paused = True
                        gui_controller.step_mode = False
                        gui_controller._update_button_states()

                        # Collect rigid object states - store in lists for batch processing
                        env_objs = env.scene.rigid_objects
                        all_keys = list(env_objs.keys())

                        current_obj_state = env.scene.get_state(is_relative=True)["rigid_object"]
                        # runtime_state[asset_type][asset_name][state_name][runtime_env_index].squeeze(0)

                        for obj_name in all_keys:
                            # Initialize object state tracking lists if not exists
                            if obj_name not in obj_states:
                                obj_states[obj_name] = {
                                    'obj_root_pose': [],
                                    'obj_root_vel': [],
                                    'ref_root_pose': [],
                                    'ref_root_vel': []
                                }
                            if obj_name not in obj_force_states:
                                obj_force_states[obj_name] = {
                                    'obj_force': [],
                                }

                            obj_root_pose = current_obj_state[obj_name]["root_pose"].cpu().numpy()
                            obj_root_vel = current_obj_state[obj_name]["root_velocity"].cpu().numpy()
                            # Append current object states to lists (will be written to hdf5 at episode end)
                            obj_states[obj_name]['obj_root_pose'].append(obj_root_pose)
                            obj_states[obj_name]['obj_root_vel'].append(obj_root_vel)
                            if f'{obj_name}_contact' in env.scene.sensors:
                                # Get force data for specific environment and create a copy to avoid reference issues
                                temp_force = env.scene.sensors[f'{obj_name}_contact']._data.net_forces_w[:, 0, :].cpu().numpy().copy()
                                obj_force_states[obj_name]['obj_force'].append(temp_force)
                                get_default_logger().info(f"Obj {obj_name} force shape: {temp_force.shape}, force: {temp_force}")
                            else:
                                get_default_logger().warning(f"Sensor {f'{obj_name}_contact'} not found")

                            # Get corresponding reference states from dataset for current step
                            if obj_name in env_episode_data_map[env_id]._data['states']['rigid_object']:
                                current_state_idx = env_episode_data_map[env_id].next_state_index
                                total_states = len(env_episode_data_map[env_id]._data['states']['rigid_object'][obj_name]['root_pose'])

                                if current_state_idx < total_states:
                                    # Get the specific state for current step, ensuring correct indexing
                                    ref_root_pose = env_episode_data_map[env_id]._data['states']['rigid_object'][obj_name]['root_pose'][current_state_idx].unsqueeze(0).cpu().numpy()
                                    ref_root_vel = env_episode_data_map[env_id]._data['states']['rigid_object'][obj_name]['root_velocity'][current_state_idx].unsqueeze(0).cpu().numpy()

                                    # Append reference states to lists (will be written to hdf5 at episode end)
                                    obj_states[obj_name]['ref_root_pose'].append(ref_root_pose)
                                    obj_states[obj_name]['ref_root_vel'].append(ref_root_vel)
                                else:
                                    get_default_logger().warning(f"State index {current_state_idx} out of range for {obj_name} (total: {total_states})")

                        ee_poses.append(obs['policy']['ee_pose'].cpu().numpy())
                        joint_pos_list.append(obs["policy"]["joint_pos"].cpu().numpy())
                        if args_cli.replay_mode == "joint_target":
                            joint_target_list.append(get_robot_joint_target_from_scene(env.scene)["joint_pos_target"].cpu().numpy())
                            gt_joint_target_list.append(actions.reshape(env.cfg.decimation, -1)[-1:, ...].cpu().numpy())
                        if app_launcher._enable_cameras and video_processor:
                            camera_names = [n for n, c in env.cfg.observation_cameras.items() if env.cfg.task_type in c["tags"]]
                            # Process images asynchronously
                            video_processor.add_frame(obs, camera_names)

                        state_from_dataset = env_episode_data_map[env_id].get_next_state()
                        if state_validation_enabled:
                            if state_from_dataset is not None:
                                print(
                                    f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1 :4}",
                                    end="",
                                )
                                current_runtime_state = env.scene.get_state(is_relative=True)
                                states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                                if states_matched:
                                    print("\t- matched.")
                                else:
                                    print("\t- mismatched.")
                                    print(comparison_log)
                        if ter:
                            has_success = True
                            env_episode_data_map[env_id]._next_action_index += 9999999999

                    # Close progress bar
                    pbar.close()

                    # Process episode data after episode completion
                    if replayed_episode_count and isinstance(joint_pos_list, list):
                        # compare ee_poses with gt_ee_poses, calculate pose divergence
                        # Try to read ee_poses file for comparison (generated by state mode)
                        ee_poses_path = save_dir / "replay_state_ee_poses.hdf5"
                        if ee_poses_path.exists():
                            with h5py.File(ee_poses_path, "r") as gt_ee_poses_f:
                                gt_ee_poses = gt_ee_poses_f["data"][episode_name]["ee_poses"][:]
                                if ee_poses and len(ee_poses) > 0:
                                    ee_poses = np.concatenate(ee_poses, axis=0)
                                    pose_divergence = np.linalg.norm(ee_poses[:-1, :, :3] - gt_ee_poses[:len(ee_poses) - 1, :, :3], axis=-1)
                                    pose_divergence_norm = np.linalg.norm(pose_divergence, axis=-1)
                                    print(f"Pose divergence: last step: {pose_divergence[-1].tolist()}, mean: {np.mean(pose_divergence,axis=0).tolist()} max: {np.max(pose_divergence,axis=0).tolist()}")
                                    success_info[episode_name] = {
                                        "pose_divergence_last_step": pose_divergence[-1].tolist(),
                                        "pose_divergence_mean": np.mean(pose_divergence, axis=0).tolist(),
                                        "pose_divergence_max": np.max(pose_divergence, axis=0).tolist(),
                                        "pose_divergence_norm_last_step": pose_divergence_norm[-1].tolist(),
                                        "pose_divergence_norm_mean": np.mean(pose_divergence_norm, axis=0).tolist(),
                                        "pose_divergence_norm_max": np.max(pose_divergence_norm, axis=0).tolist(),
                                        "success": has_success
                                    }
                                else:
                                    print(f"Warning: No ee_poses data collected for episode {episode_name}, skipping pose divergence calculation")
                                    success_info[episode_name] = {
                                        "success": has_success
                                    }
                                # save pose divergence to hdf5 (only if we have data)
                                if ee_poses is not None and len(ee_poses) > 0:
                                    pose_divergence_f.create_dataset(f"data/{episode_name}/ee_poses", data=ee_poses[:])
                                    pose_divergence_f.create_dataset(f"data/{episode_name}/pose_divergence", data=pose_divergence)
                                    pose_divergence_f.create_dataset(f"data/{episode_name}/pose_divergence_norm", data=pose_divergence_norm)
                        else:
                            print(f"Warning: ee_poses file not found at {ee_poses_path}, skipping pose divergence calculation")
                            success_info[episode_name] = {
                                "success": has_success
                            }

                        # compare joint_pos_list with gt_joint_pos, calculate joint divergence
                        if joint_pos_list and len(joint_pos_list) > 0:
                            joint_pos_list = np.concatenate(joint_pos_list, axis=0)
                            joint_divergence = joint_pos_list[:-1] - gt_joint_pos.cpu().numpy()[:len(joint_pos_list) - 1]
                            # print(f"Joint divergence: last step: {joint_divergence[-1]}, mean: {joint_divergence.mean()} max: {joint_divergence.max()}")
                            # save joint divergence to hdf5
                            pose_divergence_f.create_dataset(f"data/{episode_name}/joint_pos", data=joint_pos_list[:])
                            pose_divergence_f.create_dataset(f"data/{episode_name}/joint_pos_divergence", data=joint_divergence[:])
                            if args_cli.replay_mode == "joint_target" and joint_target_list and len(joint_target_list) > 0:
                                joint_target_list = np.concatenate(joint_target_list, axis=0)
                                gt_joint_target_list = np.concatenate(gt_joint_target_list, axis=0)
                                joint_target_divergence = joint_target_list - gt_joint_target_list
                                pose_divergence_f.create_dataset(f"data/{episode_name}/joint_target", data=joint_target_list[:])
                                pose_divergence_f.create_dataset(f"data/{episode_name}/joint_target_divergence", data=joint_target_divergence[:])
                        else:
                            print(f"Warning: No joint_pos data collected for episode {episode_name}, skipping joint divergence calculation")

                        if "obj_force_states" in locals() and obj_force_states:
                            for obj_name, obj_force_data in obj_force_states.items():
                                if len(obj_force_data['obj_force']) > 0:
                                    obj_force_array = np.concatenate(obj_force_data['obj_force'], axis=0)
                                    obj_force_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_force", data=obj_force_array[:])
                                else:
                                    get_default_logger().warning(f"Object {obj_name} has no force data")
                                    continue

                        if 'obj_states' in locals() and obj_states:
                            get_default_logger().info(f"Processing {len(obj_states)} objects for episode {episode_name}")
                            for obj_name, obj_data in obj_states.items():
                                if obj_data['obj_root_pose'] and obj_data['ref_root_pose']:
                                    # Convert accumulated lists to numpy arrays for batch processing
                                    obj_root_pose_array = np.concatenate(obj_data['obj_root_pose'], axis=0)
                                    obj_root_vel_array = np.concatenate(obj_data['obj_root_vel'], axis=0)
                                    ref_root_pose_array = np.concatenate(obj_data['ref_root_pose'], axis=0)
                                    ref_root_vel_array = np.concatenate(obj_data['ref_root_vel'], axis=0)

                                    # Ensure arrays have the same length before calculating divergences
                                    min_length = min(len(obj_root_pose_array), len(ref_root_pose_array))

                                    if min_length > 0:
                                        obj_root_pose_array = obj_root_pose_array[:min_length]
                                        obj_root_vel_array = obj_root_vel_array[:min_length]
                                        ref_root_pose_array = ref_root_pose_array[:min_length]
                                        ref_root_vel_array = ref_root_vel_array[:min_length]

                                        # Calculate divergences between simulation and reference data
                                        obj_root_pose_divergence = obj_root_pose_array - ref_root_pose_array
                                        obj_root_vel_divergence = obj_root_vel_array - ref_root_vel_array

                                        # Batch write all object state data to hdf5 file at once
                                        obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_root_pose", data=obj_root_pose_array[:])
                                        obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/ref_root_pose", data=ref_root_pose_array[:])
                                        obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_root_pose_divergence", data=obj_root_pose_divergence[:])
                                        obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_root_vel", data=obj_root_vel_array[:])
                                        obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/ref_root_vel", data=ref_root_vel_array[:])
                                        obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_root_vel_divergence", data=obj_root_vel_divergence[:])
                                    else:
                                        get_default_logger().warning(f"No valid data for object {obj_name}, skipping divergence calculation")
                                else:
                                    get_default_logger().warning(f"Object {obj_name} has no data: obj_pose={len(obj_data['obj_root_pose'])}, ref_pose={len(obj_data['ref_root_pose'])}")

                    # save pose divergence to hdf5
                    with open(replay_json_path, 'w') as f:
                        json.dump(success_info, f, indent=4)

    print("Closing process start")
    # Wait for all video processing tasks to complete and cleanup
    if video_processor:
        video_processor.shutdown()
        print("Shut down video processor")

        # Check video file after processing (for state mode)
        if args_cli.replay_mode == "state" and video_processor:
            video_path = video_processor.get_video_path()
            if os.path.exists(video_path):
                from lwlab.scripts.teleop.teleop_launcher import get_video_duration
                video_duration = get_video_duration(video_path)
                print(f"Video duration: {video_duration} seconds")
            else:
                print(f"Video file not found: {video_path}")

            video_meta_json_path = video_path.parent / "video_meta.json"
            with open(video_meta_json_path, 'w') as f:
                json.dump({"video_duration": video_duration}, f, indent=2)

    def save_metrics():
        """Save metrics data to JSON file"""
        metrics_data = {}
        if hasattr(env_cfg, 'get_metrics'):
            metrics_data = env_cfg.get_metrics()

        # Save metrics to JSON file
        if metrics_data:
            metrics_file_path = os.path.join(save_dir, "metrics.json")
            try:
                with open(metrics_file_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                print(f"Metrics saved to: {metrics_file_path}")
            except Exception as e:
                print(f"Failed to save metrics: {e}")

    # Only save metrics for action and joint_target modes, not for state mode
    if args_cli.replay_mode != "state":
        save_metrics()
    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    if args_cli.record:
        env.recorder_manager.export_episodes()

    env.close()
    print("Close simulation env")

    # Force close Isaac Sim
    try:
        import carb
        carb.log_info("Force closing Isaac Sim...")
        # Try to close the simulation app more forcefully
        if hasattr(simulation_app, 'close'):
            simulation_app.close()

        # Additional cleanup
        import omni.kit.app
        if hasattr(omni.kit.app, 'get_app'):
            app = omni.kit.app.get_app()
            if hasattr(app, 'close'):
                app.close()

        # Force exit if needed
        import sys
        import os
        print("Force exiting Isaac Sim...")
        os._exit(0)
    except Exception as e:
        print(f"Error during force close: {e}")
        import sys
        import os
        os._exit(0)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
