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

"""
Teleoperation Utilities Module

This module provides utility functions for teleoperation operations including:
- Checkpoint management (save/load)
- Enhanced reset functionality with data preservation
- Episode data manipulation
- State management utilities
"""

import torch
import os
import copy


def convert_list_to_2d_tensor(data_node):
    if isinstance(data_node, dict):
        converted = {}
        for key, value in data_node.items():
            converted[key] = convert_list_to_2d_tensor(value)
        return converted
    elif isinstance(data_node, list):
        return torch.stack(data_node, dim=0)
    elif isinstance(data_node, torch.Tensor):
        return data_node
    else:
        return data_node


def reset_and_keep_to(env, episode_data, target_frame_index):
    """
    Reset environment to a specific frame while preserving episode data up to that frame.
    This implements a true "load checkpoint" functionality.

    Args:
        env: The environment to reset
        episode_data: The current episode data
        target_frame_index: The frame index to reset to (0-based)

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"[reset_and_keep_to] Resetting to frame {target_frame_index}")

    # Validate target frame index
    if target_frame_index < 0:
        print(f"[reset_and_keep_to] Error: Invalid frame index {target_frame_index}")
        return False

    # Check if episode data is available
    if not hasattr(env.recorder_manager, "_episodes") or not env.recorder_manager._episodes:
        print("[reset_and_keep_to] Error: No episode data available")
        return False

    # 1. Backup current complete episode data
    episodes_backup = {}
    try:
        for env_id, ep_data in env.recorder_manager._episodes.items():
            episodes_backup[env_id] = ep_data._data.copy()
        print(f"[reset_and_keep_to] Backed up episode data for {len(episodes_backup)} environments")
    except Exception as e:
        print(f"[reset_and_keep_to] Error backing up episode data: {e}")
        return False

    # 2. Get target frame state
    target_state = episode_data.get_state(target_frame_index)
    target_state = copy.deepcopy(target_state)
    if target_state is None:
        print(f"[reset_and_keep_to] Error: Failed to get state for frame {target_frame_index}")
        return False

    # Move state to correct device
    if hasattr(env, 'device'):
        target_state = _move_state_to_device(target_state, env.device)
    print(f"[reset_and_keep_to] Retrieved state for frame {target_frame_index}")

    # 3. Reset environment to target state
    try:
        target_state = convert_list_to_2d_tensor(target_state)
        env.reset_to_check_state(target_state, torch.tensor([0], device=env.device), is_relative=True)
        print(f"[reset_and_keep_to] Environment reset to frame {target_frame_index}")
    except Exception as e:
        print(f"[reset_and_keep_to] Error resetting environment: {e}")
        return False

    # 4. Restore and truncate episode data to target frame
    try:
        for env_id, ep_data in env.recorder_manager._episodes.items():
            if env_id in episodes_backup:
                # Truncate data to target frame (inclusive)
                truncated_data = _truncate_episode_data(episodes_backup[env_id], target_frame_index + 1)
                ep_data._data = truncated_data
                print(f"[reset_and_keep_to] Restored episode data for env {env_id} up to frame {target_frame_index}")
    except Exception as e:
        print(f"[reset_and_keep_to] Error restoring episode data: {e}")
        # Even if data restoration fails, environment is reset, so partial success
        print(f"[reset_and_keep_to] Warning: Environment reset but episode data restoration failed")

    # Render the environment
    if hasattr(env, 'sim') and hasattr(env.sim, 'render'):
        env.sim.render()

    print(f"[reset_and_keep_to] Successfully reset to frame {target_frame_index}")
    return True


def _truncate_episode_data(episode_data, max_frames):
    """
    Truncate episode data to keep only the first max_frames frames.

    Args:
        episode_data: The episode data to truncate
        max_frames: Maximum number of frames to keep

    Returns:
        dict: Truncated episode data
    """
    truncated_data = {}

    for key, value in episode_data.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            truncated_data[key] = _truncate_episode_data(value, max_frames)
        elif isinstance(value, list):
            # Truncate list to specified length
            if len(value) > max_frames:
                truncated_data[key] = value[:max_frames]
            else:
                truncated_data[key] = value
        elif isinstance(value, torch.Tensor):
            # Truncate tensor to specified length
            if value.ndim > 0 and value.shape[0] > max_frames:
                truncated_data[key] = value[:max_frames]
            else:
                truncated_data[key] = value
        else:
            # Copy other types directly
            truncated_data[key] = value

    return truncated_data


def _move_state_to_device(state, device):
    """
    Move state data to the specified device.

    Args:
        state: The state data to move
        device: Target device

    Returns:
        State data moved to target device
    """
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if isinstance(state, dict):
        return {k: _move_state_to_device(v, device) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        converted = [_move_state_to_device(v, device) for v in state]
        return type(state)(converted)
    return state


def save_checkpoint(env, checkpoint_path):
    """
    Save current frame index to checkpoint file.

    Args:
        env: The environment
        checkpoint_path: Path to save checkpoint

    Returns:
        int: frame index, -1 if failed
    """
    print("save checkpoint")
    try:
        episodes = getattr(env.recorder_manager, "_episodes", None)
        if not isinstance(episodes, dict) or 0 not in episodes:
            print("[checkpoint] save failed: recorder episode not ready (env_id=0)")
            return -1

        episode_data = episodes[0]
        frame_index = _get_current_frame_index(episode_data)
        if frame_index is None:
            print("[checkpoint] save failed: unable to infer current frame index from recorder")
            return -1

        payload = {"frame_index": int(frame_index)}
        torch.save(payload, checkpoint_path)
        print(f"[checkpoint] saved frame_index={frame_index} to: {checkpoint_path}")
        # Minimal UI update if env has label binding
        try:
            if hasattr(env, "_task_desc_label") and hasattr(env, "_base_desc") and env._task_desc_label is not None:
                env._task_desc_label.text = env._base_desc + f"\n Checkpoints: saved (frame {frame_index})"
        except Exception:
            pass
        return frame_index
    except Exception as e:
        print(f"[checkpoint] save failed: {e}")
        return -1


def load_checkpoint(env, checkpoint_path):
    """
    Load checkpoint and reset to saved frame with data preservation.

    Args:
        env: The environment
        checkpoint_path: Path to checkpoint file

    Returns:
        bool: True if successful, False otherwise
    """
    print("load checkpoint")
    # Minimal UI update: show loading state
    try:
        if hasattr(env, "_task_desc_label") and hasattr(env, "_base_desc") and env._task_desc_label is not None:
            env._task_desc_label.text = env._base_desc + "\n Checkpoints: loading..."
            if hasattr(env, 'sim') and hasattr(env.sim, 'render'):
                env.sim.render()
    except Exception:
        pass
    if not os.path.exists(checkpoint_path):
        print(f"[checkpoint] file not found: {checkpoint_path}")
        return False

    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
        frame_index = int(payload.get("frame_index", -1))
        if frame_index < 0:
            print(f"[checkpoint] load failed: invalid frame_index in payload: {payload}")
            return False

        episodes = getattr(env.recorder_manager, "_episodes", None)
        if not isinstance(episodes, dict) or 0 not in episodes:
            print("[checkpoint] load failed: recorder episode not ready (env_id=0)")
            return False

        episode_data = episodes[0]

        # Use enhanced reset_and_keep_to method for true load functionality
        success = reset_and_keep_to(env, episode_data, frame_index)
        if success:
            print(f"[checkpoint] Successfully loaded frame_index={frame_index} from: {checkpoint_path}")
        else:
            print(f"[checkpoint] Failed to load frame_index={frame_index}")
        # Minimal UI update: show loaded or failed
        try:
            if hasattr(env, "_task_desc_label") and hasattr(env, "_base_desc") and env._task_desc_label is not None:
                if success:
                    env._task_desc_label.text = env._base_desc + f"\n Checkpoints: loaded (frame {frame_index})"
                else:
                    env._task_desc_label.text = env._base_desc + "\n Checkpoints: load failed"
                if hasattr(env, 'sim') and hasattr(env.sim, 'render'):
                    env.sim.render()
        except Exception:
            pass
        return success
    except Exception as e:
        print(f"[checkpoint] load failed: {e}")
        return False


def _get_current_frame_index(episode_data):
    """
    Infer current frame index from in-memory episode data.

    Strategy:
    1) Prefer the canonical path used by replay: states/articulation/robot/joint_position
    2) Fallback to finding the first leaf list or tensor under states and use its length - 1

    Args:
        episode_data: The episode data

    Returns:
        int or None: Current frame index, or None if cannot determine
    """
    try:
        data = episode_data._data
    except Exception:
        return None

    def get_len(node):
        if isinstance(node, list):
            return (len(node) - 1) if len(node) > 0 else None
        if isinstance(node, torch.Tensor):
            return (int(node.shape[0]) - 1) if node.ndim > 0 and node.shape[0] > 0 else None
        return None

    def get_from_path(d, path):
        cur = d
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return get_len(cur)

    preferred_paths = [
        ["states", "articulation", "robot", "joint_position"],
        ["states", "articulation", "robot", "root_state_w"],
    ]
    for p in preferred_paths:
        idx = get_from_path(data, p)
        if idx is not None:
            return idx

    def find_first_len(node):
        ln = get_len(node)
        if ln is not None:
            return ln
        if isinstance(node, dict):
            for val in node.values():
                res = find_first_len(val)
                if res is not None:
                    return res
        return None

    if isinstance(data, dict) and "states" in data:
        return find_first_len(data["states"])
    return None


def _get_state_by_frame(episode_data, frame_index: int):
    """
    Return state dict at frame_index from in-memory episode.

    Prefer EpisodeData.get_state; if it raises due to list-based storage,
    reconstruct by indexing lists/tensors under the "states" subtree.

    Args:
        episode_data: The episode data
        frame_index: The frame index to retrieve

    Returns:
        dict or None: State data for the specified frame
    """
    # Try native helper first
    try:
        state = episode_data.get_state(frame_index)
        return state
    except Exception:
        pass

    # Fallback: reconstruct from raw in-memory structure
    data = getattr(episode_data, "_data", None)
    if not isinstance(data, dict) or "states" not in data:
        return None

    def index_node(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                val = index_node(v)
                if val is None:
                    return None
                out[k] = val
            return out
        if isinstance(node, list):
            if frame_index < 0 or frame_index >= len(node):
                return None
            return node[frame_index]
        if isinstance(node, torch.Tensor):
            if node.ndim == 0:
                return node
            if frame_index < 0 or frame_index >= int(node.shape[0]):
                return None
            return node[frame_index, None]
        # Unsupported leaf type
        return None

    return index_node(data["states"])


def quick_rewind(env, frames_back=10):
    """
    Quick rewind function that goes back a specified number of frames.

    Args:
        env: The environment
        frames_back: Number of frames to go back (default: 10)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        episodes = getattr(env.recorder_manager, "_episodes", None)
        if not isinstance(episodes, dict) or 0 not in episodes:
            print("[quick_rewind] Error: No episode data available")
            return False

        episode_data = episodes[0]
        current_frame = _get_current_frame_index(episode_data)
        if current_frame is None:
            print("[quick_rewind] Error: Cannot determine current frame")
            return False

        target_frame = max(0, current_frame - frames_back)
        print(f"[quick_rewind] Going back from frame {current_frame} to frame {target_frame}")

        return reset_and_keep_to(env, episode_data, target_frame)
    except Exception as e:
        print(f"[quick_rewind] Error: {e}")
        return False
