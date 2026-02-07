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
PI Policy Implementation
"""

import torch
import sys
import os
from typing import Dict, Any

# Add current directory to path
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
from policy.base import BasePolicy
try:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ImportError as e:
    print(f"PI not found, please install PI first: {e} , if you not run pi policy, please ignore this error")


class PIPolicyActionReplay(BasePolicy):
    """PI Policy Implementation"""

    def __init__(self, usr_args: Dict[str, Any]):
        super().__init__(usr_args)

    def get_model(self, usr_args: Dict[str, Any]):
        """Get PI model instance"""

        train_config_name = usr_args["train_config_name"]
        checkpoint = usr_args["checkpoint"]
        observation_config = usr_args.get("observation_config", None)
        config = _config.get_config(train_config_name)
        self.action_chunk_size = usr_args["action_chunk_size"]
        # self.model = _policy_config.create_trained_policy(config, checkpoint)
        print("loading PI0.5 model success!")
        self.observation_window = None
        self.observation_config = observation_config or {}

    def custom_action_mapping(self, action: torch.Tensor) -> torch.Tensor:
        # define your own action mapping here
        return action
    
    def get_action(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Get action from observation"""
        return self.custom_action_mapping(observation)

    def eval(self, task_env: Any, observation: Dict[str, Any],
             usr_args: Dict[str, Any], video_writer: Any) -> bool:
        """Replay actions from a LeRobot dataset episode."""
        repo_id = usr_args.get("lerobot_repo_id", "/home/jialeng/Desktop/LightwheelData/Tasks_lerobot")
        episode_index = self._get_episode_index(usr_args)
        dataset = lerobot_dataset.LeRobotDataset(repo_id)
        actions = self._collect_episode_actions(dataset, episode_index, repo_id)

        terminated = False
        for action in actions:
            action = torch.as_tensor(action).float()
            action = self.custom_action_mapping(action)
            if torch.cuda.is_available():
                action = action.cuda()
            observation, terminated = self.step_environment(task_env, action.unsqueeze(0), usr_args)
            self.add_video_frame(video_writer, observation, usr_args['record_camera'])
            if terminated:
                return terminated
        return terminated

    @staticmethod
    def _get_episode_index(usr_args: Dict[str, Any]) -> int:
        for key in ("episode_index", "episode_idx", "episode"):
            if key in usr_args:
                return int(usr_args[key])
        raise ValueError("Missing episode index in usr_args (episode_index/episode_idx/episode).")

    @staticmethod
    def _to_int(value: Any) -> int:
        if hasattr(value, "item"):
            value = value.item()
        elif isinstance(value, (list, tuple)):
            value = value[0] if value else -1
        return int(value)

    def _collect_episode_actions(self, dataset: Any, episode_index: int, repo_id: str) -> list:
        num_episodes = getattr(dataset, "num_episodes", None)
        if num_episodes is not None and episode_index >= int(num_episodes):
            raise ValueError(f"episode_index {episode_index} out of range (num_episodes={num_episodes}).")

        actions = []
        index_map = getattr(dataset, "episode_data_index", None)
        if not (isinstance(index_map, dict) and "from" in index_map and "to" in index_map):
            raise RuntimeError("Dataset missing episode_data_index; cannot select episode without full scan.")

        start = int(index_map["from"][episode_index])
        end = int(index_map["to"][episode_index])
        for i in range(start, end):
            sample = dataset[i]
            actions.append(sample["actions"])

        if not actions:
            raise ValueError(f"No actions found for episode {episode_index} in {repo_id}.")

        return actions

    def reset_model(self) -> None:
        """Reset PI model state"""
        self._reset_obsrvationwindows()

    def _reset_obsrvationwindows(self):
        self.observation_window = None
        print("successfully unset observation window")
