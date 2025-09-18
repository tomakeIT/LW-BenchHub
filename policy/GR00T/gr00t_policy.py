import os
import sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

import torch
from typing import Dict, Any
from policy.base import BasePolicy
try:
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
except ImportError as e:
    print(f"gr00t not found, please install gr00t first: {e}")


class GR00TPolicy(BasePolicy):
    """GR00T Policy Implementation"""

    def __init__(self, usr_args):
        super().__init__(usr_args)

    def _load_policy(self):
        """Load the policy from the model path."""
        # Use the same data preprocessor as the loaded fine-tuned ckpts
        self.data_config = DATA_CONFIG_MAP[self.usr_args["data_config"]]

        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()
        # load the policy
        return Gr00tPolicy(
            model_path=self.usr_args["ckpt_setting"],
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.usr_args["embodiment_tag"],
            denoising_steps=self.usr_args["denoising_steps"],
            device=self.simulation_device,
        )

    def get_model(self, usr_args):
        """Get GR00T model instance"""
        observation_config = usr_args.get("observation_config", None)
        action_config = usr_args.get("action_config", None)
        self.simulation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observation_config = observation_config or {}
        self.action_config = action_config or {}
        self.model = self._load_policy()

    def _build_observation_window(self, obs):
        custom_mapping = self.observation_config.get("custom_mapping", {})
        obs_window = {"annotation.human.action.task_description": [self.instruction]}
        for key, mapping in custom_mapping.items():
            if isinstance(mapping, dict):
                obs_window[key] = obs[next(iter(mapping.keys()))][..., next(iter(mapping.values()))]
                obs_window[key] = obs_window[key][None, ...]  # N, env, ...
            else:
                obs_window[key] = obs[mapping][None, ...]  # N, env, H,W,C
        return obs_window

    def encode_obs(self, observation):
        observation = super().encode_obs(observation, transpose=False, keep_dim_env=True)
        observation = self._build_observation_window(observation)
        return observation

    def _mapping_action(self, actions):
        robot_actions = []
        for key in self.action_config.keys():
            robot_actions.append(torch.tensor(actions[key], device=self.simulation_device))
        robot_actions = torch.concat(robot_actions, dim=-1)
        return robot_actions

    def get_action(self, observation):
        robot_action_policy = self.model.get_action(observation)
        robot_actions = self._mapping_action(robot_action_policy)
        return robot_actions

    def eval(self, task_env: Any, observation: Dict[str, Any],
             usr_args: Dict[str, Any], video_writer: Any) -> bool:
        for _ in range(usr_args['time_out_limit']):
            observation = self.encode_obs(observation)
            actions = self.get_action(observation)  # env, horizon, action_dim
            for i in range(self.usr_args["num_feedback_actions"]):
                observation, _, terminated, _, _ = task_env.step(actions[:, i])
                self.add_video_frame(video_writer, observation, usr_args['record_camera'])
                if terminated:
                    return terminated
        return terminated

    def reset_model(model):
        pass
