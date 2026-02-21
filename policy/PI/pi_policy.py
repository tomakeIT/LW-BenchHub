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

import dataclasses
import os
import sys
from typing import Any, Dict

import cv2
import numpy as np
import torch

# Add current directory to path
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
from policy.base import BasePolicy
try:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
except ImportError as e:
    print(f"PI not found, please install PI first: {e} , if you not run pi policy, please ignore this error")


class PIPolicy(BasePolicy):
    """PI Policy Implementation"""

    def __init__(self, usr_args: Dict[str, Any]):
        super().__init__(usr_args)

    def get_model(self, usr_args: Dict[str, Any]):
        """Get PI model instance"""

        train_config_name = usr_args["train_config_name"]
        checkpoint = usr_args["checkpoint"]
        jax_params_dtype = usr_args.get("jax_params_dtype")
        observation_config = usr_args.get("observation_config", None)
        config = _config.get_config(train_config_name)
        jax_model_dtype = usr_args.get("jax_model_dtype")
        if jax_model_dtype is not None:
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, dtype=jax_model_dtype),
            )
        self.action_chunk_size = usr_args["action_chunk_size"]
        self.model = _policy_config.create_trained_policy(
            config,
            checkpoint,
            jax_params_dtype=jax_params_dtype,
        )
        print("loading PI0.5 model success!")
        self.observation_window = None
        self.observation_config = observation_config or {}
        # breakpoint()
        print("Config:", config)

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.model.infer(self.observation_window)["actions"]

    def encode_obs(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Encode observation"""

        observation = super().encode_obs(observation)
        observation = self._build_observation_window(observation)
        return observation

    def _build_observation_window(self, obs):
        custom_mapping = self.observation_config.get("custom_mapping", {})
        obs_window = {"prompt": self.instruction}
        for key, mapping in custom_mapping.items():
            if isinstance(mapping, dict):
                obs_window[key] = {k: obs[v] for k, v in mapping.items()}
            else:
                obs_window[key] = obs[mapping]
        return obs_window


    def custom_action_mapping(self, action: torch.Tensor) -> torch.Tensor:
        # define your own action mapping here
        return action

    def custom_obs_mapping(self, obs):
        # define your own obs mapping here
        return obs

    def eval(self, task_env: Any, observation: Dict[str, Any],
             usr_args: Dict[str, Any], video_writer: Any) -> bool:
        """Evaluate PI policy"""
        num_envs = self._infer_num_envs(observation)
        done = np.zeros(num_envs, dtype=bool)
        for _ in range(usr_args['time_out_limit']):
            actions_per_env = []
            action_dim = None
            for env_idx in range(num_envs):
                if done[env_idx]:
                    actions_per_env.append(None)
                    continue
                obs_env = self._slice_observation(observation, env_idx)
                self.observation_window = self.encode_obs(obs_env)
                self.observation_window = self.custom_obs_mapping(self.observation_window)
                actions = self.get_action()
                actions = self.custom_action_mapping(actions)
                actions_per_env.append(actions)
                if action_dim is None:
                    action_dim = actions.shape[-1]
            if action_dim is None:
                return done

            chunk = self.action_chunk_size
            for i in range(chunk):
                action_batch = []
                for env_idx in range(num_envs):
                    if done[env_idx]:
                        action_batch.append(np.zeros(action_dim, dtype=np.float32))
                    else:
                        action_batch.append(actions_per_env[env_idx][i])
                actions_tensor = torch.from_numpy(np.stack(action_batch, axis=0)).float().cuda()
                observation, terminated = self.step_environment(task_env, actions_tensor, usr_args)
                for env_idx in range(num_envs):
                    writer = video_writer[env_idx] if isinstance(video_writer, (list, tuple)) else video_writer
                    self.add_video_frame(writer, observation, usr_args['record_camera'], env_idx=env_idx)
                terminated = self._normalize_terminated(terminated, num_envs)
                done = np.logical_or(done, terminated)
                if done.all():
                    return done
        return done

    def _infer_num_envs(self, observation: Dict[str, Any]) -> int:
        for key, value in observation.get("policy", {}).items():
            if torch.is_tensor(value):
                return int(value.shape[0])
        return 1

    def _slice_observation(self, observation: Dict[str, Any], env_idx: int) -> Dict[str, Any]:
        obs_env = {"policy": {}, "embodiment_general_obs": {}}
        for group in obs_env.keys():
            for key, value in observation.get(group, {}).items():
                if torch.is_tensor(value):
                    obs_env[group][key] = value[env_idx:env_idx + 1]
                else:
                    obs_env[group][key] = value
        return obs_env

    def _normalize_terminated(self, terminated, num_envs: int) -> np.ndarray:
        if torch.is_tensor(terminated):
            terminated = terminated.cpu().numpy()
        terminated = np.asarray(terminated)
        if terminated.shape == ():
            terminated = np.full((num_envs,), bool(terminated), dtype=bool)
        return terminated.astype(bool)

    def reset_model(self) -> None:
        """Reset PI model state"""
        self._reset_obsrvationwindows()

    def _reset_obsrvationwindows(self):
        self.observation_window = None
        print("successfully unset observation window")
