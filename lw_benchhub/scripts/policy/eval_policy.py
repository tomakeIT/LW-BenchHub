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
import contextlib
import copy
import importlib
import json
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import tqdm
import yaml

sys.path.append("./")
sys.path.append("../../policy")

# add argparse arguments
# Get project root directory (assuming script is in lw_benchhub/scripts/policy/)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
default_config_path = project_root / "policy" / "PI" / "deploy_policy_lerobot.yml"

parser = argparse.ArgumentParser(description="Eval policy in Isaac Lab environments.")
parser.add_argument("--config", type=str, default=str(default_config_path))
parser.add_argument("--overrides", nargs=argparse.REMAINDER)

# parse the arguments
args_cli = parser.parse_args()


def parse_args_and_config():

    with open(args_cli.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]

            try:
                value = eval(value)
            except Exception:
                pass

            # use ':' to split config
            if ':' in key:
                keys = key.split(':')
                current_level = override_dict
                for k in keys[:-1]:
                    if k not in current_level:
                        current_level[k] = {}
                    current_level = current_level[k]
                current_level[keys[-1]] = value
            else:
                override_dict[key] = value

        return override_dict

    def deep_merge(original, update):
        for key, value in update.items():
            if (key in original and
                isinstance(original[key], dict) and
                    isinstance(value, dict)):
                deep_merge(original[key], value)
            else:
                original[key] = value
        return original

    if args_cli.overrides:
        overrides = parse_override_pairs(args_cli.overrides)
        config = deep_merge(config, overrides)

    return config


def _build_env_cfg(usr_args):
    from lw_benchhub.distributed.restful import DotDict

    env_cfg = DotDict(usr_args.get("env_cfg") or {})
    defaults = {
        "scene_backend": "robocasa",
        "task_backend": "robocasa",
        "device": "cuda:0",
        "robot_scale": 1.0,
        "first_person_view": False,
        "disable_fabric": False,
        "num_envs": 1,
        "usd_simplify": False,
        "video": False,
        "for_rl": False,
        "variant": "Visual",
        "concatenate_terms": False,
        "distributed": False,
        "seed": 42,
        "sources": None,
        "object_projects": None,
        "execute_mode": "eval",
        "replay_cfgs": {"add_camera_to_observation": True},
    }
    for key, value in defaults.items():
        if key not in env_cfg:
            env_cfg[key] = value
    return env_cfg


def _normalize_layouts(env_cfg):
    if "layouts" in env_cfg and env_cfg["layouts"]:
        layouts = env_cfg["layouts"]
    else:
        layouts = env_cfg.get("layout")
    if isinstance(layouts, (list, tuple)):
        return list(layouts)
    if layouts is None:
        return []
    return [layouts]


def main(usr_args):

    from lw_benchhub.distributed.proxy import RemoteEnv
    env = RemoteEnv.make(address=('127.0.0.1', 50000), authkey=b'lightwheel')
    env_cfg = _build_env_cfg(usr_args)
    layouts = _normalize_layouts(env_cfg)

    policy_name = usr_args["policy_name"]
    policy_module = importlib.import_module("policy")
    policy_class = getattr(policy_module, policy_name)
    policy = policy_class(usr_args)

    test_num = usr_args.get("test_num", 10)  # default 10
    total_success = 0
    total_tests = 0
    per_scene_results = []
    with (
        contextlib.suppress(KeyboardInterrupt),  # and torch.inference_mode(),
    ):
        for layout in layouts or [None]:
            if layout is not None:
                env_cfg["layout"] = layout
            scene_env_cfg = copy.deepcopy(env_cfg)
            env.attach(scene_env_cfg)
            if "actions_dim" not in usr_args:
                usr_args["actions_dim"] = env.action_space.shape[1]
            if "decimation" not in usr_args:
                usr_args["decimation"] = env.unwrapped.cfg.decimation
            num_envs = int(scene_env_cfg.get("num_envs", 1))
            scene_success = 0
            for idx in tqdm.tqdm(range(test_num), desc=f"scene={layout}"):
                eval_root = Path(f"./eval_result/video/{layout}") if layout else Path("./eval_result/video")
                eval_root.mkdir(parents=True, exist_ok=True)
                writers = []
                try:
                    for env_idx in range(num_envs):
                        eval_video_path = eval_root / f"episode{idx}_env{env_idx}.mp4"
                        writers.append(
                            media.VideoWriter(
                                path=eval_video_path,
                                shape=(usr_args["height"], usr_args["width"] * len(usr_args["record_camera"])),
                                fps=30,
                            )
                        )
                        writers[-1].__enter__()
                    obs, _ = env.reset()
                    policy.reset_model()
                    has_success = policy.eval(env, obs, usr_args, writers if num_envs > 1 else writers[0])
                finally:
                    for writer in writers:
                        writer.__exit__(None, None, None)
                if isinstance(has_success, (list, tuple, np.ndarray)):
                    scene_success += int(np.sum(has_success))
                else:
                    scene_success += int(bool(has_success))
                total_done = (idx + 1) * num_envs
                print(f"[{layout}] Current test result: {has_success}. Success/total tested: {scene_success}/{total_done}")
            per_scene_results.append(
                {
                    "layout": layout,
                    "test_count": test_num * num_envs,
                    "success_count": scene_success,
                    "success_rate": scene_success / (test_num * num_envs) if test_num else 0.0,
                }
            )
            total_success += scene_success
            total_tests += test_num * num_envs
            env.detach()
    overall_rate = total_success / total_tests if total_tests else 0.0
    print(f"Overall success rate: {overall_rate}")

    results = {
        "test_count": total_tests,
        "success_count": total_success,
        "success_rate": overall_rate,
        "per_scene": per_scene_results,
    }
    with open("./eval_result/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    env.close()
    env.close_connection()


if __name__ == "__main__":
    # example: "python lw_benchhub/scripts/policy/eval_policy.py --config policy/GR00T/deploy_policy_piper.yml \
    #           --overrides --env_cfg:task SizeSorting --env_cfg:layout robocasakitchen \
    #           --instruction  "Stack objects on counter from large to small" --test_num 10
    # run the main function
    usr_args = parse_args_and_config()
    main(usr_args)
