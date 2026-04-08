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
import shutil
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
parser.add_argument("--remote_protocol", type=str, choices=["ipc", "restful"], default=None)
parser.add_argument("--server_host", type=str, default=None)
parser.add_argument("--server_port", type=int, default=None)
parser.add_argument("--ipc_authkey", type=str, default=None)
parser.add_argument("--debug_step_interval", type=int, default=None)
parser.add_argument("--debug_client_flow", action="store_true")

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

    if args_cli.remote_protocol is not None:
        config["remote_protocol"] = args_cli.remote_protocol
    if args_cli.server_host is not None:
        config["server_host"] = args_cli.server_host
    if args_cli.server_port is not None:
        config["server_port"] = args_cli.server_port
    if args_cli.ipc_authkey is not None:
        config["ipc_authkey"] = args_cli.ipc_authkey
    if args_cli.debug_step_interval is not None:
        config["debug_step_interval"] = args_cli.debug_step_interval
    if args_cli.debug_client_flow:
        config["debug_client_flow"] = True

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


def _get_scene_eval_status(eval_root: Path, test_num: int, num_envs: int):
    """
    Check if scene has been fully evaluated by counting episode files.
    Returns (is_complete, success_count) for skip decision and overall rate.
    """
    if not eval_root.exists():
        return False, 0
    required_count = test_num * num_envs
    success_count = 0
    completed_count = 0
    for idx in range(test_num):
        for env_idx in range(num_envs):
            run_id = f"episode{idx}_env{env_idx}"
            normal_path = eval_root / f"{run_id}.mp4"
            success_path = eval_root / f"{run_id}_success.mp4"
            if success_path.exists():
                completed_count += 1
                success_count += 1
            elif normal_path.exists():
                completed_count += 1
    if completed_count >= required_count:
        return True, success_count
    return False, 0


try:
    from lw_benchhub.scripts.policy.utils import (
        as_bool,
        create_episode_dir,
        export_episode_rollout_data,
        remote_force_export_current_episode,
        remote_get_recorded_episode_count,
        switch_remote_recorder_output,
    )
except ImportError:
    from utils import (  # type: ignore
        as_bool,
        create_episode_dir,
        export_episode_rollout_data,
        remote_force_export_current_episode,
        remote_get_recorded_episode_count,
        switch_remote_recorder_output,
    )


def main(usr_args):
    debug_client_flow = as_bool(usr_args.get("debug_client_flow", False))
    if debug_client_flow:
        print(f"[CLIENT-DEBUG] loaded config keys: {sorted(list(usr_args.keys()))}")

    remote_protocol = usr_args.get("remote_protocol", "ipc")
    server_host = usr_args.get("server_host")
    server_port = usr_args.get("server_port")
    ipc_authkey = usr_args.get("ipc_authkey", "lightwheel")
    if remote_protocol == "restful":
        from lw_benchhub.distributed.restful_proxy import RestfulRemoteEnv
        host = server_host or usr_args.get("restful_host", "127.0.0.1")
        port = int(server_port or usr_args.get("restful_port", 8000))
        print(f"Connecting to RESTful environment server at {host}:{port}...")
        env = RestfulRemoteEnv.make(address=(host, port))
    else:
        from lw_benchhub.distributed.proxy import RemoteEnv
        host = server_host or usr_args.get("ipc_host", "127.0.0.1")
        port = int(server_port or usr_args.get("ipc_port", 50000))
        print(f"Connecting to IPC environment server at {host}:{port}...")
        env = RemoteEnv.make(address=(host, port), authkey=ipc_authkey.encode())
    if debug_client_flow:
        print(f"[CLIENT-DEBUG] remote_protocol={remote_protocol}, address={host}:{port}")
    env_cfg = _build_env_cfg(usr_args)
    layouts = _normalize_layouts(env_cfg)
    if debug_client_flow:
        print(f"[CLIENT-DEBUG] normalized layouts={layouts}")

    policy_name = usr_args["policy_name"]
    policy_module = importlib.import_module("policy")
    policy_class = getattr(policy_module, policy_name)
    if debug_client_flow:
        print(f"[CLIENT-DEBUG] initializing policy={policy_name}")
    policy = policy_class(usr_args)
    if debug_client_flow:
        print("[CLIENT-DEBUG] policy initialized")

    test_num = usr_args.get("test_num", 10)  # default 10
    total_success = 0
    total_tests = 0
    eval_result_dir = Path(usr_args.get("eval_result_dir", "./eval_result"))
    save_data = as_bool(usr_args.get("save_data", False))
    save_data_only_success = as_bool(usr_args.get("save_data_only_success", True), default=True)
    save_data_dir = Path(usr_args.get("save_data_dir", eval_result_dir / "episode_data"))
    save_data_video_name = str(usr_args.get("save_data_video_name", "isaac_replay_state.mp4"))
    save_data_hdf5_success_name = str(usr_args.get("save_data_hdf5_success_name", "dataset_success.hdf5"))
    save_data_hdf5_failed_name = str(usr_args.get("save_data_hdf5_failed_name", "dataset_failed.hdf5"))
    save_data_task_name = usr_args.get("save_data_task_name")
    if save_data_task_name is not None:
        save_data_task_name = str(save_data_task_name)
    if save_data:
        save_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SAVE-DATA] Enabled. Output root: {save_data_dir}")
        if save_data_task_name:
            print(f"[SAVE-DATA] Task folder override: {save_data_task_name}")

    with (
        contextlib.suppress(KeyboardInterrupt),  # and torch.inference_mode(),
    ):
        for layout in layouts or [None]:
            if layout is not None:
                env_cfg["layout"] = layout
            scene_env_cfg = copy.deepcopy(env_cfg)
            task_name = scene_env_cfg.get("task", "unknown_task")
            export_task_name = save_data_task_name or task_name
            num_envs = int(scene_env_cfg.get("num_envs", 1))
            if layout is not None:
                eval_root = eval_result_dir / "video" / task_name / str(layout)
            else:
                eval_root = eval_result_dir / "video" / task_name

            # Skip if scene already fully evaluated
            is_complete, prev_success = _get_scene_eval_status(eval_root, test_num, num_envs)
            if is_complete:
                print(f"[SKIP] Scene {layout} already evaluated ({prev_success}/{test_num * num_envs} success).")
                total_success += prev_success
                total_tests += test_num * num_envs
                continue

            # Attach environment - on any error, skip and continue to next scene
            if debug_client_flow:
                print(f"[CLIENT-DEBUG] attach begin layout={layout}, env_cfg_keys={list(scene_env_cfg.keys())}")
            try:
                env.attach(scene_env_cfg)
            except Exception as e:
                print(f"[SKIP] Scene {layout} failed to load, continuing: {e}")
                continue
            if debug_client_flow:
                print(f"[CLIENT-DEBUG] attach done layout={layout}")
            if "actions_dim" not in usr_args:
                usr_args["actions_dim"] = env.action_space.shape[1]
            if "decimation" not in usr_args:
                usr_args["decimation"] = env.unwrapped.cfg.decimation
            if debug_client_flow:
                print(f"[CLIENT-DEBUG] actions_dim={usr_args.get('actions_dim')} decimation={usr_args.get('decimation')}")
            scene_success = 0
            save_data_for_scene = bool(save_data)
            if save_data and num_envs != 1:
                print(
                    "[SAVE-DATA] Reusing recorder export currently supports num_envs=1. "
                    f"Skip exporting scene={layout} with num_envs={num_envs}."
                )
                save_data_for_scene = False
            eval_root.mkdir(parents=True, exist_ok=True)
            for idx in tqdm.tqdm(range(test_num), desc=f"scene={layout}"):
                writers = []
                has_success = False
                episode_export_dir = None
                if save_data_for_scene:
                    episode_export_dir = create_episode_dir(save_data_dir, export_task_name)
                    dataset_success_path = episode_export_dir / save_data_hdf5_success_name
                    if not switch_remote_recorder_output(env, dataset_success_path):
                        shutil.rmtree(episode_export_dir, ignore_errors=True)
                        episode_export_dir = None
                try:
                    if debug_client_flow:
                        print(f"[CLIENT-DEBUG] episode={idx} writer init begin num_envs={num_envs} root={eval_root}")
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
                    if debug_client_flow:
                        print(f"[CLIENT-DEBUG] episode={idx} reset begin")
                    obs, _ = env.reset()
                    if debug_client_flow:
                        print(f"[CLIENT-DEBUG] episode={idx} reset done")
                        if isinstance(obs, dict):
                            print(f"[CLIENT-DEBUG] episode={idx} obs groups={list(obs.keys())}")
                    if debug_client_flow:
                        print(f"[CLIENT-DEBUG] episode={idx} policy.reset_model begin")
                    policy.reset_model()
                    if debug_client_flow:
                        print(f"[CLIENT-DEBUG] episode={idx} policy.reset_model done; eval begin")
                    has_success = policy.eval(env, obs, usr_args, writers if num_envs > 1 else writers[0])
                    if debug_client_flow:
                        print(f"[CLIENT-DEBUG] episode={idx} eval end has_success={has_success}")
                finally:
                    for writer in writers:
                        writer.__exit__(None, None, None)
                    if debug_client_flow:
                        print(f"[CLIENT-DEBUG] episode={idx} writers closed")
                if isinstance(has_success, (list, tuple, np.ndarray)):
                    success_list = [bool(x) for x in has_success]
                    if len(success_list) < num_envs:
                        success_list.extend([False] * (num_envs - len(success_list)))
                    elif len(success_list) > num_envs:
                        success_list = success_list[:num_envs]
                    scene_success += int(np.sum(success_list))
                else:
                    success_list = [bool(has_success) for _ in range(num_envs)]
                    scene_success += int(np.sum(success_list))
                for env_idx, succeeded in enumerate(success_list):
                    run_id = f"episode{idx}_env{env_idx}"
                    if succeeded:
                        old_path = eval_root / f"{run_id}.mp4"
                        new_path = eval_root / f"{run_id}_success.mp4"
                        if old_path.exists():
                            old_path.rename(new_path)
                    if save_data_for_scene and env_idx == 0 and episode_export_dir is not None:
                        recorded_demo_count = remote_get_recorded_episode_count(env)
                        if (
                            not succeeded
                            and not save_data_only_success
                            and recorded_demo_count <= 0
                        ):
                            # Reuse recorder_manager export path so failed rollouts are also persisted.
                            remote_force_export_current_episode(env)
                            recorded_demo_count = remote_get_recorded_episode_count(env)
                        export_episode_rollout_data(
                            save_data_only_success=save_data_only_success,
                            save_data_video_name=save_data_video_name,
                            save_data_hdf5_success_name=save_data_hdf5_success_name,
                            save_data_hdf5_failed_name=save_data_hdf5_failed_name,
                            episode_dir=episode_export_dir,
                            run_id=run_id,
                            succeeded=succeeded,
                            recorded_demo_count=recorded_demo_count,
                            eval_root=eval_root,
                            record_camera=usr_args.get("record_camera", []),
                        )
                total_done = (idx + 1) * num_envs
                print(f"[{layout}] Current test result: {has_success}. Success/total tested: {scene_success}/{total_done}")
            total_success += scene_success
            total_tests += test_num * num_envs
            if debug_client_flow:
                print(f"[CLIENT-DEBUG] detach begin layout={layout}")
            env.detach()
            if debug_client_flow:
                print(f"[CLIENT-DEBUG] detach done layout={layout}")
    overall_rate = total_success / total_tests if total_tests else 0.0
    print(f"Overall success rate: {overall_rate}")

    if debug_client_flow:
        print("[CLIENT-DEBUG] closing env")
    with contextlib.suppress(Exception, KeyboardInterrupt):
        env.close()
    with contextlib.suppress(Exception, KeyboardInterrupt):
        env.close_connection()
    if debug_client_flow:
        print("[CLIENT-DEBUG] env closed")


if __name__ == "__main__":
    # example: "python lw_benchhub/scripts/policy/eval_policy.py --config policy/GR00T/deploy_policy_piper.yml \
    #           --overrides --env_cfg:task SizeSorting --env_cfg:layout robocasakitchen \
    #           --instruction  "Stack objects on counter from large to small" --test_num 10
    # run the main function
    usr_args = parse_args_and_config()
    main(usr_args)
