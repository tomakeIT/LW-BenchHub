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

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import random
from functools import partial

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher
from isaaclab.utils.datasets import HDF5DatasetFileHandler

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--remote_protocol", type=str, default="ipc", help="Remote protocol, can be ipc or restful")
parser.add_argument("--ipc_host", type=str, default="127.0.0.1", help="IPC host")
parser.add_argument("--ipc_port", type=int, default=50000, help="IPC port")
parser.add_argument("--ipc_authkey", type=str, default="lightwheel", help="IPC authkey")
parser.add_argument("--restful_host", type=str, default="0.0.0.0", help="Restful host")
parser.add_argument("--restful_port", type=int, default=8000, help="Restful port")
parser.add_argument("--episode_data", type=str, default=None, help="Path to HDF5 episode data file")
parser.add_argument("--episode_index", type=int, default=0, help="Index of episode to load from HDF5 file")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

if args_cli.remote_protocol == "restful":
    from lw_benchhub.distributed.restful import RestfulEnvWrapper
    RemoteEnvWrapper = partial(RestfulEnvWrapper, address=(args_cli.restful_host, args_cli.restful_port))
elif args_cli.remote_protocol == "ipc":   # ipc
    from lw_benchhub.distributed.ipc import IpcDistributedEnvWrapper
    RemoteEnvWrapper = partial(IpcDistributedEnvWrapper, address=(args_cli.ipc_host, args_cli.ipc_port), authkey=args_cli.ipc_authkey.encode())


app_launcher_args = vars(args_cli)
app_launcher = None
simulation_app = None


def make_env_cfg(cfg):
    from isaaclab_tasks.utils import parse_env_cfg

    if "-" in cfg.task:
        env_cfg = parse_env_cfg(
            cfg.task, device=cfg.device, num_envs=cfg.num_envs, use_fabric=not cfg.disable_fabric
        )
        task_name = cfg.task
    else:
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode, str_to_execute_mode

        env_cfg = parse_env_cfg(
            scene_backend=cfg.scene_backend,
            task_backend=cfg.task_backend,
            task_name=cfg.task,
            robot_name=cfg.robot,
            scene_name=cfg.layout,
            rl_name=cfg.rl,
            robot_scale=cfg.robot_scale,
            device=cfg.device,
            num_envs=cfg.num_envs,
            use_fabric=not cfg.disable_fabric,
            first_person_view=cfg.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=str_to_execute_mode(cfg.execute_mode),
            headless_mode=args_cli.headless,
            usd_simplify=cfg.usd_simplify,
            seed=cfg.seed,
            sources=cfg.sources,
            object_projects=cfg.object_projects,
            replay_cfgs=cfg.replay_cfgs,
            initial_state= getattr(cfg, 'initial_state', None),
            resample_objects_placement_on_reset=getattr(cfg, 'resample_objects_placement_on_reset', None),
            resample_robot_placement_on_reset=getattr(cfg, 'resample_robot_placement_on_reset', None),
        )
        task_name = f"Robocasa-{cfg.task}-{cfg.robot}-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True
        )

    env_cfg.observations.policy.concatenate_terms = cfg.concatenate_terms
    # modify configuration
    env_cfg.terminations.time_out = None
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = cfg.num_envs if cfg.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = cfg.device if cfg.device is not None else env_cfg.sim.device
    # multi-gpu training config
    if cfg.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # randomly sample a seed if seed = -1
    if cfg.seed == -1:
        cfg.seed = random.randint(0, 10000)
    return task_name, env_cfg


def initialize_env_cfg_from_hdf5(env_cfg, episode_data_path, episode_data_index, width=None, height=None):
    """Initialize environment configuration from HDF5 episode data file.
    
    Args:
        env_cfg: Environment configuration object (will be modified in place)
        episode_data_path: Path to HDF5 episode data file
        episode_data_index: Index of episode to load
        width: Render width (optional)
        height: Render height (optional)
    
    Returns:
        tuple: (updated_env_cfg, initial_state, episode_data_handler)
            - updated_env_cfg: Updated environment configuration
            - initial_state: Initial state from episode data (None if not loaded)
            - episode_data_handler: HDF5DatasetFileHandler instance (None if not loaded)
    """
    if not episode_data_path or not os.path.exists(episode_data_path):
        return env_cfg, None, None
    
    
    # Open HDF5 file
    episode_data_handler = HDF5DatasetFileHandler()
    episode_data_handler.open(episode_data_path)
    episode_count = episode_data_handler.get_num_episodes()
    
    if episode_count == 0:
        raise ValueError(f"No episodes found in {episode_data_path}")
    
    if episode_data_index >= episode_count:
        raise ValueError(f"Episode index {episode_data_index} out of range. File has {episode_count} episodes.")
    
    # Get episode name
    episode_names = list(episode_data_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    episode_name = episode_names[episode_data_index]
    
    print(f"Loading initial state from episode {episode_data_index} ({episode_name}) in {episode_data_path}")
    
    # Get device from env_cfg
    device_str = getattr(env_cfg, 'sim', None)
    if device_str and hasattr(device_str, 'device'):
        device = torch.device(device_str.device)
    else:
        device = torch.device('cuda:0')
    
    # Load episode data
    episode_data = episode_data_handler.load_episode(episode_name, device)
    initial_state = episode_data.get_initial_state()
    print(f"Loaded initial state from episode data")
    
    # Update replay_cfgs render_resolution if width/height provided
    if width and height and hasattr(env_cfg, 'replay_cfgs') and isinstance(env_cfg.replay_cfgs, dict):
        env_cfg.replay_cfgs['render_resolution'] = (width, height)
    
    # Ensure num_envs is 1 for replay
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'num_envs'):
        env_cfg.scene.num_envs = 1  # reset_to method cannot take tensor as env_ids
    
    return env_cfg, initial_state, episode_data_handler


def make_env(cfg, launcher_args, args_override: dict = None):
    global app_launcher
    global simulation_app
    if app_launcher is None:
        args_override = args_override or {}
        app_launcher_args_ = {**launcher_args, **args_override}
        app_launcher = AppLauncher(app_launcher_args_)
        simulation_app = app_launcher.app
    from isaaclab.envs import ManagerBasedEnv
    from lw_benchhub.utils.place_utils.env_utils import warmup_rendering
    initial_state = None
    env_args = {}
    
    # 支持从配置文件或命令行参数读取episode_data和episode_index
    # 优先使用命令行参数,如果没有则从cfg读取
    episode_data_path = args_cli.episode_data
    episode_data_index = args_cli.episode_index
    
    if episode_data_path is None and hasattr(cfg, 'episode_data') and cfg.episode_data:
        episode_data_path = cfg.episode_data
        print(f"[INFO] Using episode_data from config: {episode_data_path}")
    
    if episode_data_path and hasattr(cfg, 'episode_index') and cfg.episode_index is not None:
        episode_data_index = cfg.episode_index
        print(f"[INFO] Using episode_index from config: {episode_data_index}")
    
    # Initialize from HDF5 if episode_data is provided (before creating env_cfg)
    # This allows us to override cfg values from HDF5 before calling make_env_cfg
    if episode_data_path:
        
        # Load environment arguments from HDF5
        episode_data_handler_temp = HDF5DatasetFileHandler()
        episode_data_handler_temp.open(episode_data_path)
        env_args = json.loads(episode_data_handler_temp._hdf5_data_group.attrs["env_args"])
        
        # Get episode name for replay_cfgs
        episode_names = list(episode_data_handler_temp.get_episode_names())
        episode_names.sort(key=lambda x: int(x.split("_")[-1]))
        episode_name = episode_names[episode_data_index]
        device = torch.device(getattr(cfg, "device", "cuda:0"))
        initial_state = episode_data_handler_temp.load_episode(episode_name, device).get_initial_state()
        # Load initial state early so it can be passed into parse_env_cfg
        episode_data_handler_temp.close()
        
        print(f"[INFO] Loaded episode '{episode_name}' (index {episode_data_index}) from {episode_data_path}")
        
        # Override cfg with values from HDF5
        cfg.task = env_args.get("task_name", getattr(cfg, 'task', None))
        cfg.robot = env_args.get("robot_name", getattr(cfg, 'robot', None))
        if "scene_type" in env_args:
            cfg.layout = f"{env_args['scene_type']}-{env_args['layout_id']}-{env_args['style_id']}"
        cfg.scene_backend = env_args.get("scene_backend", getattr(cfg, 'scene_backend', 'robocasa'))
        cfg.task_backend = env_args.get("task_backend", getattr(cfg, 'task_backend', 'robocasa'))
        cfg.seed = env_args.get("seed", getattr(cfg, 'seed', 42))
        cfg.sources = env_args.get("sources", getattr(cfg, 'sources', None))
        cfg.object_projects = env_args.get("object_projects", getattr(cfg, 'object_projects', None))
        cfg.num_envs = 1  # reset_to method cannot take tensor as env_ids
        cfg.resample_objects_placement_on_reset = False
        cfg.resample_robot_placement_on_reset = False
        # Force execute_mode for action replay if episode data is provided.
        cfg.execute_mode = "eval"
        cfg.initial_state = initial_state
        
        # evict cache_usd_version
        if "cache_usd_version" in env_args:
            del env_args["cache_usd_version"]
        # Set replay_cfgs (render_resolution will be updated later if width/height available)

        cfg.replay_cfgs.update({
            "hdf5_path": episode_data_path,
            "ep_meta": env_args,
            "ep_names": episode_name,
            "add_camera_to_observation": True
        })

    
    # # Create environment configuration
    task_name, env_cfg = make_env_cfg(cfg)
    # save cfg to config.json
    
    # Initialize from HDF5 if episode_data is provided (after creating env_cfg)
    # Update replay_cfgs with actual render resolution and load initial state
    episode_data_handler = None
    if args_cli.episode_data:
        # Get render resolution from app_launcher if available
        width = getattr(app_launcher, '_width', None) if app_launcher else None
        height = getattr(app_launcher, '_height', None) if app_launcher else None
        
        # Update render_resolution in replay_cfgs if width/height available
        if width and height and hasattr(env_cfg, 'replay_cfgs') and isinstance(env_cfg.replay_cfgs, dict):
            env_cfg.replay_cfgs['render_resolution'] = (width, height)
        env_cfg, initial_state, episode_data_handler = initialize_env_cfg_from_hdf5(
            env_cfg,
            args_cli.episode_data,
            args_cli.episode_index,
            width=width,
            height=height
        )

    gym_env = gym.make(
        task_name,
        cfg=env_cfg,
        render_mode="rgb_array" if cfg.video else None
    )
    
    env: ManagerBasedEnv = gym_env.unwrapped
    warmup_rendering(env)
    
    if initial_state is not None:
        from lw_benchhub.utils.place_utils.env_utils import reset_physx, sample_object_placements
        import copy
        is_relative = False
        if hasattr(args_cli, 'episode_data') and args_cli.episode_data:
            if "robot_name" in env_args:
                is_relative = env_args["robot_name"].endswith("Rel")
        
        # 配置需要随机化的物体列表
        # 支持从cfg中读取,或使用默认列表
        randomize_objects = getattr(cfg, 'randomize_objects_on_reset', None)
        if randomize_objects is None:
            # 默认情况下,对task中的主要交互物体进行随机化
            # 例如: 对于PickCoffeeMug任务,随机化'obj'(咖啡杯)
            randomize_objects = []  # 空列表表示完全固定场景
        
        # 配置随机化参数
        randomize_pos_offset = getattr(cfg, 'randomize_pos_offset', 0.05)  # 默认±5cm
        randomize_rot_offset = getattr(cfg, 'randomize_rot_offset', 0.3)   # 默认±17度(0.3 rad)
        
        # 机器人随机化参数
        randomize_robot = getattr(cfg, 'randomize_robot', False)
        randomize_robot_pos_offset = getattr(cfg, 'randomize_robot_pos_offset', 0.10)  # 默认±10cm
        randomize_robot_rot_offset = getattr(cfg, 'randomize_robot_rot_offset', 0.2)   # 默认±11.5度
        
        # 确保base_seed不是None
        base_seed = cfg.seed if hasattr(cfg, 'seed') and cfg.seed is not None else 42
        
        # 打印配置信息
        if randomize_objects and len(randomize_objects) > 0:
            print("="*60)
            print("[INFO] Object randomization enabled for eval:")
            print(f"  Objects to randomize: {randomize_objects}")
            print(f"  Position offset: ±{randomize_pos_offset*100:.1f}cm")
            print(f"  Rotation offset: ±{randomize_rot_offset*180/3.14159:.1f}°")
            print(f"  Base seed: {base_seed}")
            if randomize_robot:
                print(f"\n[INFO] Robot randomization enabled:")
                print(f"  Position offset: ±{randomize_robot_pos_offset*100:.1f}cm")
                print(f"  Rotation offset: ±{randomize_robot_rot_offset*180/3.14159:.1f}°")
            print("\n  Reproducibility guarantee:")
            print(f"    - Same session: Each rollout has different randomization")
            print(f"    - Different sessions: Same rollout sequence if seed={base_seed}")
            print("="*60)
        else:
            print("[INFO] Object randomization disabled - scene will be completely fixed")
        
        # 用于跟踪reset次数,确保每次随机化都不同但可重复
        # 同一session内: reset_counter递增,每次rollout不同
        # 不同session间: reset_counter重置为0,相同seed产生相同序列
        reset_counter = [0]
        
        def _reset_to_initial_state_with_randomization(seed=None, env_ids=None):
            """
            Reset to initial state, but with randomization for specified objects.
            """
            reset_physx(env)
            # 使用base_seed作为后备,确保seed永远不是None
            if seed is None:
                seed = env.cfg.seed if hasattr(env.cfg, 'seed') and env.cfg.seed is not None else base_seed
            
            # 深拷贝initial_state以避免修改原始数据
            state_to_use = copy.deepcopy(initial_state)
            
            # 先完整reset到initial_state
            result = env.reset_to(state_to_use, env_ids, seed=seed, is_relative=is_relative)
            
            # 如果指定了需要随机化的物体,在reset后修改它们的位置
            if randomize_objects and len(randomize_objects) > 0:
                # 设置随机种子以获得可重复但不同的随机化结果
                # seed + reset_counter 确保:
                #   - 同一session内递增: rollout 1, 2, 3... 使用不同种子
                #   - 不同session间一致: 相同的base seed产生相同的序列
                reset_counter[0] += 1
                current_seed = seed + reset_counter[0]
                torch.manual_seed(current_seed)
                
                # 打印随机化信息
                if reset_counter[0] <= 3:  # 只打印前3次,避免刷屏
                    print(f"\n[RESET #{reset_counter[0]}] Seed: {current_seed} (base={seed} + counter={reset_counter[0]})")
                elif reset_counter[0] == 4:
                    print(f"\n[INFO] Subsequent resets will use seed={seed}+4, {seed}+5, ... (output suppressed)")
                
                # 收集需要随机化的物体及其原始位置
                objects_to_randomize = []
                
                for obj_name in randomize_objects:
                    # 检查物体在哪个类别中 (articulation 或 rigid_object)
                    if 'rigid_object' in state_to_use and obj_name in state_to_use['rigid_object']:
                        original_pose = state_to_use['rigid_object'][obj_name]['root_pose'].clone()
                        objects_to_randomize.append((obj_name, 'rigid_object', original_pose))
                    elif 'articulation' in state_to_use and obj_name in state_to_use['articulation']:
                        original_pose = state_to_use['articulation'][obj_name]['root_pose'].clone()
                        objects_to_randomize.append((obj_name, 'articulation', original_pose))
                    else:
                        print(f"[WARNING] Object '{obj_name}' not found in initial_state. Available objects:")
                        if 'rigid_object' in state_to_use:
                            print(f"  rigid_object: {list(state_to_use['rigid_object'].keys())}")
                        if 'articulation' in state_to_use:
                            print(f"  articulation: {list(state_to_use['articulation'].keys())}")
                
                # 对需要随机化的物体添加offset
                for obj_name, obj_type, original_pose in objects_to_randomize:
                    # 解析原始pose (format: [x, y, z, qw, qx, qy, qz])
                    original_pos = original_pose[0, :3]
                    original_quat = original_pose[0, 3:]  # wxyz format
                    
                    # 添加位置随机offset
                    pos_offset = torch.rand(3, device=env.device) * 2 * randomize_pos_offset - randomize_pos_offset
                    new_pos = original_pos + pos_offset
                    
                    # 添加旋转随机offset (绕z轴)
                    rot_offset = (torch.rand(1, device=env.device) * 2 - 1) * randomize_rot_offset
                    # 将原始四元数转换为欧拉角
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_quat(original_quat[[1,2,3,0]].cpu().numpy())  # wxyz -> xyzw for scipy
                    euler = r.as_euler('xyz')
                    euler[2] += rot_offset.cpu().numpy()[0]  # 只修改yaw
                    new_quat_xyzw = R.from_euler('xyz', euler).as_quat()
                    new_quat = torch.tensor([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]], 
                                           device=env.device)  # xyzw -> wxyz
                    
                    # 组合新的pose (确保dtype与原始pose一致)
                    new_pose = torch.cat([new_pos, new_quat]).unsqueeze(0).to(dtype=original_pose.dtype)
                    
                    # 打印随机化信息(便于调试)
                    if reset_counter[0] <= 3:  # 只打印前3次
                        print(f"[INFO] Randomized object '{obj_name}':")
                        print(f"  Original pos: {original_pos.cpu().numpy()}")
                        print(f"  New pos: {new_pos.cpu().numpy()}")
                        print(f"  Position offset: {pos_offset.cpu().numpy()}")
                        print(f"  Rotation offset (deg): {rot_offset.cpu().item() * 180 / 3.14159:.2f}")
                    
                    # 写入simulation
                    if obj_type == 'rigid_object':
                        env.scene.rigid_objects[obj_name].write_root_pose_to_sim(new_pose, env_ids=env_ids)
                    else:
                        env.scene.articulations[obj_name].write_root_pose_to_sim(new_pose, env_ids=env_ids)
                
                # 随机化机器人位置 (如果启用)
                if randomize_robot and 'robot' in state_to_use.get('articulation', {}):
                    original_robot_pose = state_to_use['articulation']['robot']['root_pose'].clone()
                    original_robot_pos = original_robot_pose[0, :3]
                    original_robot_quat = original_robot_pose[0, 3:]  # wxyz format
                    
                    # 添加位置随机offset
                    robot_pos_offset = torch.rand(3, device=env.device) * 2 * randomize_robot_pos_offset - randomize_robot_pos_offset
                    new_robot_pos = original_robot_pos + robot_pos_offset
                    
                    # 添加旋转随机offset (只绕z轴)
                    robot_rot_offset = (torch.rand(1, device=env.device) * 2 - 1) * randomize_robot_rot_offset
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_quat(original_robot_quat[[1,2,3,0]].cpu().numpy())
                    euler = r.as_euler('xyz')
                    euler[2] += robot_rot_offset.cpu().numpy()[0]  # 只修改yaw
                    new_robot_quat_xyzw = R.from_euler('xyz', euler).as_quat()
                    new_robot_quat = torch.tensor([new_robot_quat_xyzw[3], new_robot_quat_xyzw[0], 
                                                   new_robot_quat_xyzw[1], new_robot_quat_xyzw[2]], 
                                                  device=env.device)
                    
                    # 组合新的pose
                    new_robot_pose = torch.cat([new_robot_pos, new_robot_quat]).unsqueeze(0).to(dtype=original_robot_pose.dtype)
                    
                    # 打印随机化信息
                    if reset_counter[0] <= 3:
                        print(f"[INFO] Randomized robot:")
                        print(f"  Original pos: {original_robot_pos.cpu().numpy()}")
                        print(f"  New pos: {new_robot_pos.cpu().numpy()}")
                        print(f"  Position offset: {robot_pos_offset.cpu().numpy()}")
                        print(f"  Rotation offset (deg): {robot_rot_offset.cpu().item() * 180 / 3.14159:.2f}")
                    
                    # 写入simulation
                    env.scene.articulations['robot'].write_root_pose_to_sim(new_robot_pose, env_ids=env_ids)
                
                # 更新物理引擎
                env.sim.forward()
                if env.sim.has_rtx_sensors() and env.cfg.rerender_on_reset:
                    env.sim.render()
                
                # 重新计算observations
                obs_buf = env.observation_manager.compute(update_history=True)
                return obs_buf, env.extras
            else:
                # 如果没有指定随机化物体,直接返回reset结果
                return result

        print("resetting environment to initial state")
        # 初始化reset不消耗counter，确保第一个rollout从seed+1开始
        _reset_to_initial_state_with_randomization(env_ids=torch.tensor([0], device=env.device))
        # 重置counter，因为初始化reset不应该算作rollout
        if randomize_objects and len(randomize_objects) > 0:
            reset_counter[0] = 0
            print("[INFO] Environment initialized. Next reset will be rollout #1 with seed={base_seed}+1")

        # Override reset to keep the scene with controlled randomization
        def _reset_override(seed=None, env_ids=None, options=None):
            return _reset_to_initial_state_with_randomization(seed=seed, env_ids=env_ids)

        env.reset = _reset_override
    return env


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    """Rest everything follows."""
    with RemoteEnvWrapper(env_initializer=partial(make_env, launcher_args=app_launcher_args)) as env_server:
        env_server.serve()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
