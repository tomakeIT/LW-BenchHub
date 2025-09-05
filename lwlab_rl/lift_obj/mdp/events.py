import torch
import math

from typing import Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.envs import ManagerBasedRLEnv

try:
    from pxr import Gf
except ImportError:
    # Fallback if pxr is not available
    Gf = None


def randomize_camera_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation uniformly within the given ranges.

    * It samples the camera position and rotation from the given ranges and adds them to the
      default camera position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    asset: Camera = env.scene[asset_cfg.name]

    ori_pos_w = asset.data.pos_w
    if convention == "ros":
        ori_quat_w = asset.data.quat_w_ros
    elif convention == "opengl":
        ori_quat_w = asset.data.quat_w_opengl
    elif convention == "world":
        ori_quat_w = asset.data.quat_w_world

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = ori_pos_w[:, 0:3] + rand_samples[:, 0:3]  # camera usually spawn with robot, so no need to add env_origins
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)


def randomize_scene_lighting(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (400.0, 1200.0),
    color_variation: float = 0.1,
    default_intensity: float = 800.0,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):

    from isaaclab.assets import AssetBase
    import random

    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    color_attr = light_prim.GetAttribute("inputs:color")

    intensity_attr.Set(default_intensity)
    color_attr.Set(default_color)

    new_intensity = random.uniform(intensity_range[0], intensity_range[1])
    intensity_attr.Set(new_intensity)

    new_color = _sample_random_color(base=default_color, variation=color_variation)
    color_attr.Set(new_color)


def _sample_random_color(base=(0.75, 0.75, 0.75), variation=0.1):

    import random

    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    avg_offset = sum(offsets) / 3
    balanced_offsets = [offset - avg_offset for offset in offsets]
    new_color = tuple(max(0, min(1, base_component + offset))
                      for base_component, offset in zip(base, balanced_offsets))

    return new_color
