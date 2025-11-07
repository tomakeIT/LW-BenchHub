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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from lwlab.utils.isaaclab_utils.assets import GeneralAssetCfg
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg


@configclass
class USDSceneCfg(InteractiveSceneCfg):
    """Configuration for the kitchen scene with a robot and a kitchen.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """
    _usd_path: str = MISSING
    main_scene: GeneralAssetCfg = None  # will be populated at __post_init__

    def __post_init__(self):
        super().__post_init__()

        self.main_scene = GeneralAssetCfg(
            prim_path="{ENV_REGEX_NS}/Scene",
            # Make sure to set the correct path to the generated scene
            spawn=sim_utils.UsdFileCfg(usd_path=self._usd_path, activate_contact_sensors=False),
        )

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=9000.0),
    )


# monkey patch to add the _usd_path field to the InteractiveSceneCfg class,
# so that InteractiveSceneCfg._add_entities_from_cfg can ignore _usd_path
# InteractiveSceneCfg.__dataclass_fields__["_usd_path"] = USDSceneCfg.__dataclass_fields__["_usd_path"]

from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.assets.background import Background
from lwlab.core.context import get_context
from lwlab.utils.isaaclab_utils import NoDeepcopyMixin


class LocalScene(Scene, NoDeepcopyMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.context = get_context()
        self.scene_usd_path = self.context.scene_name
        self.scene_type = "local"
        self.fixtures = {}
        self.fxtr_placements = {}
        self.is_replay_mode = False
        assert self.scene_usd_path.endswith(".usd"), "Scene USD path must end with .usd"

    def setup_env_config(self, orchestrator):
        background = Background(
            name=self.scene_type,
            usd_path=self.scene_usd_path,
            object_min_z=0.1,
        )
        # flush self.assets
        self.assets = {}
        self.add_asset(background)

    def get_ep_meta(self):
        return {
            "floorplan_version": None,
        }

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg):
        env_cfg.sim.render.enable_translucency = True
        return env_cfg
