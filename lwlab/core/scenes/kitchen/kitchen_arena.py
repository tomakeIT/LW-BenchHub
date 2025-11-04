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

# base class for kitchen arena

import os
import time

from lightwheel_sdk.loader import floorplan_loader
from lwlab.utils.usd_utils import OpenUsd as usd
from lwlab.core.models.scenes.scene_parser import parse_fixtures
from lwlab.core.models.fixtures.fixture_types import FixtureType, FIXTURE_TYPE_TO_REGISTRY_NAME


class KitchenArena:
    """
    Kitchen arena class holding all of the fixtures

    Args:
        layout_id (int or LayoutType): layout of the kitchen to load

        style_id (int or StyleType): style of the kitchen to load

        scene_cfg (RoboCasaSceneCfg): scene configuration

        layout_registry_names (list, optional): List of fixture registry names (FixtureType, int, or str) required for layout filtering.

    """

    def __init__(self, layout_id, style_id, exclude_layouts=[], enable_fixtures=[], movable_fixtures=[], scene_cfg=None, scene_type='robocasakitchen', layout_registry_names: list[str | FixtureType | int] | None = None,):
        # download floorplan usd
        self.scene_cfg = scene_cfg
        self.floorplan_version = scene_cfg.floorplan_version
        self.enable_fixtures = enable_fixtures
        self.movable_fixtures = movable_fixtures
        self.load_floorplan(layout_id, style_id, exclude_layouts=exclude_layouts, layout_registry_names=layout_registry_names)
        self.stage = usd.get_stage(self.usd_path)

        # enable fixtures in usd
        if self._is_updated_usd():
            dir_name = os.path.dirname(self.usd_path)
            base_name = os.path.basename(self.usd_path)
            new_path = os.path.join(dir_name, base_name.replace(".usd", "_enabled.usd"))
            self.stage.GetRootLayer().Export(new_path)
            self.usd_path = new_path
        # load fixtures
        self.scene_cfg.fixtures = parse_fixtures(self.stage, scene_cfg.context.num_envs, scene_cfg.context.seed, scene_cfg.context.device)

    def _is_updated_usd(self):
        is_updated_usd = False
        if self.enable_fixtures is not None:
            is_updated_usd = True
            for fixture in self.enable_fixtures:
                usd.activate_prim(self.stage, fixture)
        if self.movable_fixtures is not None:
            is_updated_usd = True
            root_prim = self.stage.GetPseudoRoot()
            for fixture in self.movable_fixtures:
                prims = usd.get_prim_by_prefix(root_prim, fixture)
                for prim in prims:
                    fixed_joints = usd.get_prim_by_type(prim, include_types=["PhysicsFixedJoint"])
                    for fix_joint_prim in fixed_joints:
                        fix_joint_prim.SetActive(False)
        return is_updated_usd

    def get_fixture_cfgs(self):
        """
        Returns config data for all fixtures in the arena

        Returns:
            list: list of fixture configurations
        """
        fixture_cfgs = []
        for (name, fxtr) in self.scene_cfg.fixtures.items():
            cfg = {}
            cfg["name"] = name
            cfg["model"] = fxtr
            cfg["type"] = "fixture"
            if hasattr(fxtr, "_placement"):
                cfg["placement"] = fxtr._placement

            fixture_cfgs.append(cfg)

        return fixture_cfgs

    def load_floorplan(self, layout_id, style_id, exclude_layouts=[], layout_registry_names: list[str | FixtureType | int] | None = None):
        start_time = time.time()

        # Convert FixtureType to registry name strings
        if layout_registry_names is not None:
            registry_name_strings = []
            for item in layout_registry_names:
                if isinstance(item, FixtureType) or isinstance(item, int):
                    # Convert FixtureType enum to string
                    if item in FIXTURE_TYPE_TO_REGISTRY_NAME:
                        registry_name_strings.append(FIXTURE_TYPE_TO_REGISTRY_NAME[item])
                    else:
                        print(f"Warning: FixtureType {item} not found in mapping, skipping")
                elif isinstance(item, str):
                    # Already a string, keep as is
                    registry_name_strings.append(item)
                else:
                    print(f"Warning: Unknown type {type(item)} for layout_registry_names item {item}, skipping")
            # Remove duplicates while preserving order
            layout_registry_names = list(dict.fromkeys(registry_name_strings))
            print(f"Converted layout_registry_names: {layout_registry_names}")

        print(f"load floorplan usd", end="...")
        if layout_id is None:
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_type, version=self.floorplan_version, exclude_layout_ids=exclude_layouts, layout_registry_names=layout_registry_names)
        elif style_id is None:
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_type, layout_id=layout_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts, layout_registry_names=layout_registry_names)
        else:
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_type, layout_id=layout_id, style_id=style_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts, layout_registry_names=layout_registry_names)
        usd_path, self.floorplan_meta = res.result()
        self.usd_path = str(usd_path)
        self.backend = self.floorplan_meta.get("backend")
        self.scene_type = self.floorplan_meta.get("scene")
        self.layout_id = self.floorplan_meta.get("layout_id")
        self.style_id = self.floorplan_meta.get("style_id")
        self.version_id = self.floorplan_meta.get("version_id")
        print(f"{self.backend}-{self.scene_type}-{self.layout_id}-{self.style_id} loaded in {time.time() - start_time:.2f}s\n")
        print(f"version_id: {self.version_id}")
