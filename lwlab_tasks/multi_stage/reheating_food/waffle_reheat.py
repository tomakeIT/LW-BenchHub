import numpy as np
from lwlab.core.tasks.base import LwLabTaskBase
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg, RigidObjectCollectionCfg
# @configclass
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class WaffleReheat(LwLabTaskBase):
    """
    Waffle Reheat: composite task for Reheating Food activity.

    Simulates the task of reheating a waffle.

    Steps:
        Open the microwave. Place the bowl with waffle inside the microwave, then
        close the microwave door and turn it on.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.MICROWAVE]
    task_name: str = "WaffleReheat"
    # exclude layout 8 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [8]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_ref = self.microwave

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Open the microwave, place the bowl with waffle inside the microwave, "
            "then close the microwave door and turn it on."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="waffle",
                obj_groups="waffle",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                    try_to_place_in="bowl",
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        gripper_far = OU.gripper_obj_far(env, "waffle")
        waffle_in_bowl = OU.check_obj_in_receptacle(env, "waffle", "waffle_container")
        bowl_in_microwave = OU.obj_inside_of(env, "waffle_container", self.microwave)
        microwave_on = self.microwave.get_state()["turned_on"]
        return waffle_in_bowl & bowl_in_microwave & microwave_on & gripper_far
