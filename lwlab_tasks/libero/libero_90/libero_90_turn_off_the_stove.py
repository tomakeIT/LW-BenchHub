import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class L90K8TurnOffTheStove(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = 'L90K8TurnOffTheStove'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["stovetop", "mokapot_1", "mokapot_2"]
    removable_fixtures: list[str] = ["mokapot_1", "mokapot_2"]

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Turn off the stove."
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.mokapot_1 = self.register_fixture_ref("mokapot_1", dict(id=FixtureType.MOKA_POT))
        self.mokapot_2 = self.register_fixture_ref("mokapot_2", dict(id=FixtureType.MOKA_POT))
        self.init_robot_base_ref = self.dining_table

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)
        self.stove.set_knob_state(self.env, knob="center", mode="on")

    def _check_success(self):
        knob_success = torch.tensor([False], device=self.env.device).repeat(self.env.num_envs)
        knobs_state = self.stove.get_knobs_state(env=self.env)
        for knob_name, knob_value in knobs_state.items():
            abs_knob = torch.abs(knob_value)
            lower, upper = 0.35, 2 * np.pi - 0.35
            knob_off = (abs_knob < lower)
            knob_success = knob_success | knob_off
        return knob_success & OU.gripper_obj_far(self.env, self.stove.name, 0.3)
