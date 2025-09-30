import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L10K8PutBothMokaPotsOnTheStove(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = 'L10K8PutBothMokaPotsOnTheStove'
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
        ] = f"Put both moka pots on the stove."
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.mokapot_1 = self.register_fixture_ref("mokapot_1", dict(id="mokapot_1"))
        self.mokapot_2 = self.register_fixture_ref("mokapot_2", dict(id="mokapot_2"))
        self.init_robot_base_ref = self.dining_table

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _check_success(self):
        '''
        Check if the moka pot is placed on the plate.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(self.env, self.mokapot_1.name) & OU.gripper_obj_far(self.env, self.mokapot_2.name)
        pot_on_stove = OU.check_place_obj1_on_obj2(self.env, self.mokapot_1, self.stove, th_xy_dist=0.5) & OU.check_place_obj1_on_obj2(self.env, self.mokapot_2, self.stove, th_xy_dist=0.5)
        return is_gripper_obj_far & pot_on_stove
