import torch
import numpy as np
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg
from dataclasses import MISSING
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import lwlab.core.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from lwlab.core.models.fixtures import Counter


class OpenBlenderLid(BaseTaskEnvCfg, RobocasaKitchenEnvCfg):
    """
    Class encapsulating the atomic open blender lid task.
    """

    task_name: str = "OpenBlenderLid"
    enable_fixtures: list[str] = ["blender"]
    EXCLUDE_LAYOUTS = [49]

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the open blender lid task.
        """
        super()._setup_kitchen_references()
        self.blender = self.get_fixture(FixtureType.BLENDER)
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER))
        self.init_robot_base_ref = self.blender

    def get_ep_meta(self):

        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the blender by taking off the lid and placing it on the counter."
        return ep_meta

    def _setup_scene(self, env_ids=None):
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="blender1",
                asset_name="Blender012.usd",
                merged_obj=True,
                is_fixture=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.65, 0.65),
                ),
                # auxiliary_obj_placement=dict(
                #     fixture=self.counter,
                #     size=(0.4, 0.4),
                #     pos=(0, -1.0),
                # ),
            )
        )

        # cfgs.append(
        #     dict(
        #         name="blender2",
        #         obj_groups="Blender012.usd",
        #         merged_obj=True,
        #         is_fixture=True,
        #         placement=dict(
        #             fixture=self.counter,
        #             size=(0.65, 0.65),
        #         ),
        #     )
        # )

        return cfgs

    def _check_success(self):
        # check lid contact with any counter
        gripper_lid_far = OU.gripper_obj_far(
            self.env, "blender1" + "_lid", th=0.15
        )
        lid_not_on_blender = self.fixture_refs['blender1'].get_state()['lid_not_on_blender']
        lid_counter_contact = OU.check_obj_any_counter_contact(self.env, self, 'blender1_lid')
        return gripper_lid_far and lid_not_on_blender and lid_counter_contact


class CloseBlenderLid(BaseTaskEnvCfg, RobocasaKitchenEnvCfg):
    """
    Class encapsulating the atomic close blender lid task.
    """

    task_name: str = "CloseBlenderLid"
    enable_fixtures: list[str] = ["blender"]
    EXCLUDE_LAYOUTS = [49]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.blender = self.get_fixture(FixtureType.BLENDER)
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER))
        self.init_robot_base_ref = self.blender

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Close the lid blender by securely placing the lid on top."
        return ep_meta

    def _setup_scene(self, env_ids=None):
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="blender1",
                obj_groups="Blender012.usd",
                merged_obj=True,
                is_fixture=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.65, 0.65),
                ),
                # auxiliary_obj_placement=dict(
                #     fixture=self.counter,
                #     size=(0.4, 0.4),
                #     pos=(0, -1.0),
                # ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the blender lid is closed.
        """
        lid_on_blender = self.fixtures['blender1'].get_state()["lid_on_blender"]
        gripper_lid_far = OU.gripper_obj_far(self.env, 'blender1' + "_lid", th=0.15)
        return lid_on_blender and gripper_lid_far


class TurnOnBlender(BaseTaskEnvCfg, RobocasaKitchenEnvCfg):
    """
    Class encapsulating the atomic turn on blender task.
    """

    task_name: str = "TurnOnBlender"
    enable_fixtures: list[str] = ["blender"]
    EXCLUDE_LAYOUTS = [49]

    # def __init__(self, enable_fixtures=None, *args, **kwargs):
    #     enable_fixtures = enable_fixtures or []
    #     enable_fixtures = list(enable_fixtures) + ["blender"]
    #     super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # self.blender = self.get_fixture(FixtureType.BLENDER)
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER))
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):

        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Turn on the blender by pressing the power button."
        return ep_meta

    def _setup_scene(self, env_ids=None):
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            # dict(
            #     name="obj",
            #     obj_groups=("fruit"),
            #     object_scale=0.80,
            #     placement=dict(
            #         fixture=self.blender,
            #         size=(0.40, 0.40),
            #         pos=(0, 0),
            #     ),
            # )

            dict(
                name="blender1",
                obj_groups="Blender012.usd",
                merged_obj=True,
                is_fixture=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.65, 0.65),
                ),
                # auxiliary_obj_placement=dict(
                #     fixture=self.counter,
                #     size=(0.4, 0.4),
                #     pos=(0, -1.0),
                # ),
            )
        )

        return cfgs

    def _check_success(self):
        return self.fixtures['blender1'].get_state()["turned_on"]
