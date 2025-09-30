import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class L90K9PutTheFryingPanOnTopOfTheCabinet(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = 'L90K9PutTheFryingPanOnTopOfTheCabinet'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["stovetop"]

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"put the frying pan on top of the cabinet."
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.dining_table
        self.shelf = "shelf"
        self.frying_pan = "frying_pan"
        self.bowl = "bowl"

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.shelf,
                obj_groups="shelf",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.75, 0.25),
                    rotation=np.pi / 2,
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/shelf/Shelf073/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(1.0, -0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl009/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.frying_pan,
                obj_groups="pot",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.5, -0.25),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/pot/Pot086/model.xml",
                ),
            )
        )

        return cfgs

    def _check_success(self):

        success_state = OU.check_place_obj1_on_obj2(self.env, self.frying_pan, self.shelf)
        return success_state & OU.gripper_obj_far(self.env, self.frying_pan, th=0.4)
