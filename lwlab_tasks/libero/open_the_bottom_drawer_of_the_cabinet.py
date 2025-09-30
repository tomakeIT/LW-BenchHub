import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90K1OpenTheBottomDrawerOfTheCabinet(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = "L90K1OpenTheBottomDrawerOfTheCabinet"
    enable_fixtures = ["storage_furniture"]

    def __post_init__(self):
        self.obj_name = []
        super().__post_init__()
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE, ref=self.dining_table))

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )

        self.init_robot_base_ref = self.dining_table

    def _load_model(self):
        super()._load_model()
        for cfg in self.object_cfgs:
            self.obj_name.append(cfg["name"])

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)
        self.botton_joint_name = list(self.drawer._joint_infos.keys())[-1]
        self.top_joint_name = list(self.drawer._joint_infos.keys())[0]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"put the black bowl on the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"akita_black_bowl",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml"
                ),
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.5),
                    margin=0.02,
                    pos=(0.1, -0.3),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"plate",
                obj_groups=["plate"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/plate/Plate012/model.xml"
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(0.2, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self):
        return self.drawer.is_open(self.env, [self.botton_joint_name], th=0.5) & OU.gripper_obj_far(self.env, self.drawer.name, th=0.5)


class L90K1OpenTheTopDrawerOfTheCabinet(L90K1OpenTheBottomDrawerOfTheCabinet):
    task_name: str = "L90K1OpenTheTopDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"open the top drawer of the cabinet."
        return ep_meta

    def _check_success(self):
        return self.drawer.is_open(self.env, [self.top_joint_name], th=0.5) & OU.gripper_obj_far(self.env, self.drawer.name)
