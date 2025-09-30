import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90K2StackTheBlackBowlAtTheFrontOnTheBlackBowlInTheMiddle(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = "L90K2StackTheBlackBowlAtTheFrontOnTheBlackBowlInTheMiddle"
    enable_fixtures = ["storage_furniture"]

    def __post_init__(self):
        self.obj_name = []
        super().__post_init__()
        self.drawer = self.register_fixture_ref("singlecabinet", dict(id=FixtureType.STORAGE_FURNITURE))

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.6)),
        )

        self.init_robot_base_ref = self.dining_table

    def _load_model(self):
        super()._load_model()
        for cfg in self.object_cfgs:
            self.obj_name.append(cfg["name"])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"stack the black bowl at the front on the black bowl in the middle."
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
                    size=(0.50, 0.35),
                    margin=0.02,
                    pos=(0.0, -0.6),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"akita_black_bowl_front",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml"
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.35),
                    margin=0.02,
                    pos=(0.0, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"akita_black_bowl_back",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml"
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.35),
                    margin=0.02,
                    pos=(-0.1, -0.3),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"plate",
                obj_groups=["plate"],
                graspable=True,
                washable=True,
                object_scale=0.6,
                info=dict(
                    mjcf_path="/objects/lightwheel/plate/Plate012/model.xml"
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.6, 0.35),
                    margin=0.02,
                    pos=(-0.3, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self):
        ret = OU.check_place_obj1_on_obj2(self.env, "akita_black_bowl_front", "akita_black_bowl")
        ret1 = OU.check_place_obj1_on_obj2(self.env, "akita_black_bowl_front", "akita_black_bowl_back")
        ret2 = OU.check_place_obj1_on_obj2(self.env, "akita_black_bowl_back", "akita_black_bowl_front")
        ret3 = OU.check_place_obj1_on_obj2(self.env, "akita_black_bowl_back", "akita_black_bowl")
        ret4 = OU.check_place_obj1_on_obj2(self.env, "akita_black_bowl", "akita_black_bowl_front")
        ret5 = OU.check_place_obj1_on_obj2(self.env, "akita_black_bowl", "akita_black_bowl_back")
        return ret | ret1 | ret2 | ret3 | ret4 | ret5
