import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class PubBookInCaddy(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = "PubBookInCaddy"

    def __post_init__(self):
        self.init_dish_rack_pos = None
        self.obj_name = []
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref(
            "table",
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
        ] = f"pick up the book and place it in the right compartment of the caddy."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="black_book",
                obj_groups="book",
                graspable=True,
                object_scale=0.4,
                info=dict(
                    mjcf_path="/objects/lightwheel/book/Book042/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )

        cfgs.append(
            dict(
                name="red_coffee_mug",
                obj_groups="cup",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/cup/Cup030/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self):
        # 打开柜子底部抽屉
        return False


class PubBookInCaddy1(PubBookInCaddy):

    task_name = "PubBookInCaddy1"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the red mug and place it to the right compartment of the caddy."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="black_book",
                obj_groups="book",
                object_scale=0.4,
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/book/Book042/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )

        cfgs.append(
            dict(
                name="red_coffee_mug",
                obj_groups="cup",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/cup/Cup030/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )
        cfgs.append(
            dict(
                name="porcelain_mug",
                obj_groups="cup",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/cup/Cup012/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )
        return cfgs


class L90S3PickUpTheRedMugAndPlaceItToTheRightOfTheCaddy(PubBookInCaddy):
    task_name = "L90S3PickUpTheRedMugAndPlaceItToTheRightOfTheCaddy"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the red mug and place it to the right compartment of the caddy."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="desk_caddy",
                obj_groups="desk_caddy",
                graspable=True,
                object_scale=2.0,
                init_robot_here=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/desk_caddy/DeskCaddy001/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.8, 0.5),
                    pos=(-0.2, -0.1),
                ),
            )
        )
        cfgs.append(
            dict(
                name="black_book",
                obj_groups="book",
                object_scale=0.4,
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/book/Book042/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.4, -0.5)
                ),
            )
        )

        cfgs.append(
            dict(
                name="red_coffee_mug",
                obj_groups="cup",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/cup/Cup030/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(0.3, -0.6)
                ),
            )
        )
        cfgs.append(
            dict(
                name="porcelain_mug",
                obj_groups="cup",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/cup/Cup012/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self):
        success = OU.check_place_obj1_side_by_obj2(
            self.env,
            "red_coffee_mug",
            "desk_caddy",
            {
                "gripper_far": True,
                "contact": False,
                "side": "right",
                "side_threshold": 1.5,
                "margin_threshold": [0.001, 0.2],
                "stable_threshold": 0.5,
            }
        )
        return success
