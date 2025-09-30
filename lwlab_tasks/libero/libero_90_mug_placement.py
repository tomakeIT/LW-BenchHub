import copy
import re
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np
import torch


class LiberoMugPlacementBase(LiberoEnvCfg, BaseTaskEnvCfg):
    """
    LiberoMugPlacementBase: base class for all libero mug placement tasks
    """

    task_name: str = "LiberoMugPlacementBase"
    # EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _reset_internal(self, env_ids):
        super()._reset_internal(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos=(0.0, -1), size=(0.5, 0.5)):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                margin=0.02,
                ensure_valid_placement=True,
            )

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None):
            if mjcf_path is not None:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        info=dict(mjcf_path=mjcf_path),
                        placement=placement,
                    )
                )
            else:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        placement=placement,
                    )
                )

        return cfgs


class L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug(LiberoMugPlacementBase):
    """
    L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug: put the yellow and white mug to the front of the white mug

    Steps:
        pick up the yellow and white mug
        put the yellow and white mug to the front of the white mug

    """

    task_name: str = "L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug"
    # EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS
    enable_fixtures = ['microwave']

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.white_yellow_mug = "white_yellow_mug"
        self.porcelain_mug = "porcelain_mug"
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))

    def _setup_scene(self, env_ids=None):
        super()._setup_scene(env_ids)
        self.microwave.set_joint_state(0.9, 1.0, self.env, self.microwave.door_joint_names)

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos=(0.0, -1), size=(0.5, 0.5), rotation=None):
            cfg = dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                margin=0.02,
                ensure_valid_placement=True,
            )
            if rotation:
                cfg["rotation"] = rotation
            return cfg

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None):
            if mjcf_path is not None:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        info=dict(mjcf_path=mjcf_path),
                        placement=placement,
                    )
                )
            else:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        placement=placement,
                    )
                )

        # 白色马克杯放在后面，黄白色马克杯放在前面
        white_mug_placement = get_placement(pos=(0.2, -0.6), size=(0.3, 0.3))
        yellow_white_mug_placement = get_placement(pos=(-0.2, -0.2), size=(0.3, 0.3), rotation=-np.pi / 2.0)

        add_cfg(self.porcelain_mug, "cup", False, white_mug_placement,
                mjcf_path="/objects/lightwheel/cup/Cup012/model.xml")
        add_cfg(self.white_yellow_mug, "cup", True, yellow_white_mug_placement,
                mjcf_path="/objects/lightwheel/cup/Cup014/model.xml")

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the yellow and white mug to the front of the white mug."
        return ep_meta

    def _check_success(self):
        # 检查黄白色马克杯是否在白色马克杯前面
        return OU.check_place_obj1_side_by_obj2(
            self.env,
            self.white_yellow_mug,
            self.porcelain_mug,
            {
                "gripper_far": True,
                "contact": False,
                "side": "front",
                "side_threshold": 0.7,
                "margin_threshold": [0.001, 0.2],
                "stable_threshold": 0.5,
            }
        )


class L90K6CloseTheMicrowave(L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug):
    task_name: str = "L90K6CloseTheMicrowave"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the microwave."
        return ep_meta

    def _check_success(self):
        return self.microwave.is_closed(self.env) & OU.gripper_obj_far(self.env, self.microwave.name, th=0.4)


class L10K6PutTheYellowAndWhiteMugInTheMicrowaveAndCloseIt(L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug):
    task_name: str = "L10K6PutTheYellowAndWhiteMugInTheMicrowaveAndCloseIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the yellow and white mug in the microwave and close it."
        return ep_meta

    def _check_success(self):
        mug_pos = OU.get_object_pos(self.env, self.white_yellow_mug)
        mug_success = OU.point_in_fixture(mug_pos, self.microwave)
        mug_success = torch.tensor(mug_success, dtype=torch.bool, device="cpu").repeat(self.env.num_envs)
        return mug_success & self.microwave.is_closed(self.env) & OU.gripper_obj_far(self.env, self.microwave.name, th=0.4)


class L90L5PutTheRedMugOnTheLeftPlate(LiberoMugPlacementBase):
    """
    L90L5PutTheRedMugOnTheLeftPlate: put the red mug on the left plate

    Steps:
        pick up the red mug
        put the red mug on the left plate

    """

    task_name: str = "L90L5PutTheRedMugOnTheLeftPlate"
    # EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.red_coffee_mug = "red_coffee_mug"
        self.plate_left = "plate_left"
        self.plate_right = "plate_right"
        self.porcelain_mug = "porcelain_mug"
        self.white_yellow_mug = "white_yellow_mug"

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos=(0.0, -1), size=(0.5, 0.5)):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                margin=0.02,
                ensure_valid_placement=True,
            )

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None, init_robot_here=False):
            if mjcf_path is not None:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        info=dict(mjcf_path=mjcf_path),
                        placement=placement,
                        **({"init_robot_here": True} if init_robot_here else {}),
                    )
                )
            else:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        placement=placement,
                        **({"init_robot_here": True} if init_robot_here else {}),
                    )
                )

        # 左边盘子，右边盘子，红色马克杯放在右边，其他马克杯作为干扰物
        plate_left_placement = get_placement(pos=(-0.6, -0.4), size=(0.35, 0.35))
        plate_right_placement = get_placement(pos=(0.6, -0.4), size=(0.35, 0.35))
        red_mug_placement = get_placement(pos=(0.0, -1.2), size=(0.3, 0.3))
        white_yellow_mug_placement = get_placement(pos=(-0.3, -1), size=(0.3, 0.3))
        porcelain_mug_placement = get_placement(pos=(0.3, -1), size=(0.3, 0.3))

        add_cfg(self.plate_left, "plate", False, plate_left_placement,
                mjcf_path="/objects/lightwheel/plate/Plate012/model.xml", init_robot_here=True)
        add_cfg(self.plate_right, "plate", False, plate_right_placement,
                mjcf_path="/objects/lightwheel/plate/Plate012/model.xml")
        add_cfg(self.red_coffee_mug, "cup", True, red_mug_placement,
                mjcf_path="/objects/lightwheel/cup/Cup030/model.xml")
        add_cfg(self.porcelain_mug, "cup", True, porcelain_mug_placement,
                mjcf_path="/objects/lightwheel/cup/Cup012/model.xml")
        add_cfg(self.white_yellow_mug, "cup", True, white_yellow_mug_placement,
                mjcf_path="/objects/lightwheel/cup/Cup014/model.xml")

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the red mug on the left plate."
        return ep_meta

    def _check_success(self):
        # 检查红色马克杯是否在左边盘子上
        success = OU.check_place_obj1_on_obj2(
            self.env,
            'red_coffee_mug',
            'plate_left'
        )
        return success
