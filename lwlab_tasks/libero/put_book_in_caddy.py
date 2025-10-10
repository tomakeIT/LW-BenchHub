import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import copy
from isaaclab.utils.math import matrix_from_quat
from time import time

import numpy as np


class L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = "L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy"

    # counter_id: FixtureType = FixtureType.TABLE

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the book and place it in the right compartment of the caddy."
        return ep_meta

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _reset_internal(self, env_ids):
        super()._reset_internal(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        placement = dict(
            fixture=self.counter,
            size=(0.8, 0.4),
            pos=(0.0, -0.6),
            offset=(-0.05, 0),
            ensure_valid_placement=True,
        )
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
                placement=placement,
            )
        )
        cfgs.append(
            dict(
                name="black_book",
                object_scale=0.4,
                obj_groups="book",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/book/Book042/model.xml",
                ),
                placement=dict(
                    fixture=self.counter,
                    size=(0.4, 0.3),
                    pos=(0.2, -0.9),
                    offset=(-0.05, 0),
                    ensure_valid_placement=True,
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
                placement=placement,
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
                placement=placement,
            )
        )
        return cfgs

    def _check_success(self):
        book_success = OU.check_obj_in_receptacle(self.env, "black_book", "desk_caddy")
        gipper_far_success = OU.gripper_obj_far(self.env, "black_book", 0.35)
        return book_success & gipper_far_success


class L90S2PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy(L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy):

    task_name: str = "L90S2PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy"

    def __post_init__(self):
        self.init_dish_rack_pos = None
        self.obj_name = []
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref(
            "table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
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
                object_scale=0.4,
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/book/Book042/model.xml",
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.3, -0.8)
                ),
            )
        )
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
                    size=(0.8, 0.45),
                    margin=0.02,
                    pos=(0.6, -0.5)
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
                    pos=(-0.1, -0.1)
                ),
            )
        )
        return cfgs


class L90S3PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy(L90S2PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy):

    task_name: str = "L90S3PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the book and place it in the left compartment of the caddy."
        return ep_meta


# --- Libero90 variants consolidated from libero_90_put_black_book_in_caddy_compartments.py ---

class _BaseBookInCaddy(LiberoEnvCfg, BaseTaskEnvCfg):
    task_name: str = "_BaseBookInCaddy"

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter
        self.black_book = "black_book"
        self.desk_caddy = "desk_caddy"
        self.red_coffee_mug = "red_coffee_mug"

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos, size, rotation=None):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                rotation=rotation,
                margin=0.02,
                ensure_valid_placement=True,
            )

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None, init_robot_here=False, object_scale=None):
            cfg = dict(name=name, obj_groups=obj_groups, graspable=graspable, placement=placement)
            if mjcf_path is not None:
                # TODO: asset missing locally, replace mjcf_path when available [[Book042 / DeskCaddy001 / Cup030]]
                cfg["info"] = dict(mjcf_path=mjcf_path)
            if init_robot_here:
                cfg["init_robot_here"] = True
            if object_scale is not None:
                cfg["object_scale"] = object_scale
            cfgs.append(cfg)

        # Initial layout: caddy back-right, book front-left, mug as distractor front-right caddy_pl = get_placement((0, -0.5), (0.5, 0.6), np.pi / 8)
        caddy_pl = get_placement((0, -0.4), (0.6, 0.4), np.pi / 8)
        book_pl = get_placement((0.2, -0.5), (0.35, 0.35), 0)
        mug_pl = get_placement((-0.3, -0.6), (0.35, 0.3), 0)

        add_cfg(self.desk_caddy, "desk_caddy", True, caddy_pl, mjcf_path="/objects/lightwheel/desk_caddy/DeskCaddy001/model.xml", object_scale=2.0)
        add_cfg(self.black_book, "book", True, book_pl, mjcf_path="/objects/lightwheel/book/Book042/model.xml", object_scale=0.4)
        add_cfg(self.red_coffee_mug, "cup", True, mug_pl, mjcf_path="/objects/lightwheel/cup/Cup030/model.xml")

        return cfgs

    def _success_common(self):
        in_caddy = OU.check_obj_in_receptacle(self.env, self.black_book, self.desk_caddy)
        gripper_far_success = OU.gripper_obj_far(self.env, self.black_book, 0.35)
        return in_caddy & gripper_far_success


class L90S2PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy(_BaseBookInCaddy):
    task_name: str = "L90S2PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the book and place it in the left compartment of the caddy."
        return ep_meta

    def _check_success(self):
        # TODO: add compartment-level check (left half of caddy) when utility is available
        return self._success_common()


class L90S2PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy(_BaseBookInCaddy):
    task_name: str = "L90S2PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the book and place it in the back compartment of the caddy."
        return ep_meta

    def _check_success(self):
        # TODO: add compartment-level check (back half of caddy) when utility is available
        return self._success_common()


class L10S1PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy(L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy):

    task_name: str = "L10S1PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy"

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
                    fixture=self.counter,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.3, -0.5)
                ),
            )
        )
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
                    fixture=self.counter,
                    size=(0.8, 0.45),
                    margin=0.02,
                    pos=(0.3, -0.5)
                ),
            )
        )
        cfgs.append(
            dict(
                name="white_yellow_mug",
                obj_groups="cup",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/cup/Cup014/model.xml",
                ),
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )
        return cfgs


# --- Study Scene 4 Book Tasks ---

class _BaseStudyScene4(LiberoEnvCfg, BaseTaskEnvCfg):
    """Base class for Study Scene 4 tasks with books and shelves"""
    task_name: str = "_BaseStudyScene4"

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter
        self.black_book = "black_book"
        self.yellow_book = "yellow_book"
        self.middle_book = "midlle_book"
        self.shelf = "shelf"

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos, size, rotation=None):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                rotation=rotation,
                margin=0.02,
                ensure_valid_placement=True,
            )

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None, init_robot_here=False, object_scale=None):
            cfg = dict(name=name, obj_groups=obj_groups, graspable=graspable, placement=placement)
            if mjcf_path is not None:
                cfg["info"] = dict(mjcf_path=mjcf_path)
            if init_robot_here:
                cfg["init_robot_here"] = True
            if object_scale is not None:
                cfg["object_scale"] = object_scale
            cfgs.append(cfg)

        # Books placed symmetrically on left and right sides
        left_book_pl = get_placement((-0.35, 0.8), (0.35, 0.35))
        right_book_pl = get_placement((0.35, -0.6), (0.35, 0.35))
        middle_book_pl = get_placement((0, -0.6), (0.35, 0.35))
        shelf_pl = get_placement((0.7, -0.5), (0.5, 0.75), rotation=-np.pi / 2)

        add_cfg(self.shelf, "shelf", True, shelf_pl, mjcf_path="/objects/lightwheel/shelf/Shelf073/model.xml", object_scale=1.0)
        add_cfg(self.black_book, "book", True, left_book_pl, mjcf_path="/objects/lightwheel/book/Book042/model.xml", object_scale=0.4)
        add_cfg(self.yellow_book, "book", True, right_book_pl, mjcf_path="/objects/lightwheel/book/Book043/model.xml", object_scale=0.4)
        add_cfg(self.middle_book, "book", True, middle_book_pl, mjcf_path="/objects/lightwheel/book/Book043/model.xml", object_scale=0.4)

        return cfgs


class L90S4PickUpTheBookOnTheLeftAndPlaceItOnTopOfTheShelf(_BaseStudyScene4):
    task_name: str = "L90S4PickUpTheBookOnTheLeftAndPlaceItOnTopOfTheShelf"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the book on the left and place it on top of the shelf."
        return ep_meta

    def _check_success(self):
        # Check if black book (left book) is placed on top of the desk_caddy (shelf)
        book_on_shelf_result = OU.check_place_obj1_on_obj2(
            self.env,
            self.black_book,
            self.shelf,
            th_z_axis_cos=0.0,  # 不检查Z轴角度
            th_xy_dist=0.25,    # 保持XY距离检查
            th_xyz_vel=0.5      # 保持稳定性检查
        )
        return book_on_shelf_result


class L90S4PickUpTheBookOnTheRightAndPlaceItOnTheCabinetShelf(_BaseStudyScene4):
    task_name: str = "L90S4PickUpTheBookOnTheRightAndPlaceItOnTheCabinetShelf"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the book on the right and place it on the cabinet shelf."
        return ep_meta

    def _check_success(self):
        # Check if yellow book (right book) is placed on top of the desk_caddy (cabinet shelf)
        book_on_shelf_result = OU.check_place_obj1_on_obj2(
            self.env,
            self.yellow_book,
            self.shelf,
            th_z_axis_cos=0.0,  # 不检查Z轴角度
            th_xy_dist=0.25,    # 保持XY距离检查
            th_xyz_vel=0.5      # 保持稳定性检查
        )
        return book_on_shelf_result
