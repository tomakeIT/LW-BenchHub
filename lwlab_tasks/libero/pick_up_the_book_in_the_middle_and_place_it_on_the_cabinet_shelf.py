import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class L90S4PickUpTheBookInTheMiddleAndPlaceItOnTheCabinetShelf(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = "L90S4PickUpTheBookInTheMiddleAndPlaceItOnTheCabinetShelf"

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )
        self.init_robot_base_ref = self.table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "pick up the book in the middle and place it on the cabinet shelf."
        return ep_meta

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="wooden_two_layer_shelf",
                obj_groups="shelf",
                graspable=True,
                object_scale=1.2,
                placement=dict(
                    fixture=self.table,
                    size=(0.7, 0.7),
                    pos=(-0.4, 0.10),
                    rotation=np.pi / 2,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/shelf/Shelf073/model.xml",
                ),
            )
        )
        cfgs.append(
            dict(
                name="black_book",
                obj_groups="book",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/book/Book042/model.xml",
                ),
                object_scale=0.4,
                placement=dict(
                    fixture=self.table,
                    size=(0.25, 0.25),
                    pos=(0.3, 0.1),
                ),
            )
        )
        cfgs.append(
            dict(
                name="yellow_book1",
                obj_groups="book",
                graspable=True,
                object_scale=0.4,
                info=dict(
                    mjcf_path="/objects/lightwheel/book/Book043/model.xml",
                ),
                placement=dict(
                    fixture=self.table,
                    size=(0.25, 0.25),
                    pos=(0.4, -0.5),
                ),
            )
        )
        cfgs.append(
            dict(
                name="yellow_book",
                obj_groups="book",
                object_scale=0.4,
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/book/Book043/model.xml",
                ),
                placement=dict(
                    fixture=self.table,
                    size=(0.25, 0.25),
                    pos=(0.0, 0.1),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        book1 = OU.check_obj_in_receptacle_no_contact(self.env, "black_book", "wooden_two_layer_shelf", th=0.2)
        book2 = OU.check_obj_in_receptacle_no_contact(self.env, "yellow_book", "wooden_two_layer_shelf", th=0.2)
        book3 = OU.check_obj_in_receptacle_no_contact(self.env, "yellow_book1", "wooden_two_layer_shelf", th=0.2)
        book_success = book1 | book2 | book3
        gipper_success = OU.gripper_obj_far(self.env, "black_book") & OU.gripper_obj_far(self.env, "yellow_book") & OU.gripper_obj_far(self.env, "yellow_book1", th=0.4)
        return book_success & gipper_success


class L90S4PickUpTheBookOnTheRightAndPlaceItUnderTheCabinetShelf(L90S4PickUpTheBookInTheMiddleAndPlaceItOnTheCabinetShelf):
    task_name: str = "L90S4PickUpTheBookOnTheRightAndPlaceItUnderTheCabinetShelf"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "pick up the book on the right and place it under the cabinet shelf."
        return ep_meta
