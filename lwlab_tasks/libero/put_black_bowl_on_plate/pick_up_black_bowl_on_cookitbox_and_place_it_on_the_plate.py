import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from .put_black_bowl_on_plate import PutBlackBowlOnPlate
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import copy


class LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate(PutBlackBowlOnPlate):

    task_name: str = 'LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl on the cookies box and place it on the plate."
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.bowl = "bowl"
        self.bowl_target = "bowl_target"
        self.storage_furniture.set_target_reg_int(("int1",))

    def _load_model(self):
        super()._load_model()
        cookie_pos = list(self.object_placements[self.cookies][0])
        cookie_size = self.object_placements[self.cookies][2].size
        bowl_obj = copy.deepcopy(self.object_placements[self.bowl_target])
        bowl_pos = list(bowl_obj[0])
        bowl_pos[0] = cookie_pos[0]
        bowl_pos[1] = cookie_pos[1]
        bowl_pos[2] = cookie_pos[2] + cookie_size[2] / 2.0 + bowl_obj[2].size[2] / 2.0
        bowl_obj = list(bowl_obj)
        bowl_obj[0] = bowl_pos
        self.object_placements[self.bowl_target] = tuple(bowl_obj)

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                info=dict(
                    mjcf_path=self.bowl_mjcf_path
                ),
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.2, -0.6),
                    ensure_valid_placement=True,
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.bowl_target,
                obj_groups="bowl",
                info=dict(
                    mjcf_path=self.bowl_mjcf_path
                ),
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.4),
                    pos=(-0.4, -0.4),
                    ensure_valid_placement=True,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        '''
        Check if the bowl is placed on the plate.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(self.env, self.bowl_target)
        object_on_plate = OU.check_obj_in_receptacle(self.env, self.bowl_target, self.plate)
        return object_on_plate & is_gripper_obj_far


class LSPickUpTheBlackBowlOnTheRamekinAndPlaceItOnThePlate(LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate):
    task_name: str = 'LSPickUpTheBlackBowlOnTheRamekinAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl on the ramekin and place it on the plate."
        return ep_meta

    def _load_model(self):
        super()._load_model()
        ramekin_pos = list(self.object_placements[self.ramekin][0])
        bowl_obj = copy.deepcopy(self.object_placements[self.bowl_target])
        bowl_pos = list(bowl_obj[0])
        bowl_pos[0] = ramekin_pos[0]
        bowl_pos[1] = ramekin_pos[1]
        bowl_pos[2] = bowl_pos[2] + self.object_placements[self.ramekin][2].size[2] - 0.05
        bowl_obj = list(bowl_obj)
        bowl_obj[0] = bowl_pos
        self.object_placements[self.bowl_target] = tuple(bowl_obj)


class LSPickUpTheBlackBowlOnTheStoveAndPlaceItOnThePlate(LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate):
    task_name: str = 'LSPickUpTheBlackBowlOnTheStoveAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl on the stove and place it on the plate."
        return ep_meta

    def _load_model(self):
        super()._load_model()
        stove_pos = self.stove.pos
        bowl_obj = copy.deepcopy(self.object_placements[self.bowl_target])
        bowl_pos = list(bowl_obj[0])
        bowl_pos[0] = stove_pos[0]
        bowl_pos[1] = stove_pos[1]
        bowl_pos[2] = bowl_pos[2] + self.stove.size[2]
        bowl_obj = list(bowl_obj)
        bowl_obj[0] = bowl_pos
        self.object_placements[self.bowl_target] = tuple(bowl_obj)


class LSPickUpTheBlackBowlOnTheWoodenCabinetAndPlaceItOnThePlate(LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate):
    task_name: str = 'LSPickUpTheBlackBowlOnTheWoodenCabinetAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl on the wooden cabinet and place it on the plate."
        return ep_meta

    def _load_model(self):
        super()._load_model()
        cabinet_pos = self.storage_furniture.pos
        bowl_obj = copy.deepcopy(self.object_placements[self.bowl_target])
        bowl_pos = list(bowl_obj[0])
        bowl_pos[0] = cabinet_pos[0]
        bowl_pos[1] = cabinet_pos[1]
        bowl_pos[2] = cabinet_pos[2] + self.storage_furniture.size[2] / 2 + bowl_obj[2].size[2] / 2.0 + 0.01
        bowl_obj = list(bowl_obj)
        bowl_obj[0] = bowl_pos
        self.object_placements[self.bowl_target] = tuple(bowl_obj)
