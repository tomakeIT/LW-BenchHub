import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from .put_black_bowl_on_plate import PutBlackBowlOnPlate
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class LSPickUpTheBlackBowlNextToTheCookieBoxAndPlaceItOnThePlate(PutBlackBowlOnPlate):

    task_name: str = 'LSPickUpTheBlackBowlNextToTheCookieBoxAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl next to the cookies box and place it on the plate."
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.bowl = "bowl"
        self.bowl_target = "bowl_target"

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
        is_gripper_obj_far = OU.gripper_obj_far(self.env, self.bowl_target, th=0.35)
        object_on_plate = OU.check_obj_in_receptacle(self.env, self.bowl_target, self.plate)
        return object_on_plate & is_gripper_obj_far
