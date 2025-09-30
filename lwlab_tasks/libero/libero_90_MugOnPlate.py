import copy
import re
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from .libero_10_MugOnAndChocolateRightPlate import L10L6PutTheWhiteMugOnThePlateAndPutTheChocolatePuddingToTheRightOfThePlate


class L90L6PutTheRedMugOnThePlate(L10L6PutTheWhiteMugOnThePlateAndPutTheChocolatePuddingToTheRightOfThePlate):
    """
    L90L6PutTheRedMugOnThePlate: put the red mug on the right plate
    """

    task_name: str = "L90L6PutTheRedMugOnThePlate"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the red mug and put it on the plate"
        return ep_meta

    def _check_success(self):
        return OU.check_place_obj1_on_obj2(
            self.env,
            self.red_coffee_mug,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )


class L90L5PutTheRedMugOnTheRightPlate(L10L6PutTheWhiteMugOnThePlateAndPutTheChocolatePuddingToTheRightOfThePlate):
    """
    L90L5PutTheRedMugOnTheRightPlate: put the red mug on the right plate
    """

    task_name: str = "L90L5PutTheRedMugOnTheRightPlate"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the red mug and put it on the plate"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] == self.chocolate_pudding:
                cfgs[cfg_index]['name'] = self.white_yellow_mug
                cfgs[cfg_index]['info']['mjcf_path'] = "/objects/lightwheel/cup/Cup014/model.xml"
                break
            cfg_index += 1
        return cfgs

    def _check_success(self):
        return OU.check_place_obj1_on_obj2(
            self.env,
            self.red_coffee_mug,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
