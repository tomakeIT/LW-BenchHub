import copy
from lwlab.core.tasks.base import LwLabTaskBase
import re
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate(LwLabTaskBase):
    """
    L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate: put the white mug on the plate and put the chocolate pudding to the right of the plate
    """

    task_name: str = "L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter
        self.place_success = {}
        self.chocolate_pudding = "chocolate_pudding"
        self.plate = "plate"
        self.porcelain_mug = "porcelain_mug"
        self.red_coffee_mug = "red_coffee_mug"
        self.white_yellow_mug = "white_yellow_mug"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _reset_internal(self, env_ids):
        super()._reset_internal(env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the white mug and put it on the plate, and put the chocolate pudding to the right of the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos=(0.0, -1), size=(0.8, 0.6)):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                margin=0.02,
                ensure_valid_placement=True,
            )
        plate_placement = get_placement()
        chocolate_pudding_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.0, -0.7),
            margin=0.02,
            rotation=np.pi / 2.0,
            ensure_valid_placement=True,
        )
        porcelain_mug_placement = get_placement()
        red_coffee_mug_placement = get_placement()

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None, init_robot_here=False):
            if mjcf_path is not None:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        init_robot_here=init_robot_here,
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
                        init_robot_here=init_robot_here,
                        placement=placement,
                    )
                )
        add_cfg(self.chocolate_pudding, self.chocolate_pudding, True, chocolate_pudding_placement, mjcf_path="/objects/lightwheel/chocolate_pudding/ChocolatePudding001/model.xml")
        add_cfg(self.plate, self.plate, True, plate_placement, mjcf_path="/objects/lightwheel/plate/Plate012/model.xml", init_robot_here=True)
        add_cfg(self.porcelain_mug, self.porcelain_mug, True, porcelain_mug_placement, mjcf_path="/objects/lightwheel/cup/Cup012/model.xml")
        add_cfg(self.red_coffee_mug, self.red_coffee_mug, True, red_coffee_mug_placement, mjcf_path="/objects/lightwheel/cup/Cup030/model.xml")

        return cfgs

    def _check_success(self, env):
        success_porcelain_mug = OU.check_place_obj1_on_obj2(
            env,
            self.porcelain_mug,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5,     # velocity vector length less than 0.5
            gipper_th=0.35
        )
        success_chocolate_pudding = OU.check_place_obj1_side_by_obj2(
            env,
            self.chocolate_pudding,
            self.plate,
            check_states={
                "side": "right",    # right side of obj2
                "side_threshold": 0.4,    # threshold for distance between obj1 and obj2 in other directions, 0.25*(min(obj2_obj.size[:2]) + min(obj1_obj.size[:2]))/ 2
                "margin_threshold": [0.001, 0.1],     # threshold for distance between obj1 and obj2, 0.001
                "parallel": [0, 0, 1],    # parallel to y and z axis
                "parallel_threshold": 0.95,
                "gripper_far": True,
                "contact": False,   # not allowed to contact with obj2
            },
            gipper_th=0.35
        )
        print(f"porcelain_mug success state: {success_porcelain_mug}")
        print(f"chocolate_pudding success state: {success_chocolate_pudding}")
        return success_porcelain_mug & success_chocolate_pudding
