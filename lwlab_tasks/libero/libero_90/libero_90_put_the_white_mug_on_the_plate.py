import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L90L6PutTheWhiteMugOnThePlate(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = 'L90L6PutTheWhiteMugOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the white mug on the plate."
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.init_robot_base_ref = self.dining_table
        self.chocolate_pudding = "chocolate_pudding"
        self.plate = "plate"
        self.porcelain_mug = "porcelain_mug"
        self.red_coffee_mug = "red_coffee_mug"

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                object_scale=0.5,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(0.5, -0.6),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/chocolate_pudding/ChocolatePudding001/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(0.25, -0.8),
                    ensure_valid_placement=True,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/plate/Plate012/model.xml",
                )
            )
        )

        cfgs.append(
            dict(
                name=self.porcelain_mug,
                obj_groups=self.porcelain_mug,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.25, -0.8),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/cup/Cup012/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.red_coffee_mug,
                obj_groups=self.red_coffee_mug,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.5, -0.8),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/cup/Cup030/model.xml",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        '''
        Check if the bowl is placed on the plate.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(self.env, self.porcelain_mug)
        object_on_plate = OU.check_obj_in_receptacle(self.env, self.porcelain_mug, self.plate)
        return object_on_plate & is_gripper_obj_far
