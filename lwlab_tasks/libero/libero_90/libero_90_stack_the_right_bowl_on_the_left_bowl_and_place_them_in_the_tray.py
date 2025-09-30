import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L90L4StackTheRightBowlOnTheLeftBowlAndPlaceThemInTheTray(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = 'L90L4StackTheRightBowlOnTheLeftBowlAndPlaceThemInTheTray'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["saladdressing"]
    removable_fixtures: list[str] = ["saladdressing"]

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Stack the right bowl on the left bowl and place them in the tray."
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.init_robot_base_ref = self.dining_table
        self.chocolate_pudding = "chocolate_pudding"
        self.tray = "tray"
        self.bowl_left = "bowl_left"
        self.bowl_right = "bowl_right"

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.tray,
                obj_groups=self.tray,
                object_scale=0.6,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.25, 0.5),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/tray/Tray016/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                object_scale=0.5,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(-0.25, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/chocolate_pudding/ChocolatePudding001/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.bowl_left,
                obj_groups="bowl",
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml"
                ),
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.5, -0.5),
                    ensure_valid_placement=True,
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.bowl_right,
                obj_groups="bowl",
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml"
                ),
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.0, -0.5),
                    ensure_valid_placement=True,
                ),
            )
        )

        return cfgs

    def _check_success(self):

        is_gripper_obj_far = OU.gripper_obj_far(self.env, self.bowl_right)
        bowl_on_bowl = OU.check_obj_in_receptacle_no_contact(self.env, self.bowl_right, self.bowl_left)
        bowl_on_plate = OU.check_obj_in_receptacle_no_contact(self.env, self.bowl_left, self.tray)
        return bowl_on_plate & bowl_on_bowl & is_gripper_obj_far
