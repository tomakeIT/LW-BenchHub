import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class VeggieDipPrep(LwLabTaskBase):
    """
    Veggie Dip Prep: composite task for Snack Preparation activity.

    Simulates the preparation of a vegetable dip snack.

    Steps:
        Place the two vegetables and a bowl onto the tray for setting up a vegetable
        dip station.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER]
    task_name: str = "VeggieDipPrep"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(1, 0.6))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Place the two vegetables and bowl onto the tray for setting up a vegetable dip station."
        return ep_meta

    def _reset_internal(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        # Tray in the center of the counter
        cfgs.append(
            dict(
                name="tray",
                obj_groups="tray",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(0, -1),
                ),
            )
        )

        # Two "dippable" vegetables to the left of tray
        cfgs.append(
            dict(
                name="cucumber",
                obj_groups="cucumber",
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.5),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="carrot",
                obj_groups="carrot",
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.5),
                    pos=(0, -1.0),
                ),
            )
        )

        # Bowl to the right of tray
        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.6),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        gripper_far = torch.logical_and(
            OU.gripper_obj_far(env, "bowl"),
            torch.logical_and(OU.gripper_obj_far(env, "cucumber"), OU.gripper_obj_far(env, "carrot")))
        vegetables_in_tray = torch.logical_and(
            OU.check_obj_in_receptacle(env, "cucumber", "tray"),
            OU.check_obj_in_receptacle(env, "carrot", "tray"))
        bowl_in_tray = OU.check_obj_in_receptacle(env, "bowl", "tray")

        return gripper_far & vegetables_in_tray & bowl_in_tray
