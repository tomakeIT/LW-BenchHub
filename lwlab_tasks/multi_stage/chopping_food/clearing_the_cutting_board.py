import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class ClearingTheCuttingBoard(LwLabTaskBase):
    """
    Clearing The Cutting Board: composite task for Chopping Food activity.

    Simulates the task of clearing the cutting board.

    Steps:
        Clear the non-vegetable object off the cutting board and place the
        vegetables onto it.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER]
    task_name: str = "ClearingTheCuttingBoard"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.5, 0.5))
        )
        self.init_robot_base_ref = self.counter

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Clear the non-vegetable object off the cutting board and place the vegetables onto it."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="non_vegetable",
                graspable=True,
                obj_groups="food",
                exclude_obj_groups="vegetable",
                placement=dict(
                    fixture=self.counter,
                    size=(0.2, 0.2),
                    ensure_object_boundary_in_range=False,
                    pos=(0, -0.3),
                    try_to_place_in="cutting_board",
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        vegetable1_cutting_board_contact = OU.check_obj_in_receptacle(
            env, "vegetable1", "non_vegetable_container"
        )
        vegetable2_cutting_board_contact = OU.check_obj_in_receptacle(
            env, "vegetable2", "non_vegetable_container"
        )
        cutting_board_cleared = ~ OU.check_obj_in_receptacle(
            env, "non_vegetable", "non_vegetable_container"
        )
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="non_vegetable_container")

        return (
            vegetable1_cutting_board_contact
            & vegetable2_cutting_board_contact
            & gripper_obj_far
            & cutting_board_cleared
        )
