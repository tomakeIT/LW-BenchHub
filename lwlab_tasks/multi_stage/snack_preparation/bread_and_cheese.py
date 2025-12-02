import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class BreadAndCheese(LwLabTaskBase):
    """
    Bread And Cheese: composite task for Snack Preparation activity.

    Simulates the preparation of a bread and cheese snack.

    Steps:
        Pick the bread and cheese, place them on the cutting board.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER_NON_CORNER]
    task_name: str = "BreadAndCheese"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the bread and cheese on the cutting board."
        return ep_meta

    def _reset_internal(self, env, env_ids=None):
        return super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("bread"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        top_size=(0.6, 0.6),
                    ),
                    size=(0.50, 0.30),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        top_size=(0.6, 0.6),
                    ),
                    size=(0.5, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="obj2",
                obj_groups="cheese",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        top_size=(0.6, 0.6),
                    ),
                    size=(0.3, 0.15),
                    pos=(0.0, -1.0),
                    offset=(-0.05, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env, "obj") & OU.gripper_obj_far(env, "obj2") & OU.gripper_obj_far(env, "container")
        food_on_cutting_board = OU.check_obj_in_receptacle(
            env, "obj", "container"
        ) & OU.check_obj_in_receptacle(env, "obj2", "container")
        return food_on_cutting_board & gripper_obj_far
