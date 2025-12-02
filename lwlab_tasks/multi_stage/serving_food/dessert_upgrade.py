import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class DessertUpgrade(LwLabTaskBase):
    """
    Dessert Upgrade: composite task for Serving Food activity.

    Simulates the task of serving dessert.

    Steps:
        Move the dessert items from the plate to the tray.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER_NON_CORNER]
    task_name: str = "DessertUpgrade"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(1.0, 0.4))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = f"Move the dessert items from the plate to the tray."

        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _reset_internal(self, env, env_ids=None):
        return super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="tray",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(1, 0.5),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert1",
                obj_groups="sweets",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(0, -1),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert2",
                obj_groups="sweets",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(0, -1),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        sweets_on_tray = OU.check_obj_in_receptacle(
            env, "dessert1", "receptacle"
        ) & OU.check_obj_in_receptacle(env, "dessert2", "receptacle")

        return sweets_on_tray & OU.gripper_obj_far(env, "receptacle")
