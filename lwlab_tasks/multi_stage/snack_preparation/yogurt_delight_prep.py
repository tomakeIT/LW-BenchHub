import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class YogurtDelightPrep(LwLabTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]

    """
    Yogurt Delight Prep: composite task for Snack Preparation activity.

    Simulates the preparation of a yogurt delight snack.

    Steps:
        Place the yogurt and fruit onto the counter.
    """

    task_name: str = "YogurtDelightPrep"
    EXCLUDE_LAYOUTS = LwLabTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        # want space for all the objects
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the yogurt and fruit onto the counter."
        return ep_meta

    def _reset_internal(self, env, env_ids=None):
        return super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="yogurt",
                obj_groups="yogurt",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.2),
                    pos=(0, -1),
                    offset=(0, -0.02),
                ),
            )
        )

        self.num_fruits = self.rng.choice([1, 2])
        for i in range(self.num_fruits):
            cfgs.append(
                dict(
                    name=f"fruit_{i}",
                    obj_groups="fruit",
                    placement=dict(
                        fixture=self.cab,
                        size=(0.5, 0.15),
                        pos=(0, -1),
                        offset=(0.05 * i, -0.02),
                    ),
                )
            )

        return cfgs

    def _check_success(self, env):
        fruits_on_counter = torch.stack([
            OU.check_obj_fixture_contact(env, f"fruit_{i}", self.counter)
            for i in range(self.num_fruits)
        ], dim=0).all(dim=0)

        yogurt_on_counter = OU.check_obj_fixture_contact(env, "yogurt", self.counter)
        items_on_counter = fruits_on_counter & yogurt_on_counter

        gripper_far_fruits = torch.stack([
            OU.gripper_obj_far(env, f"fruit_{i}")
            for i in range(self.num_fruits)
        ], dim=0).all(dim=0)

        gripper_far_yogurt = OU.gripper_obj_far(env, "yogurt")
        objs_far = gripper_far_fruits & gripper_far_yogurt
        return items_on_counter & objs_far
