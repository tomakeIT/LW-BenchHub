import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class BeverageOrganization(LwLabTaskBase):
    """
    Beverage Organization: composite task for Setting The Table activity.

    Simulates the task of organizing beverages.

    Steps:
        Move the drinks to the dining counter.

    Restricted to layouts which have a dining table (long counter area with
    stools).
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.STOOL]
    task_name: str = "BeverageOrganization"
    EXCLUDE_LAYOUTS = LwLabTaskBase.STOOL_EXCLUDED_LAYOUT

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        if "counter" in self.fixture_refs:
            self.counter = self.fixture_refs["counter"]
            self.dining_table = self.fixture_refs["dining_table"]
        else:

            self.dining_table = self.register_fixture_ref(
                "dining_table",
                dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
            )
            self.counter = self.get_fixture(id=FixtureType.COUNTER)
            # do not want to sample the dining table or a counter with a builtin sink
            # TODO Change later!
            while self.counter == self.dining_table or "corner" in self.counter.name:
                self.counter = self.get_fixture(FixtureType.COUNTER)
            self.fixture_refs["counter"] = self.counter

        self.init_robot_base_ref = self.counter
        self.num_bev = -1

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Move the drinks to the dining counter."
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

        self.num_bev = self.rng.choice([2, 3, 4])
        for i in range(self.num_bev):
            cfgs.append(
                dict(
                    name=f"obj_{i}",
                    obj_groups="drink",
                    placement=dict(
                        fixture=self.counter,
                        size=(0.6, 0.4),
                        pos=(0, -1.0),
                    ),
                )
            )

        return cfgs

    def get_beverage_num(self, env):
        nums = 0
        for object in env.cfg.object_cfgs:
            if object.get("obj_groups") == "drink":
                nums += 1
        return nums

    def _check_success(self, env):
        if self.is_replay_mode:
            self.num_bev = self.get_beverage_num(env)
        drinks_on_dining = torch.stack([
            OU.check_obj_fixture_contact(env, f"obj_{i}", self.dining_table)
            for i in range(self.num_bev)
        ], dim=0).all(dim=0)
        result = drinks_on_dining
        for i in range(self.num_bev):
            result = result & OU.gripper_obj_far(env, f"obj_{i}")
        return result
