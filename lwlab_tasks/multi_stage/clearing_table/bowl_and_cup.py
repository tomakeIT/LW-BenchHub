import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter


class BowlAndCup(LwLabTaskBase):
    """
    Bowl And Cup: composite task for Clearing Table activity.

    Simulates the process of efficiently clearing the table.

    Steps:
        Place the cup inside the bowl on the dining table and move it to any counter.

    Restricted to layouts with a dining table.
    """

    layout_registry_names: list[int] = [FixtureType.DINING_COUNTER, FixtureType.STOOL]
    task_name: str = "BowlAndCup"
    EXCLUDE_LAYOUTS = LwLabTaskBase.DINING_COUNTER_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool, size=(0.50, 0.35)),
        )

        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Place the cup inside the bowl on the dining table and move the bowl to any counter."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"cup",
                obj_groups=["cup"],
                graspable=True,
                washable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.50, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"bowl",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    ref_obj="cup",
                    size=(0.50, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        cup_in_bowl = OU.check_obj_in_receptacle(env, "cup", "bowl")
        bowl_on_counter = torch.stack(
            [
                OU.check_obj_fixture_contact(env, "bowl", fxtr)
                for (_, fxtr) in self.fixtures.items()
                if isinstance(fxtr, Counter) and fxtr != self.dining_table
            ], dim=0)
        bowl_on_counter = bowl_on_counter.any(dim=0)
        return cup_in_bowl & bowl_on_counter & OU.gripper_obj_far(env, "bowl")
