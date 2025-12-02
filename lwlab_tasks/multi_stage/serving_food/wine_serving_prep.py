import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class WineServingPrep(LwLabTaskBase):
    """
    Wine Serving Prep: composite task for Serving Food activity.

    Simulates the task of serving wine.

    Steps:
        Move the wine and the cup from the cabinet to the dining table.

    Restricted to layouts which have a dining table (long counter area
    with stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the wine and
            cup are picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.SINK, FixtureType.STOOL]
    task_name: str = "WineServingPrep"
    EXCLUDE_LAYOUTS = LwLabTaskBase.STOOL_EXCLUDED_LAYOUT
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        alcohol_name = self.get_obj_lang("alcohol")
        cup_name = self.get_obj_lang("cup")
        decoration_name = self.get_obj_lang("decoration")
        ep_meta["lang"] = (
            "Open the cabinet directly in front. "
            f"Then move the {alcohol_name} and the {cup_name} to the counter with the {decoration_name} on it."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.close_door(env=env, env_ids=env_ids)

    def _reset_internal(self, env, env_ids=None):
        return super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="alcohol",
                obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(-0.6, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cup",
                obj_groups=["cup", "mug"],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0.6, -1.0),
                ),
            )
        )

        # adding indicator
        cfgs.append(
            dict(
                name="decoration",
                obj_groups="decoration",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),
                    pos=(0.0, 0.0),
                ),
            )
        )

        # adding distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups=["vegetable", "fruit", "sweets", "dairy"],
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_alcohol_far = OU.gripper_obj_far(env, obj_name="alcohol")
        gripper_cup_far = OU.gripper_obj_far(env, obj_name="cup")
        condiment1_on_counter = OU.check_obj_fixture_contact(
            env, "alcohol", self.dining_table
        )
        condiment2_on_counter = OU.check_obj_fixture_contact(
            env, "cup", self.dining_table
        )

        return gripper_alcohol_far & gripper_cup_far & condiment1_on_counter & condiment2_on_counter
