import numpy as np
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class SpicyMarinade(LwLabTaskBase):
    """
    Spicy Marinade: composite task for Mixing And Blending activity.

    Simulates the task of preparing a spicy marinade.

    Steps:
        Open the cabinet. Place the bowl and condiment on the counter. Then place
        the lime and garlic on the cutting board.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "SpicyMarinade"
    EXCLUDE_LAYOUTS = LwLabTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        # need to fit the bowl and condiment in the cab, so use double door hinge
        self.cab = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Open the cabinet. Place the bowl and condiment on the counter. "
            "Then place the lime and garlic on the cutting board."
        )
        return ep_meta

    def _reset_internal(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env, env_ids)
        self.cab.set_door_state(min=0.0, max=0.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.8, 0.4),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.cab,
                    size=(0.6, 0.4),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.2),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="lime",
                obj_groups="lime",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.2),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="garlic",
                obj_groups="garlic",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.2),
                    pos=("ref", -1),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        objs_on_counter = OU.check_obj_fixture_contact(
            env, "bowl", self.counter
        ) & OU.check_obj_fixture_contact(env, "condiment", self.counter)
        objs_on_board = OU.check_obj_in_receptacle(
            env, "lime", "receptacle"
        ) & OU.check_obj_in_receptacle(env, "garlic", "receptacle")
        gripper_far = (
            OU.gripper_obj_far(env, "receptacle")
            & OU.gripper_obj_far(env, "bowl")
            & OU.gripper_obj_far(env, "condiment")
        )

        return objs_on_counter & objs_on_board & gripper_far
