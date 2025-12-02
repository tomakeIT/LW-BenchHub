import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class HeatMug(LwLabTaskBase):
    """
    Heat Mug: composite task for Reheating Food activity.

    Simulates the task of reheating a mug.

    Steps:
        Open the cabinet, pick the mug, place it inside the microwave, and close
        the microwave.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.MICROWAVE]
    task_name: str = "HeatMug"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=self.microwave)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the mug from the cabinet and place it inside the microwave. Then close the microwave."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)
        self.microwave.open_door(env=env, env_ids=env_ids)

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="mug",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env)
        obj_in_microwave = OU.obj_inside_of(env, "obj", self.microwave)
        door_closed = self.microwave.is_closed(env=env)

        return obj_in_microwave & gripper_obj_far & door_closed
