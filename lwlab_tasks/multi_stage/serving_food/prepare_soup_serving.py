import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class PrepareSoupServing(LwLabTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.STOVE]

    """
    Prepare Soup Serving: composite task for Serving Food activity.

    Simulates the task of serving soup.

    Steps:
        Move the ladle from the cabinet to the pot. Then, close the cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the ladle is
            picked.
    """

    task_name: str = "PrepareSoupServing"
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.cabinet = self.register_fixture_ref(
            "cab", dict(id=self.cab_id, ref=self.stove)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Open the cabinet and move the ladle to the pot. Then close the cabinet."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cabinet.close_door(env=env, env_ids=env_ids)

    def _reset_internal(self, env_ids=None):
        return super()._reset_internal(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ladle",
                obj_groups="ladle",
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                    # rotation is such that the ladle fits in the cabinet
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pot",
                obj_groups=("pot"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        ladle_in_pot = OU.check_obj_in_receptacle(env, "ladle", "pot")
        door_state = self.cabinet.get_door_state(env=env)
        joint_positions = torch.stack(list(door_state.values()), dim=0)  # (num_joints, num_envs)
        door_closed = (joint_positions <= 0.01).all(dim=0)  # (num_envs,)
        return ladle_in_pot & door_closed
