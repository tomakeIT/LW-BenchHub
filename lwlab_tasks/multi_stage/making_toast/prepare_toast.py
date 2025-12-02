import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class PrepareToast(LwLabTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.TOASTER]

    """
    Prepare Toast: composite task for Making Toast activity.

    Simulates the task of preparing toast.

    Steps:
        Open the cabinet, pick the bread, place it on the cutting board, pick the jam,
        place it on the counter, and close the cabinet
    """

    task_name: str = "PrepareToast"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=FixtureType.TOASTER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the bread from the cabinet, place it on the cutting board, "
            "pick the jam, place it on the counter, and close the cabinet."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _reset_internal(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env, env_ids)
        self.cab.set_door_state(min=0.9, max=1.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("bread"),
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
                name="container",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.5, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="obj2",
                obj_groups="jam",
                placement=dict(
                    fixture=self.cab,
                    size=(0.3, 0.15),
                    pos=(0.0, -1.0),
                    offset=(-0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj3",
                obj_groups="knife",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.3),
                    pos=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                    offset=(-0.05, 0.05),
                ),
            )
        )

        return cfgs

    def _check_door_closed(self, env):
        door_state = self.cab.get_door_state(env=env)

        success = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for joint_p in door_state.values():
            success &= ~(joint_p > 0.05)
        return success

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env)
        jam_on_counter = OU.check_obj_fixture_contact(env, "obj2", self.counter)
        bread_on_cutting_board = OU.check_obj_in_receptacle(env, "obj", "container")
        cutting_board_on_counter = OU.check_obj_fixture_contact(
            env, "container", self.counter
        )
        cabinet_closed = self._check_door_closed(env)
        return jam_on_counter & gripper_obj_far & bread_on_cutting_board & cutting_board_on_counter & cabinet_closed
