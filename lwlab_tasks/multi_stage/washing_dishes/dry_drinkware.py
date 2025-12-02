import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import math


class DryDrinkware(LwLabTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER]

    """
    Dry Drinkware: composite task for Washing Dishes activity.

    Simulates the task of drying drinkware.

    Steps:
        Pick the mug from the counter and place it upside down in the open cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet in which the mug is placed.
    """

    task_name: str = "DryDrinkware"
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.5))
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "A wet mug is on the counter and needs to be dried. "
            "Pick it up and place it upside down in the open cabinet."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _reset_internal(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env, env_ids)
        self.cab.set_door_state(min=0.9, max=1, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        x_positions = [-1, 1]
        self.rng.shuffle(x_positions)

        cfgs.append(
            dict(
                name="mug",
                obj_groups="mug",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.3),
                    pos=(x_positions[0], -1.0),
                ),
            )
        )

        return cfgs

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    def _check_success(self, env):
        mug_rot = env.scene.rigid_objects["mug"].data.body_com_quat_w[:, 0, :]
        # make sure the mug is placed upside down
        mug_euler = []
        for i in range(mug_rot.shape[0]):
            roll, pitch, yaw = self.euler_from_quaternion(mug_rot[i, 0], mug_rot[i, 1], mug_rot[i, 2], mug_rot[i, 3])
            mug_euler.append((roll, pitch, yaw))
        mug_rot = torch.tensor(mug_euler, dtype=torch.float32, device=env.device)
        return (
            OU.gripper_obj_far(env, obj_name="mug")
            & (torch.abs(mug_rot[:, 2]) > 3)
            & OU.check_obj_fixture_contact(env, "mug", self.cab)
        )
