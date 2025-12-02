import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class CupcakeCleanup(LwLabTaskBase):

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]

    """
    Cupcake Cleanup: composite task for Baking activity.

    Simulates the task of cleaning up after baking cupcakes.

    Steps:
        Move the cupcake off the tray onto the counter, and place the bowl used for
        mixing into the sink.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "CupcakeCleanup"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Move the fresh-baked cupcake off the tray onto the counter, "
            "and place the bowl used for mixing into the sink."
        )
        return ep_meta

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cupcake",
                obj_groups="cupcake",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.6, 0.4)
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    try_to_place_in="tray",
                    try_to_place_in_kwargs=dict(
                        object_scale=0.6,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.65, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_far = OU.gripper_obj_far(env, "cupcake") & OU.gripper_obj_far(env, "bowl")
        bowl_in_sink = OU.obj_inside_of(env, "bowl", self.sink)
        cupcake_on_counter = OU.check_obj_fixture_contact(env, "cupcake", self.counter)

        return gripper_far & bowl_in_sink & cupcake_on_counter
