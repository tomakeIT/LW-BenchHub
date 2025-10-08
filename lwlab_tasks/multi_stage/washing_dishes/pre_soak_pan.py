import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class PreSoakPan(BaseTaskEnvCfg, RobocasaKitchenEnvCfg):
    """
    Pre Soak Pan: composite task for Washing Dishes activity.

    Simulates the task of pre-soaking a pan.

    Steps:
        Pick the pan and sponge and place them into the sink. Then turn on the sink.
    """

    task_name: str = "PreSoakPan"

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the pan and sponge and place them into the sink. Then turn on the water."
        return ep_meta

    def _setup_scene(self, env_ids=None):
        super()._setup_scene(env_ids)
        self.sink.set_handle_state(mode="off", env=self.env)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj1",
                obj_groups=("pan"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                        # make sure sampled counter region is large enough to place the pan
                        top_size=(0.6, 0.4),
                    ),
                    size=(0.35, 0.55),
                    pos=("ref", -1.0),
                ),
                # make sure the sampled pan would fit in the sink basin
                max_size=(0.35, 0.45, None),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("sponge"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.2, 0.3),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.05),
                ),
            )
        )

        return cfgs

    def _check_pan_in_sink(self):
        return OU.obj_inside_of(self.env, "obj1", self.sink)

    def _check_sponge_in_sink(self):
        return OU.obj_inside_of(self.env, "obj2", self.sink)

    def _check_success(self):
        handle_state = self.sink.get_handle_state(env=self.env)
        water_on = handle_state["water_on"]
        pan_in_sink = OU.obj_inside_of(self.env, "obj1", self.sink, partial_check=False)
        sponge_in_sink = OU.obj_inside_of(self.env, "obj2", self.sink, partial_check=False)
        return water_on & pan_in_sink & sponge_in_sink & OU.gripper_obj_far(self.env, "obj1") & OU.gripper_obj_far(self.env, "obj2")
