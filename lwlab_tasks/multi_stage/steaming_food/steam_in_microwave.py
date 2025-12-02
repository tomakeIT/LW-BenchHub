import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import math


class SteamInMicrowave(LwLabTaskBase):
    """
    Steam In Microwave: composite task for Steaming Food activity.

    Simulates the task of steaming a vegetable in a microwave.

    Steps:
        Pick the vegetable from the sink and place it in the bowl. Then pick the
        bowl and place it in the microwave. Then close the microwave door and press
        the start button.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.MICROWAVE, FixtureType.SINK]
    task_name: str = "SteamInMicrowave"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_name = OU.get_obj_lang(self, "vegetable")
        ep_meta["lang"] = (
            f"Pick the {vegetable_name} from the sink and place it in the bowl. "
            "Then pick the bowl and place it in the microwave. "
            "Then close the microwave door and press the start button."
        )

        return ep_meta

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)
        self.sink.set_handle_state(mode="off", env=env)
        self.microwave.set_door_state(min=0.90, max=1.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.35, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.3, 0.2),
                    pos=(0.0, 1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter_0",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.40),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter_1",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        vegetable_in_bowl = OU.check_obj_in_receptacle(env, "vegetable", "bowl")
        bowl_in_microwave = OU.obj_inside_of(env, "bowl", self.microwave)

        door_state = self.microwave.get_door_state(env=env)
        door_closed = torch.ones(list(door_state.values())[0].shape[0], device=env.device)
        for joint_p in door_state.values():
            door_closed = torch.logical_and(door_closed, joint_p < 0.05)

        button_pressed = self.microwave.get_state()["turned_on"]

        return vegetable_in_bowl & bowl_in_microwave & door_closed & button_pressed
