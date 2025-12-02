import numpy as np
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from typing import Union
import torch


class FryingPanAdjustment(LwLabTaskBase):
    """
    Frying Pan Adjustment: composite task for Frying activity.

    Simulates the task of adjusting the frying pan on the stove.

    Steps:
        Move the pan from the current burner to another burner and turn on
        the burner.
    """

    layout_registry_names: list[int] = [FixtureType.STOVE]
    task_name: str = "FryingPanAdjustment"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.stove

    def _reset_internal(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """

        # First call super reset so that the pan is placed on the stove
        # then determine where it is placed and turn on the corresponding burner and update the start_loc
        super()._reset_internal(env, env_ids)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs)
        self.valid_knobs = self.stove.get_knobs_state(env=env).keys()

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.start_loc: Union[str, None] = [None] * self.context.num_envs
        self.valid_knobs = self.stove.get_knobs_state(env=env).keys()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick and place the pan from the current burner to another burner and turn the burner on."
        return ep_meta

    def _check_success(self, env):
        if all([loc is None for loc in self.start_loc]):
            pan_loc = self.stove.check_obj_location_on_stove(env, "obj", need_knob_on=False)
            for env_id in range(env.num_envs):
                for knob in self.valid_knobs:
                    if pan_loc[env_id] == knob:
                        self.start_loc[env_id] = pan_loc[env_id]
                        self.stove.set_knob_state(mode="on", knob=knob, env=env, env_ids=[env_id])
                        break
                    else:
                        self.stove.set_knob_state(mode="off", knob=knob, env=env, env_ids=[env_id])
            return torch.tensor([False], device=env.device).repeat(env.num_envs)
        else:
            # get the current location of the pan on the stove
            curr_loc = self.stove.check_obj_location_on_stove(env, "obj")
            knob_on_loc = torch.tensor([loc is not None for loc in curr_loc], device=env.device)
            not_at_start_loc = torch.tensor([loc != self.start_loc[env_id] for loc, env_id in zip(curr_loc, range(env.num_envs))], device=env.device)
            # return success if the pan is on a burner, the burner is on, and the burner is not the same as the start location
            return knob_on_loc & not_at_start_loc
