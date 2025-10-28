# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .fixture import Fixture
from .fixture_types import FixtureType
import lwlab.utils.object_utils as OU
import torch


class Blender(Fixture):
    fixture_types = [FixtureType.BLENDER]
    _BLENDER_LID_POS_THRESH = 0.04

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self.lid_closed_pos = None
        self._lid_on_blender = torch.tensor([True], dtype=torch.bool).repeat(num_envs)
        self._turned_on = torch.tensor([False], dtype=torch.bool).repeat(num_envs)
        self._button_contact_prev_timestep = torch.tensor([False], dtype=torch.bool).repeat(num_envs)
        self.blender_lid = None

    def add_auxiliary_fixture(self, auxiliary_fixture):
        self.blender_lid = auxiliary_fixture

    def get_lid_closed_pos(self, env):
        if self.lid_closed_pos is None:
            self.lid_closed_pos = OU.get_pos_after_rel_offset(self, self.anchor_offset)
        return self.lid_closed_pos

    def get_curr_lid_pos(self, env):
        if self.blender_lid is None:
            return None
        return env.scene.rigid_objects[f"{self.blender_lid.name}_main"].data.body_com_pos_w

    def update_state(self, env):
        curr_lid_pos = self.get_curr_lid_pos(env)
        if curr_lid_pos is None:
            self._lid_on_blender = False
        else:
            closed_lid_pos = self.get_lid_closed_pos(env)
            self._lid_on_blender = (
                torch.norm(curr_lid_pos - closed_lid_pos)
                < self._BLENDER_LID_POS_THRESH
            ) & OU.check_fxtr_upright(env, f"{self.blender_lid.name}_main", th=7)
        button_pressed = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        for gripper_name in [name for name in list(env.scene.sensors.keys()) if "gripper" in name and "contact" in name]:
            button_pressed |= env.cfg.check_contact(gripper_name.replace("_contact", ""), "{}_power_button_main".format(self.name))
        # since the state updates very often and the same button is used for turning on/off
        # we look at the release of the button to determine the state. If we look at the press then
        # the state will flicker between on and off
        switch_state = self._button_contact_prev_timestep & (~button_pressed)

        if not self._lid_on_blender:
            self._turned_on = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        else:
            self._turned_on[switch_state] = ~self._turned_on[switch_state]
        self._button_contact_prev_timestep = button_pressed

    def get_state(self):
        state = dict(
            lid_on_blender=self._lid_on_blender,
            turned_on=self._turned_on,
        )
        return state
