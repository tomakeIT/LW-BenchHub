import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType


class MicrowavePressButton(LwLabTaskBase):
    layout_registry_names: list[int] = [FixtureType.MICROWAVE]
    """
    Class encapsulating the atomic microwave press button tasks.

    Args:
        behavior (str): "turn_on" or "turn_off". Used to define the desired
            microwave manipulation behavior for the task
    """
    behavior: str = "turn on"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the microwave tasks
        """
        super()._setup_kitchen_references(scene)
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))
        if self.behavior == "turn_off":
            self.microwave._turned_on = torch.tensor([True], device=scene.context.device).repeat(scene.num_envs)
        self.init_robot_base_ref = self.microwave

    def get_ep_meta(self):
        """
        Get the episode metadata for the microwave tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        if self.behavior == "turn_on":
            ep_meta["lang"] = "press the start button on the microwave"
        elif self.behavior == "turn_off":
            ep_meta["lang"] = "press the stop button on the microwave"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the microwave tasks. This includes the object placement configurations.
        Place the object inside the microwave and on top of another container object inside the microwave

        Returns:
            list: List of object configurations.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="all",
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)

    def _check_success(self, env):
        """
        Check if the microwave manipulation task is successful.

        Returns:
            torch.Tensor: Bool tensor indicating success for each environment.
        """
        turned_on = self.microwave.get_state()["turned_on"]
        gripper_button_far = self.microwave.gripper_button_far(
            env, button="start_button" if self.behavior == "turn_on" else "stop_button"
        )

        if self.behavior == "turn_on":
            return turned_on & gripper_button_far
        elif self.behavior == "turn_off":
            return ~turned_on & gripper_button_far


class TurnOnMicrowave(MicrowavePressButton):
    task_name: str = "TurnOnMicrowave"
    behavior: str = "turn_on"


class TurnOffMicrowave(MicrowavePressButton):
    task_name: str = "TurnOffMicrowave"
    behavior: str = "turn_off"
