import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class OpenStandMixerHead(LwLabTaskBase):
    layout_registry_names: list[int] = [FixtureType.STAND_MIXER]

    task_name: str = "OpenStandMixerHead"
    enable_fixtures: list[str] = ["stand_mixer"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stand_mixer = self.register_fixture_ref("stand_mixer", dict(id=FixtureType.STAND_MIXER))
        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the stand mixer head."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)

    def _check_success(self, env):
        """
        Check if the stand mixer head is open.

        Returns:
            bool: True if the head is open, False otherwise.
        """
        return self.stand_mixer.get_state(env)["head"] > 0.99


class CloseStandMixerHead(LwLabTaskBase):
    layout_registry_names: list[int] = [FixtureType.STAND_MIXER]

    task_name: str = "CloseStandMixerHead"
    enable_fixtures: list[str] = ["stand_mixer"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stand_mixer = self.register_fixture_ref("stand_mixer", dict(id=FixtureType.STAND_MIXER))
        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the stand mixer head."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.stand_mixer.set_head_pos(env)

    def _check_success(self, env):
        """
        Check if the stand mixer head is closed.

        Returns:
            bool: True if the head is closed, False otherwise.
        """
        return self.stand_mixer.get_state(env)["head"] < 0.01
