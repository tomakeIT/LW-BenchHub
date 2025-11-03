import argparse
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from lwlab.core.rl.base import LwLabRL
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from isaaclab_arena.utils.configclass import combine_configclass_instances
from isaaclab.managers import ObservationGroupCfg as ObsGroup


class LwLabEnvBuilder(ArenaEnvBuilder):

    def __init__(self, arena_env: IsaacLabArenaEnvironment, args: argparse.Namespace, rl_env: LwLabRL):
        super().__init__(arena_env, args)
        self.rl_env = rl_env

    def orchestrate(self) -> None:
        super().orchestrate()
        if self.rl_env:
            self.rl_env.setup_env_config(self.arena_env.orchestrator)

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        super().modify_env_cfg(env_cfg)
        if self.rl_env:
            self.rl_env.modify_env_cfg(env_cfg)
        return env_cfg

    def compose_manager_cfg(self) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg = super().compose_manager_cfg()
        policy_observation_cfg = combine_configclass_instances(
            "PolicyObservationGroupCfg",
            self.arena_env.embodiment.get_policy_observation_cfg(),
            self.arena_env.task.get_policy_observation_cfg(),
            self.rl_env.get_policy_observation_cfg() if self.rl_env else None,
        )

        new_policy = type('PolicyObservationGroupCfg', (ObsGroup,), {})()
        for key, value in policy_observation_cfg.__dict__.items():
            if not key.startswith('_'):
                setattr(new_policy, key, value)
        env_cfg.observations.policy = new_policy
        return env_cfg
