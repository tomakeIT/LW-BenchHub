import argparse
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from lwlab.core.rl.base import LwLabRL
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from isaaclab_arena.utils.configclass import combine_configclass_instances


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
