from lwlab.core.cfg import LwBaseCfg
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from lwlab.core import mdp
from lwlab.utils.env import ExecuteMode

from lwlab.core.tasks.base import LwLabTaskBase
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from typing import List
from lwlab.core.robots.robot_arena_base import LwLabEmbodimentBase
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from lwlab.utils.isaaclab_utils import NoDeepcopyMixin
from isaaclab_arena.utils.configclass import combine_configclass_instances


@configclass
class RlBasePolicyObservationCfg(ObsGroup):
    """Observations for policy group."""
    pass


class LwLabRL(NoDeepcopyMixin):

    _rl_on_tasks: List[LwLabTaskBase] = []
    _rl_on_embodiments: List[LwLabEmbodimentBase] = []

    def __init__(self):
        self.rewards_cfg = None
        self.events_cfg = None
        self.curriculum_cfg = None
        self.commands_cfg = None
        self.policy_observation_cfg = RlBasePolicyObservationCfg()

    def _set_reward_joint_names(self, gripper_joint_names, arm_joint_names):
        pass

    def setup_env_config(self, orchestrator):
        assert type(orchestrator.task) in self._rl_on_tasks, f"task {type(orchestrator.task)} is not in {self._rl_on_tasks}"
        assert type(orchestrator.embodiment) in self._rl_on_embodiments, f"embodiment {type(orchestrator.embodiment)} is not in {self._rl_on_embodiments}"
        self._set_reward_joint_names(orchestrator.embodiment.reward_gripper_joint_names, orchestrator.embodiment.reward_arm_joint_names)
        # no rewards / curriculum / commands in task
        orchestrator.task.rewards_cfg = self.rewards_cfg
        orchestrator.task.curriculum_cfg = self.curriculum_cfg
        orchestrator.task.commands_cfg = self.commands_cfg
        # event / observation already in task
        if self.events_cfg:
            orchestrator.task.events_cfg = combine_configclass_instances(
                "EventCfg",
                orchestrator.task.events_cfg,
                self.events_cfg,
            )

        # TODO(xiaowei.song, 2025.10.24): only check success in eval mode, need verified
        # if orchestrator.task.context.execute_mode == ExecuteMode.TRAIN:
        #     orchestrator.task.termination_cfg.success = DoneTerm(
        #         func=lambda env: return torch.tensor([False], device=env.device).repeat(env.num_envs)
        #     )

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg):
        env_cfg.episode_length_s = 4.0
        env_cfg.viewer.eye = (-2.0, 2.0, 2.0)
        env_cfg.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.sim.physx.bounce_threshold_velocity = 0.2
        env_cfg.sim.physx.bounce_threshold_velocity = 0.01
        env_cfg.sim.physx.friction_correlation_distance = 0.00625
        env_cfg.decimation = 5
        env_cfg.episode_length_s = 3.2
        env_cfg.sim.dt = 0.01  # 100Hz
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 * 4
        env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 16 * 1024
        return env_cfg

    def get_policy_observation_cfg(self):
        return self.policy_observation_cfg
