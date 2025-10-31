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


@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    joint_pos = ObsTerm(func=mdp.joint_pos)
    joint_vel = ObsTerm(func=mdp.joint_vel)
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
    ee_pose = ObsTerm(func=mdp.ee_pose)

    # cabinet_joint_pos = ObsTerm(
    #     func=mdp.joint_pos_rel,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["corpus_to_drawer_0_0"])},
    # )
    # cabinet_joint_vel = ObsTerm(
    #     func=mdp.joint_vel_rel,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["corpus_to_drawer_0_0"])},
    # )
    # rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)

    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


class LwLabRL(NoDeepcopyMixin):

    _rl_on_tasks: List[LwLabTaskBase] = []
    _rl_on_embodiments: List[LwLabEmbodimentBase] = []

    def __init__(self):
        self.rewards_cfg = None
        self.events_cfg = None
        self.curriculum_cfg = None
        self.commands_cfg = None
        self.policy_cfg = PolicyCfg()

    def setup_env_config(self, orchestrator):
        assert type(orchestrator.task) in self._rl_on_tasks, f"task {type(orchestrator.task)} is not in {self._rl_on_tasks}"
        assert type(orchestrator.embodiment) in self._rl_on_embodiments, f"embodiment {type(orchestrator.embodiment)} is not in {self._rl_on_embodiments}"
        # no rewards / curriculum / commands in task
        orchestrator.task.rewards_cfg = self.rewards_cfg
        orchestrator.task.curriculum_cfg = self.curriculum_cfg
        orchestrator.task.commands_cfg = self.commands_cfg
        # event / observation already in task
        if self.events_cfg:
            for key, value in self.events_cfg.__dict__.items():
                setattr(orchestrator.task.events_cfg, key, value)
        orchestrator.task.observation_cfg.policy = self.policy_cfg

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
