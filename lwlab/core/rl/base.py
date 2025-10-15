from lwlab.core.cfg import LwBaseCfg
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from lwlab.core import mdp
from lwlab.utils.env import ExecuteMode


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out_ = DoneTerm(func=mdp.time_out, time_out=True)

    # joint_vel_limit = DoneTerm(func=mdp.joint_vel_out_of_limit, time_out=True)


class BaseRLEnvCfg(LwBaseCfg):
    task_type: str = "rl"
    terminations: TerminationsCfg = TerminationsCfg()

    def _check_success(self):
        # if self.context.execute_mode == ExecuteMode.TRAIN:
        #     # check_succuss only work in eval mode
        #     # no need to run in train mode
        #     return False
        # else:
        return super()._check_success()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 4.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
