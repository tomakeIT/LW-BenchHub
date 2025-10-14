from isaaclab.envs import ManagerBasedRLEnv

from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg
from lwlab.core.robots.base import BaseRobotCfg
from lwlab.core.tasks.base import BaseTaskEnvCfg


class BaseCompositionalEnvCfg(BaseRobotCfg, BaseTaskEnvCfg, RobocasaKitchenEnvCfg):
    env: ManagerBasedRLEnv
    num_envs: int
    device: str
