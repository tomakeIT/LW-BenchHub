import torch
from functools import cached_property
from lwlab.core.models.fixtures.fixture import Fixture
import os
from lwlab.utils.usd_utils import OpenUsd as usd
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from lwlab.core.models.fixtures.fixture_types import FixtureType


class StorageFurniture(Fixture):
    fixture_types = [FixtureType.STORAGE_FURNITURE]

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_int_tupe = ("int0", "int1", "int2", "int3", "int4")

    def get_reset_region_names(self):
        return self.reg_int_tupe

    def set_target_reg_int(self, targe_regs):
        self.reg_int_tupe = targe_regs
