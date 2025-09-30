from lwlab.core.models.fixtures.fixture import Fixture
from lwlab.utils.usd_utils import OpenUsd as usd
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
import torch
from functools import cached_property
from lwlab.core.models.fixtures.fixture_types import FixtureType


class WineRack(Fixture):
    """Adapter fixture for wine rack to enable setup_removable like other fixtures."""

    fixture_types = [FixtureType.WINE_RACK]
