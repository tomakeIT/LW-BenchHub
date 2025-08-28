from lwlab.core.models.fixtures.fixture import Fixture
from .fixture_types import FixtureType


class DishRack(Fixture):
    fixture_types = [FixtureType.DISH_RACK]
