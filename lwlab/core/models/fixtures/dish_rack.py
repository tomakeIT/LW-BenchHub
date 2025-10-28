from lwlab.core.models.fixtures.fixture import Fixture
from .fixture_types import FixtureType


class DishRack(Fixture):
    fixture_types = [FixtureType.DISH_RACK]

    def get_reset_region_names(self):
        return ("int1", "int2", "int3", "int4", "int5", "int6", "int7", "int9")


8
