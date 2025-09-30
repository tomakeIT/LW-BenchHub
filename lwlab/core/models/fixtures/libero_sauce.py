from lwlab.core.models.fixtures.fixture import Fixture
from lwlab.core.models.fixtures.fixture_types import FixtureType


class BBQSauce(Fixture):
    """
    bbq_sauce fixture class
    """
    fixture_types = [FixtureType.BBQ_SOURCE]


class SaladDressing(Fixture):
    """
    saladdressing fixture class
    """
    fixture_types = [FixtureType.SALAD_DRESSING]


class Ketchup(Fixture):
    """
    ketchup fixture class
    """
    fixture_types = [FixtureType.KETCHUP]
