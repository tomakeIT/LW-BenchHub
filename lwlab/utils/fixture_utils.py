from lwlab.core.models.fixtures import FixtureType


def fixture_is_type(fixture, fixture_type):
    """
    Check if a fixture is of a certain type

    Args:
        fixture (Fixture): The fixture to check

        fixture_type (FixtureType): The type to check against
    """
    if fixture is None:
        return False
    elif isinstance(fixture, FixtureType):
        return fixture == fixture_type
    else:
        return fixture._is_fixture_type(fixture_type)


def is_fxtr_valid(env, fxtr, size):
    """
    checks if counter is valid for object placement by making sure it is large enough

    Args:
        fxtr (Fixture): fixture to check
        size (tuple): minimum size (x,y) that the counter region must be to be valid

    Returns:
        bool: True if fixture is valid, False otherwise
    """
    for region in fxtr.get_reset_regions(env).values():
        if region["size"][0] >= size[0] and region["size"][1] >= size[1]:
            return True
    return False
