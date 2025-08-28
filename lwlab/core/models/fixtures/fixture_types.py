from enum import IntEnum


class FixtureType(IntEnum):
    """
    Enum for fixture types in lwlab kitchen environments.
    """

    MICROWAVE = 1  # DONE
    STOVE = 2  # DONE
    OVEN = 3  # DONE
    SINK = 4  # DONE
    COFFEE_MACHINE = 5  # DONE
    TOASTER = 6  # DONE
    TOASTER_OVEN = 7  # DONE
    FRIDGE = 8  # DONE
    DISHWASHER = 9  # DONE
    BLENDER = 10  # DONE
    STAND_MIXER = 11  # DONE
    ELECTRIC_KETTLE = 12  # DONE
    STOOL = 13  # DONE
    COUNTER = 14  # DONE
    ISLAND = 15
    COUNTER_NON_CORNER = 16  # DONE
    DINING_COUNTER = 17  # DONE
    CABINET = 18  # DONE
    CABINET_WITH_DOOR = 19  # DONE
    CABINET_SINGLE_DOOR = 20  # DONE
    CABINET_DOUBLE_DOOR = 21  # DONE
    SHELF = 22
    DRAWER = 23
    TOP_DRAWER = 24
    WINDOW = 25  # DONE
    DISH_RACK = 26  # DONE
    COUNTER_NON_DINING = 27
