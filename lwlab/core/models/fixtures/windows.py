from .fixture import Fixture
from .accessories import WallAccessory
from .fixture_types import FixtureType


class WindowProcBase(Fixture):

    @property
    def nat_lang(self):
        return "windows"


class WindowProc(WindowProcBase):
    pass


class Window(WallAccessory):
    fixture_types = [FixtureType.WINDOW]

    @property
    def nat_lang(self):
        return "window"
