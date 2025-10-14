from isaaclab.utils import configclass
from lwlab.utils.env import ExecuteMode

CURRENT_CONTEXT = None


@configclass
class Context:
    execute_mode: ExecuteMode | None = None
    device: str | None = None


def get_context():
    global CURRENT_CONTEXT
    if CURRENT_CONTEXT is None:
        CURRENT_CONTEXT = Context()
    return CURRENT_CONTEXT
