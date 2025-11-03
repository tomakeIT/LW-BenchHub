from isaaclab.utils import dataclass
from dataclasses import field
from lwlab.utils.env import ExecuteMode

CURRENT_CONTEXT = None


@dataclass
class Context:
    scene_name: str | None = None
    robot_name: str | None = None
    task_name: str | None = None
    execute_mode: ExecuteMode | None = None
    device: str | None = None
    ep_meta: dict | None = field(default_factory=dict)
    robot_scale: float = 1.0
    first_person_view: bool = False
    enable_cameras: bool = False
    usd_simplify: bool = False
    object_init_offset: list[float] = field(default_factory=lambda: [0.0, 0.0])
    max_scene_retry: int = 5
    max_object_placement_retry: int = 3
    seed: int | None = None
    sources: list[str] | None = None
    object_projects: list[str] | None = None
    headless_mode: bool = False
    extra_params: dict | None = None
    replay_cfgs: dict | None = field(default_factory=dict)
    resample_objects_placement_on_reset: bool | None = None
    resample_robot_placement_on_reset: bool | None = None
    num_envs: int | None = 1
    device: str | None = "cpu"
    use_fabric: bool | None = None
    add_camera_to_observation: bool = False


def get_context():
    global CURRENT_CONTEXT
    if CURRENT_CONTEXT is None:
        CURRENT_CONTEXT = Context()
    return CURRENT_CONTEXT
