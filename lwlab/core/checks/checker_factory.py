from lwlab.core.checks.motion_checker import MotionChecker
from lwlab.core.checks.kitchen_coffee_collision_checker import KitchenCoffeeCollisionChecker
from lwlab.core.checks.gripper_collision_checker import GripperCollisionChecker
from lwlab.core.checks.kitchen_clipping_checker import KitchenClippingChecker
from lwlab.core.checks.actuator_velocity_jump_checker import VelocityJumpChecker
from lwlab.core.checks.start_object_move_checker import StartObjectMoveChecker


CHECKER_REGISTRY = {
    "motion": MotionChecker,
    "kitchen_coffee_collision": KitchenCoffeeCollisionChecker,
    "gripper_collision": GripperCollisionChecker,
    "kitchen_clipping": KitchenClippingChecker,
    "velocity_jump": VelocityJumpChecker,
    "start_object_move": StartObjectMoveChecker,
}


def get_checker(checker_type):
    if checker_type not in CHECKER_REGISTRY:
        raise ValueError(f"Checker type {checker_type} not found")
    return CHECKER_REGISTRY[checker_type]


def get_checkers_from_cfg(checkers_cfg):
    checkers = []
    for checker_type in checkers_cfg.keys():
        checker_cfg = checkers_cfg[checker_type]
        checker = get_checker(checker_type)
        checkers.append(checker(warning_on_screen=checker_cfg.get("warning_on_screen", False)))
    return checkers


def form_checker_result(checkers_cfg):
    checkers_results = {}
    for check_type in checkers_cfg.keys():
        checkers_results[check_type] = {}
    return checkers_results
