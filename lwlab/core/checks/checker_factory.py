from lwlab.core.checks.motion_checker import MotionChecker

CHECKER_REGISTRY = {
    "motion": MotionChecker,
}


def get_checker(checker_type):
    if checker_type not in CHECKER_REGISTRY:
        raise ValueError(f"Checker type {checker_type} not found")
    return CHECKER_REGISTRY[checker_type]


def get_checkers_from_cfg(checkers_cfg):
    checkers = []
    for checker_type in checkers_cfg:
        checker_cfg = checkers_cfg[checker_type]
        checker = get_checker(checker_type)
        checkers.append(checker(warning_on_screen=checker_cfg.get("warning_on_screen", False)))
    return checkers
