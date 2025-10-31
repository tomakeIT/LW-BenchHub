def rl_on(task=None, embodiment=None):
    from lwlab.core.tasks.base import LwLabTaskBase
    from lwlab.core.robots.robot_arena_base import LwLabEmbodimentBase

    if task is not None:
        if not issubclass(task, LwLabTaskBase):
            raise TypeError(f"task must be a subclass of LwLabTaskBase, got {type(task)}")

    if embodiment is not None:
        if not issubclass(embodiment, LwLabEmbodimentBase):
            raise TypeError(f"embodiment must be a subclass of LwLabEmbodimentBase, got {type(embodiment)}")

    def wrapper(cls):
        if task:
            cls._rl_on_tasks.append(task)
        if embodiment:
            cls._rl_on_embodiments.append(embodiment)
        return cls

    return wrapper
