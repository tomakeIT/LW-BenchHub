import gymnasium as gym

gym.register(
    id="Local-Task-BaseTask",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_task:BaseTask",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-RobocasaBaseTask",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_task:RobocasaBaseTask",
    },
    disable_env_checker=True,
)
