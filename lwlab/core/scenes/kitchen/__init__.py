import gymnasium as gym


gym.register(
    id="Robocasa-Scene-Robocasakitchen",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen:LwLabScene",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Scene-Libero",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero:LiberoEnvCfg",
    },
    disable_env_checker=True,
)
