import gymnasium as gym

gym.register(
    id="Robocasa-Task-TestObjectsTask",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.test_object:TestObjectsTask"},
)
