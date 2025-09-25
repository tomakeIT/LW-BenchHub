import gymnasium as gym


def register_rl_env(robot_name, task_name, env_cfg_entry_point, skrl_cfg_entry_point, rsl_rl_cfg_entry_point, variant=None):
    if variant:
        task_name = f"{task_name}-{variant}"
    gym.register(
        id=f"Robocasa-Rl-{robot_name}-{task_name}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "skrl_cfg_entry_point": skrl_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
        },
        disable_env_checker=True,
    )
