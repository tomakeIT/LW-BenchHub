from setuptools import setup

setup(
    packages=[
        "lwlab",
        "policy",
        "tasks",
        "lwlab_rl"
    ],
    package_dir={
        "lwlab": "lwlab",
        "lwlab_policy": "lwlab_policy",
        "tasks": "tasks"
    }
)
