from setuptools import setup

setup(
    packages=[
        "lwlab",
        "policy",
        "lwlab_tasks",
        "lwlab_rl"
    ],
    package_dir={
        "lwlab": "lwlab",
        "lwlab_policy": "lwlab_policy",
        "lwlab_tasks": "lwlab_tasks",
    }
)
