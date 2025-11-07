import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType


class TestObjectsTask(LwLabTaskBase):
    """
    TestObjectsTask: composite task for testing objects.
    """
    task_name: str = "TestObjectsTask"
    reset_objects_enabled: bool = True

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Test objects"
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        if self.test_object_paths is not None:
            for obj_path in self.test_object_paths:
                cfgs.append(
                    dict(
                        name=obj_path.split('/')[-1].split('.')[0],
                        asset_name=obj_path,
                        load_from_local=True,
                        placement=dict(
                            fixture=self.counter,
                        ),
                    )
                )
        return cfgs

    def _check_success(self, env):
        return torch.tensor([False], device=env.device).repeat(env.num_envs)
