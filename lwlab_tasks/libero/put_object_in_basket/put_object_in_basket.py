import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType


class PutObjectInBasket(LiberoEnvCfg, BaseTaskEnvCfg):

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.floor = self.register_fixture_ref("floor", dict(id=FixtureType.FLOOR_LAYOUT))
        self.init_robot_base_ref = self.floor
        self.basket = "basket"

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.basket,
                obj_groups=self.basket,
                placement=dict(
                    fixture=self.floor,
                    size=(0.3, 0.25),
                    pos=(0.1, 0.0),
                    # ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/basket/Basket058/model.xml",
                ),
            )
        )

        return cfgs
