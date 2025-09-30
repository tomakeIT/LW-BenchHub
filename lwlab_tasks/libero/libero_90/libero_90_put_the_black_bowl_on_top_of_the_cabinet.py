import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L90K1PutTheBlackBowlOnTopOfTheCabinet(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = 'L90K1PutTheBlackBowlOnTopOfTheCabinet'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["storage_furniture"]

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the black bowl on top of the cabinet."
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.storage_furniture = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE))
        self.init_robot_base_ref = self.dining_table
        self.plate = "plate"
        self.bowl = "bowl"

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.0, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.5, -0.15),
                    ensure_valid_placement=True,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/plate/Plate012/model.xml",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return OU.check_place_obj1_on_obj2(self.env, self.bowl, self.storage_furniture)
