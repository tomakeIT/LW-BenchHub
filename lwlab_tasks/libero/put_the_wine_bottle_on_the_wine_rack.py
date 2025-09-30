import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90K4PutTheWineBottleOnTheWineRack(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = "L90K4PutTheWineBottleOnTheWineRack"
    enable_fixtures = ["winerack", "storage_furniture"]

    def __post_init__(self):
        self.obj_name = []
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )
        self.winerack = self.register_fixture_ref(
            "winerack",
            dict(id=FixtureType.WINE_RACK),
        )
        self.init_robot_base_ref = self.dining_table

    def _load_model(self):
        super()._load_model()
        for cfg in self.object_cfgs:
            self.obj_name.append(cfg["name"])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"put the wine bottle on the wine rack."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"akita_black_bowl",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml"
                ),
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.35),
                    margin=0.02,
                    pos=(-0.3, -0.7),
                    ensure_object_boundary_in_range=False
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"wine_bottle",
                obj_groups=["bottle"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/bottle/Bottle054/model.xml"
                ),
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.35),
                    margin=0.02,
                    pos=(0.1, -0.6),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        wine_bottle_pos = OU.get_object_pos(self.env, "wine_bottle")
        ret = OU.point_in_fixture(wine_bottle_pos, self.winerack, only_2d=True)
        ret_tensor = torch.tensor(ret, dtype=torch.bool, device="cpu").repeat(self.env.num_envs)
        return ret_tensor & OU.gripper_obj_far(self.env, "wine_bottle", th=0.35)
