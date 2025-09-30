import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L90K10PutTheButterAtTheBackInTheTopDrawerOfTheCabinetAndCloseIt(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = "L90K10PutTheButterAtTheBackInTheTopDrawerOfTheCabinetAndCloseIt"
    enable_fixtures = ["storage_furniture"]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )
        self.init_robot_base_ref = self.table

    def __post_init__(self):
        super().__post_init__()
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE, ref=self.table))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "put the butter at the back in the top drawer of the cabinet and close it."
        return ep_meta

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)
        self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
        self.drawer.set_joint_state(0.1, 0.2, self.env, [self.top_drawer_joint_name])

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="butter",
                obj_groups="butter",
                graspable=True,
                object_scale=0.8,
                info=dict(
                    mjcf_path="/objects/lightwheel/butter/Butter001/model.xml",
                ),
                placement=dict(
                    fixture=self.table,
                    size=(0.80, 0.3),
                    pos=(0.0, -0.6),
                ),
            )
        )

        cfgs.append(
            dict(
                name="chocolate_pudding",
                obj_groups="chocolate_pudding",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/chocolate_pudding/ChocolatePudding001/model.xml",
                ),
                placement=dict(
                    fixture=self.table,
                    size=(0.80, 0.3),
                    pos=(-0.2, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name="akita_black_bowl",
                obj_groups="bowl",
                graspable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml",
                ),
                placement=dict(
                    fixture=self.table,
                    size=(0.80, 0.3),
                    pos=(0.2, -0.6),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        gipper_success = OU.gripper_obj_far(self.env, "butter")
        butter_in_drawer = OU.obj_inside_of(self.env, "butter", self.drawer, partial_check=True)
        cabinet_closed = self.drawer.is_closed(self.env, [self.top_drawer_joint_name])
        return cabinet_closed & gipper_success & butter_in_drawer
