import copy
import re
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L90L4PickUpTheSaladDressingAndPutItInTheTray(LiberoEnvCfg, BaseTaskEnvCfg):
    """
    L90L4PickUpTheSaladDressingAndPutItInTheTray: pick up the salad dressing and put it in the tray
    """

    task_name: str = "L90L4PickUpTheSaladDressingAndPutItInTheTray"

    enable_fixtures = ["saladdressing"]
    removable_fixtures = enable_fixtures

    def __post_init__(self):
        self.activate_contact_sensors = False
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.salad_dressing = self.register_fixture_ref("saladdressing", dict(id=FixtureType.SALAD_DRESSING))
        self.init_robot_base_ref = self.counter
        self.akita_black_bowl = "akita_black_bowl"
        self.chocolate_pudding = "chocolate_pudding"
        self.wooden_tray = "wooden_tray"

    def _setup_scene(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env_ids)

    def _reset_internal(self, env_ids):
        super()._reset_internal(env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the salad dressing and put it in the tray."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos=(0.0, -1), size=(0.8, 0.8)):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                margin=0.02,
                ensure_valid_placement=True,
            )
        tray_placement = get_placement()
        akita_black_bowl_placement = get_placement()
        chocolate_pudding_placement = get_placement()

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None):
            if mjcf_path is not None:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        info=dict(mjcf_path=mjcf_path),
                        placement=placement,
                    )
                )
            else:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        placement=placement,
                    )
                )
        add_cfg(self.akita_black_bowl, self.akita_black_bowl, True, akita_black_bowl_placement, mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml")
        add_cfg(self.chocolate_pudding, self.chocolate_pudding, True, chocolate_pudding_placement, mjcf_path="/objects/lightwheel/chocolate_pudding/ChocolatePudding001/model.xml")
        add_cfg(self.wooden_tray, self.wooden_tray, True, tray_placement, mjcf_path="/objects/lightwheel/tray/Tray016/model.xml")

        return cfgs

    def _check_success(self):
        return OU.check_place_obj1_on_obj2(
            self.env,
            self.salad_dressing,
            self.wooden_tray,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.5,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
