from .put_object_in_basket import PutObjectInBasket
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
import torch


class LOPickUpTheButterAndPlaceItInTheBasket(PutObjectInBasket):

    task_name: str = f"LOPickUpTheButterAndPlaceItInTheBasket"
    enable_fixtures: list[str] = ["ketchup", "bbq_sauce"]
    removable_fixtures = enable_fixtures
    EXCLUDE_LAYOUTS: list = [63, 64]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.ketchup = self.register_fixture_ref("ketchup", dict(id=FixtureType.KETCHUP))
        self.bbq_sauce = self.register_fixture_ref("bbq_sauce", dict(id=FixtureType.BBQ_SOURCE))
        self.chocolate_pudding = "chocolate_pudding"
        self.orange_juice = "orange_juice"
        self.ketchup = "ketchup"
        self.butter = "butter"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick the butter and place it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.butter,
                obj_groups=self.butter,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.3),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/butter/Butter001/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/chocolate_pudding/ChocolatePudding001/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.orange_juice,
                obj_groups=self.orange_juice,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/orange_juice/OrangeJuice001/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups=self.ketchup,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/ketchup/Ketchup003/model.xml",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        '''
        Check if the butter is placed in the basket.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(self.env, self.butter)
        object_in_basket = OU.check_obj_in_receptacle(self.env, self.butter, self.basket)
        return object_in_basket & is_gripper_obj_far


class L90L2PickUpTheButterAndPutItInTheBasket(LiberoEnvCfg, BaseTaskEnvCfg):
    task_name: str = f"L90L2PickUpTheButterAndPutItInTheBasket"
    enable_fixtures = ["ketchup"]
    removable_fixtures = enable_fixtures

    def __post_init__(self):
        self.obj_name = []
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
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
        ] = f"pick up the butter and put it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"basket",
                obj_groups=["basket"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/basket/Basket058/model.xml"
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    margin=0.02,
                    pos=(-0.2, -0.8)
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"alphabet_soup",
                obj_groups=["alphabet_soup"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/alphabet_soup/AlphabetSoup001/model.xml"
                ),
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.3),
                    margin=0.02,
                    pos=(0.1, 0.1),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"butter",
                obj_groups=["butter"],
                graspable=True,
                washable=True,
                object_scale=0.6,
                info=dict(
                    mjcf_path="/objects/lightwheel/butter/Butter001/model.xml"
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.35),
                    margin=0.02,
                    pos=(0.1, -0.8)
                ),
            )
        )
        cfgs.append(
            dict(
                name="milk_drink",
                obj_groups="milk_drink",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.25, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/milk_drink/MilkDrink009/model.xml",
                ),
            )
        )
        cfgs.append(
            dict(
                name="orange_juice",
                obj_groups="orange_juice",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.25, 0.2),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/orange_juice/OrangeJuice001/model.xml",
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"tomato_sauce",
                obj_groups=["ketchup"],
                graspable=True,
                washable=True,
                object_scale=0.8,
                info=dict(
                    mjcf_path="/objects/lightwheel/ketchup/Ketchup003/model.xml"
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.35),
                    margin=0.02,
                    pos=(-0.1, -0.2)
                ),
            )
        )
        return cfgs

    def _check_success(self):
        '''
        Check if the butter is placed in the basket.
        '''

        far_from_objects = self._gripper_obj_farfrom_objects()

        obj_pos = torch.mean(self.env.scene.rigid_objects["butter"].data.body_com_pos_w, dim=1)  # (num_envs, 3)
        basket_pos = torch.mean(self.env.scene.rigid_objects["basket"].data.body_com_pos_w, dim=1)  # (num_envs, 3)

        xy_dist = torch.norm(obj_pos[:, :2] - basket_pos[:, :2], dim=-1)  # (num_envs,)
        object_in_basket_xy = xy_dist < 0.5

        object_stable = OU.check_object_stable(self.env, "butter", threshold=0.01)

        z_diff = obj_pos[:, 2] - basket_pos[:, 2]
        height_check = (z_diff > -0.05) & (z_diff < 0.02)

        return object_in_basket_xy & far_from_objects & object_stable & height_check

    def _gripper_obj_farfrom_objects(self):
        gripper_far_tensor = torch.tensor([True], device=self.env.device).repeat(self.env.num_envs)
        for obj_name in self.obj_name:
            gripper_far_tensor = gripper_far_tensor & OU.gripper_obj_far(self.env, obj_name)
        return gripper_far_tensor
