from .put_object_in_basket import PutObjectInBasket
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class LOPickUpTheBbqSauceAndPlaceItInTheBasket(PutObjectInBasket):

    task_name: str = f"LOPickUpTheBbqSauceAndPlaceItInTheBasket"
    enable_fixtures: list[str] = ["saladdressing", "ketchup", "bbq_sauce"]
    EXCLUDE_LAYOUTS: list = [63, 64]
    removable_fixtures: list[str] = ["saladdressing", "ketchup", "bbq_sauce"]

    def __post_init__(self):
        super().__post_init__()
        self.salad_dressing = self.register_fixture_ref("saladdressing", dict(id=FixtureType.SALAD_DRESSING))
        self.ketchup = self.register_fixture_ref("ketchup", dict(id=FixtureType.KETCHUP))
        self.bbq_sauce = self.register_fixture_ref("bbq_sauce", dict(id=FixtureType.BBQ_SOURCE))

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.alphabet_soup = "alphabet_soup"
        self.chocolate_pudding = "chocolate_pudding"
        self.cream_cheese_stick = "cream_cheese_stick"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick the bbq sauce and place it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.alphabet_soup,
                obj_groups=self.alphabet_soup,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.3),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/alphabet_soup/AlphabetSoup001/model.xml",
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
                name=self.cream_cheese_stick,
                obj_groups=self.cream_cheese_stick,
                graspable=True,
                object_scale=0.2,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/cream_cheese_stick/CreamCheeseStick013/model.xml",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        '''
        Check if the bbq_sauce is placed in the basket.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(self.env, self.bbq_sauce.name)
        fixture_in_basket = OU.check_fixture_in_receptacle(self.env, "bbq_sauce", self.bbq_sauce.name, self.basket)
        return fixture_in_basket & is_gripper_obj_far
