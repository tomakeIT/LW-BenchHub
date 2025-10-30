import copy
from lwlab.core.tasks.base import LwLabTaskBase
import re
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class Libero10PutInBasket(LwLabTaskBase):
    """
    Libero10PutInBasket: base class for all libero 10 put in basket tasks
    """

    task_name: str = "Libero10PutInBasket"

    enable_fixtures = ['ketchup']
    removable_fixtures = enable_fixtures

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.ketchup = self.register_fixture_ref("ketchup", dict(id=FixtureType.KETCHUP, ref=self.counter))

        self.init_robot_base_ref = self.counter
        self.place_success = {}
        self.alphabet_soup = "alphabet_soup"
        self.basket = "basket"
        self.butter = "butter"
        self.cream_cheese = "cream_cheese"
        self.milk = "milk"
        self.orange_juice = "orange_juice"
        self.tomato_sauce = "tomato_sauce"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _reset_internal(self, env_ids):
        super()._reset_internal(env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos=(0.0, -0.7), size=(0.5, 0.5)):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                ensure_valid_placement=True,
            )
        basket_placement = get_placement(pos=(0.7, -0.6), size=(0.6, 0.6))
        alphabet_soup_placement = get_placement(pos=(0.5, -0.8), size=(0.2, 0.2))
        butter_placement = get_placement()
        cream_cheese_placement = get_placement()
        # ketchup_placement = get_placement()
        milk_placement = get_placement()
        orange_juice_placement = get_placement()
        tomato_sauce_placement = get_placement(pos=(0.3, -0.8), size=(0.2, 0.2))

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None, object_scale=1.0):
            if mjcf_path is not None:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        object_scale=object_scale,
                        info=dict(mjcf_path=mjcf_path),
                        placement=placement,
                    )
                )
            else:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        object_scale=object_scale,
                        graspable=graspable,
                        placement=placement,
                    )
                )
        add_cfg(self.alphabet_soup, self.alphabet_soup, True, alphabet_soup_placement, mjcf_path="/objects/lightwheel/alphabet_soup/AlphabetSoup001/model.xml", object_scale=0.8)
        add_cfg(self.basket, self.basket, True, basket_placement, mjcf_path="/objects/lightwheel/basket/Basket058/model.xml")
        add_cfg(self.butter, self.butter, True, butter_placement, mjcf_path="/objects/lightwheel/butter/Butter001/model.xml")
        add_cfg(self.cream_cheese, self.cream_cheese, True, cream_cheese_placement, object_scale=0.2, mjcf_path="/objects/lightwheel/cream_cheese_stick/CreamCheeseStick013/model.xml")
        add_cfg(self.milk, self.milk, True, milk_placement, mjcf_path="/objects/lightwheel/milk_drink/MilkDrink009/model.xml")
        add_cfg(self.orange_juice, self.orange_juice, True, orange_juice_placement, mjcf_path="/objects/lightwheel/orange_juice/OrangeJuice001/model.xml")
        add_cfg(self.tomato_sauce, self.tomato_sauce, True, tomato_sauce_placement, mjcf_path="/objects/lightwheel/ketchup/Ketchup003/model.xml", object_scale=0.8)
        return cfgs

    # def _load_model(self):
    #     super()._load_model()
    #     placement_obj = self.plate
    #     obj_obj = self.akita_black_bowl
    #     # place obj on the placement
    #     z_offset = 0.1
    #     placement_placement = self.object_placements[placement_obj]
    #     obj_placement = list(self.object_placements[obj_obj])

    #     placement_pos = list(placement_placement[0])
    #     obj_pos = copy.deepcopy(placement_pos)
    #     obj_pos[2] += z_offset
    #     obj_placement[0] = tuple(obj_pos)
    #     self.object_placements[obj_obj] = tuple(obj_placement)


class L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket(Libero10PutInBasket):
    """
    L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket: put both the cream cheese box and the butter in the basket

    Steps:
        pick up the cream cheese box
        put the cream cheese box in the basket
        pick up the butter
        put the butter in the basket

    """

    task_name: str = "L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the cream cheese box and the butter, and put them in the basket."
        return ep_meta

    def _check_success(self, env):
        success_cream_cheese = OU.check_place_obj1_on_obj2(
            env,
            self.cream_cheese,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        success_butter = OU.check_place_obj1_on_obj2(
            env,
            self.butter,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        print(f"cream cheese success state: {success_cream_cheese}")
        print(f"butter success state: {success_butter}")
        return success_cream_cheese & success_butter


class L10L2PutBothTheAlphabetSoupAndTheTomatoSauceInTheBasket(Libero10PutInBasket):
    """
    L10L2PutBothTheAlphabetSoupAndTheTomatoSauceInTheBasket: put both the alphabet soup and the tomato sauce in the basket

    Steps:
        pick up the cream cheese box
        put the cream cheese box in the basket
        pick up the butter
        put the butter in the basket

    """

    task_name: str = "L10L2PutBothTheAlphabetSoupAndTheTomatoSauceInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the alphabet soup and the tomato sauce, and put them in the basket."
        return ep_meta

    def _check_success(self, env):
        success_alphabet_soup = OU.check_place_obj1_on_obj2(
            env,
            self.alphabet_soup,
            self.basket,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        success_tomato_sauce = OU.check_place_obj1_on_obj2(
            env,
            self.tomato_sauce,
            self.basket,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        print(f"alphabet soup success state: {success_alphabet_soup}")
        print(f"tomato sauce success state: {success_tomato_sauce}")
        return success_alphabet_soup & success_tomato_sauce


class L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket(Libero10PutInBasket):
    """
    L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket: put both the alphabet soup and the cream cheese box in the basket

    Steps:
        pick up the cream cheese box
        put the cream cheese box in the basket
        pick up the butter
        put the butter in the basket

    """

    task_name: str = "L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the alphabet soup and the cream cheese box, and put them in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        pop_objs = [self.butter, self.milk, self.orange_juice]
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] in pop_objs:
                cfgs.pop(cfg_index)
            else:
                cfg_index += 1
        return cfgs

    def _check_success(self, env):
        success_cream_cheese = OU.check_place_obj1_on_obj2(
            env,
            self.cream_cheese,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        success_alphabet_soup = OU.check_place_obj1_on_obj2(
            env,
            self.alphabet_soup,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        print(f"cream cheese success state: {success_cream_cheese}")
        print(f"alphabet soup success state: {success_alphabet_soup}")
        return success_cream_cheese & success_alphabet_soup
