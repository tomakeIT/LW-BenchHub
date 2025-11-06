import copy
from lwlab.core.tasks.base import LwLabTaskBase
import re
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from .libero_10_put_in_basket import Libero10PutInBasket


class L90L1PickUpTheKetchupAndPutItInTheBasket(Libero10PutInBasket):
    """
    L90L1PickUpTheKetchupAndPutItInTheBasket: pick up the ketchup and put it in the basket

    Steps:
        pick up the ketchup
        put the ketchup in the basket

    """

    task_name: str = "L90L1PickUpTheKetchupAndPutItInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the ketchup, and put it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        pop_objs = [self.milk, self.orange_juice, self.butter]
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] in pop_objs:
                cfgs.pop(cfg_index)
            else:
                cfg_index += 1
        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.ketchup,
            self.basket,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )


class L90L1PickUpTheCreamCheeseBoxAndPutItInTheBasket(L90L1PickUpTheKetchupAndPutItInTheBasket):
    """
    L90L1PickUpTheCreamCheeseBoxAndPutItInTheBasket: pick up the cream cheese box and put it in the basket

    Steps:
        pick up the cream cheese box
        put the cream cheese box in the basket

    """

    task_name: str = "L90L1PickUpTheCreamCheeseBoxAndPutItInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the cream cheese box, and put it in the basket."
        return ep_meta

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.cream_cheese,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )


class L90L2PickUpTheTomatoSauceAndPutItInTheBasket(Libero10PutInBasket):
    """
    L90L2PickUpTheTomatoSauceAndPutItInTheBasket: pick up the tomato sauce and put it in the basket

    Steps:
        pick up the tomato sauce
        put the tomato sauce in the basket

    """

    task_name: str = "L90L2PickUpTheTomatoSauceAndPutItInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Pick up the tomato sauce, and put it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        pop_objs = []
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] in pop_objs:
                cfgs.pop(cfg_index)
            else:
                cfg_index += 1
        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.tomato_sauce,
            self.basket,
            th_z_axis_cos=0.0,
            th_xy_dist=0.4,
            th_xyz_vel=0.5,
        )


class L90L2PickUpTheAlphabetSoupAndPutItInTheBasket(Libero10PutInBasket):
    """
    L90L2PickUpTheAlphabetSoupAndPutItInTheBasket: pick up the alphabet soup and put it in the basket

    Steps:
        pick up the alphabet soup
        put the alphabet soup in the basket

    """

    task_name: str = "L90L2PickUpTheAlphabetSoupAndPutItInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Pick up the alphabet soup, and put it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        pop_objs = []
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] in pop_objs:
                cfgs.pop(cfg_index)
            else:
                cfg_index += 1
        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.alphabet_soup,
            self.basket,
            th_z_axis_cos=0.0,
            th_xy_dist=0.4,
            th_xyz_vel=0.5,
        )


class L90L2PickUpTheOrangeJuiceAndPutItInTheBasket(Libero10PutInBasket):
    """
    L90L2PickUpTheOrangeJuiceAndPutItInTheBasket: pick up the orange juice and put it in the basket

    Steps:
        pick up the orange juice
        put the orange juice in the basket

    """

    task_name: str = "L90L2PickUpTheOrangeJuiceAndPutItInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Pick up the orange juice, and put it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        pop_objs = []
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] in pop_objs:
                cfgs.pop(cfg_index)
            else:
                cfg_index += 1
        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.orange_juice,
            self.basket,
            th_z_axis_cos=0.0,
            th_xy_dist=0.4,
            th_xyz_vel=0.5,
        )


class L90L1PickUpTheTomatoSauceAndPutItInTheBasket(Libero10PutInBasket):
    """
    L90L1PickUpTheTomatoSauceAndPutItInTheBasket: pick up the tomato sauce and put it in the basket (Scene1 with limited objects)

    Steps:
        pick up the tomato sauce
        put the tomato sauce in the basket

    Scene1 objects: basket, tomato_sauce, ketchup, cream_cheese, alphabet_soup
    """

    task_name: str = "L90L1PickUpTheTomatoSauceAndPutItInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Pick up the tomato sauce, and put it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        pop_objs = [self.milk, self.orange_juice, self.butter]
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] in pop_objs:
                cfgs.pop(cfg_index)
            else:
                cfg_index += 1
        return cfgs

    def _check_success(self, env):
        success_tomato_sauce = OU.check_place_obj1_on_obj2(
            env,
            self.tomato_sauce,
            self.basket,
            th_z_axis_cos=0.0,
            th_xy_dist=0.4,
            th_xyz_vel=0.5,
        )
        return success_tomato_sauce
