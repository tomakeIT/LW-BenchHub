import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class ArrangeVegetables(LwLabTaskBase):
    """
    Arrange Vegetables: composite task for Chopping Food activity.

    Simulates the task of arranging vegetables on the cutting board.

    Steps:
        Take the vegetables from the sink and place them on the cutting board.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "ArrangeVegetables"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.45, 0.55))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the vegetables from the sink and place them on the cutting board."
        return ep_meta

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.45, 0.55)
                    ),
                    size=(0.35, 0.45),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.45, 0.55)
                    ),
                    size=(0.45, 0.45),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.20),
                    pos=(-1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.20),
                    pos=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success_veg1(self, env):
        return OU.check_obj_in_receptacle(env, "vegetable1", "cutting_board")

    def _check_success_veg2(self, env):
        return OU.check_obj_in_receptacle(env, "vegetable2", "cutting_board")

    def _check_success(self, env):
        vegetable1_cutting_board_contact = OU.check_obj_in_receptacle(
            env, "vegetable1", "cutting_board"
        )
        vegetable2_cutting_board_contact = OU.check_obj_in_receptacle(
            env, "vegetable2", "cutting_board"
        )
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="cutting_board")

        return (
            vegetable1_cutting_board_contact
            & vegetable2_cutting_board_contact
            & gripper_obj_far
        )


class ArrangeVegetablesSimple(LwLabTaskBase):
    """
    Arrange Vegetables: composite task for Chopping Food activity.

    Simulates the task of arranging vegetables on the cutting board.

    Steps:
        Take the vegetables from the sink and place them on the cutting board.
    """

    task_name: str = "ArrangeVegetablesSimple"
    layout_and_style_ids: list[tuple[int, int]] = [(0, 1)]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.45, 0.55))
        )
        self.init_robot_base_ref = self.sink

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the vegetables from the sink and place them on the cutting board."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="right", top_size=(0.30, 0.55)
                    ),
                    size=(0.35, 0.45),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left", top_size=(0.45, 0.55)
                    ),
                    size=(0.45, 0.45),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="tomato",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.2, 0.20),
                    pos=(-0.5, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="cucumber",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.2, 0.20),
                    pos=(0.5, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        vegetable1_cutting_board_contact = OU.check_obj_in_receptacle(
            env, "vegetable1", "cutting_board"
        )
        vegetable2_cutting_board_contact = OU.check_obj_in_receptacle(
            env, "vegetable2", "cutting_board"
        )
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="cutting_board")

        return (
            vegetable1_cutting_board_contact
            & vegetable2_cutting_board_contact
            & gripper_obj_far
        )


class ArrangeVegetablesSimpleV2(LwLabTaskBase):
    """
    Arrange Vegetables: composite task for Chopping Food activity.

    Simulates the task of arranging vegetables on the cutting board.

    Steps:
        Take the vegetables from the sink and place them on the cutting board.
    """

    layout_ids: list[int] = [0, 2, 5]
    style_ids: list[int] = [1, 6, 3]
    task_name: str = "ArrangeVegetablesSimpleV2"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.45, 0.55))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick the vegetables from the sink and place them on the cutting board"
        return ep_meta

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.30, 0.55)
                    ),
                    size=(0.35, 0.45),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left", top_size=(0.45, 0.55)
                    ),
                    size=(0.45, 0.45),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=["tomato", "avocado", "eggplant", "lemon", "lime"],
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.2, 0.20),
                    pos=(-0.5, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups=["cucumber", "carrot", "corn", "potato", "sweet_potato"],
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.2, 0.20),
                    pos=(0.5, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        vegetable1_cutting_board_contact = OU.check_obj_in_receptacle(
            env, "vegetable1", "cutting_board"
        )
        vegetable2_cutting_board_contact = OU.check_obj_in_receptacle(
            env, "vegetable2", "cutting_board"
        )
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="cutting_board")

        return (
            vegetable1_cutting_board_contact
            & vegetable2_cutting_board_contact
            & gripper_obj_far
        )
