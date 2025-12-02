import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class PrewashFoodAssembly(LwLabTaskBase):
    """
    Prewash Food Assembly: composite task for Washing Fruits And Vegetables activity.

    Simulates the process of assembling fruits and vegetables in a bowl and
    prewashing them.

    Steps:
        Pick the fruit/vegetable from the cabinet and place it in the bowl. Then
        pick the bowl and place it in the sink. Then turn on the sink facuet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which food is picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "PrewashFoodAssembly"
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter_cab = self.register_fixture_ref(
            "counter_cab", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.counter_sink = self.register_fixture_ref(
            "counter_sink", dict(id=FixtureType.COUNTER, ref=self.sink)
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_name = OU.get_obj_lang(self, "food")
        ep_meta["lang"] = (
            f"Pick the {food_name} from the cabinet and place it in the bowl. "
            "Then pick the bowl and place it in the sink. Then turn on the sink facuet."
        )

        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _reset_internal(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env, env_ids)
        self.cab.set_door_state(min=0.90, max=1.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="food",
                obj_groups=["vegetable", "fruit"],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.10),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter_cab,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="bowl")
        food_in_bowl = OU.check_obj_in_receptacle(env, "food", "bowl")
        bowl_in_sink = OU.obj_inside_of(env, "bowl", self.sink)
        handle_state = self.sink.get_handle_state(env=env)
        water_on = handle_state["water_on"]
        return gripper_obj_far & food_in_bowl & bowl_in_sink & water_on
