import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class PanTransfer(LwLabTaskBase):
    """
    Pan Transfer: composite task for Serving Food activity.

    Simulates the task of transferring vegetables from a pan to a plate.

    Steps:
        Pick up the pan and dump the vegetables in it onto the plate.
        Then, return the pan to the stove.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.STOVE]
    task_name: str = "PanTransfer"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.stove
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the pan and dump the vegetables in it onto the plate. "
            "Then return the pan to the stove."
        )
        return ep_meta

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)
        self._robot_touched_food = False

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.stove,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="pan",
                    rotation=[(-3 * torch.pi / 8, -torch.pi / 4), (torch.pi / 4, 3 * torch.pi / 8)],
                ),
            )
        )
        # cfgs.append(dict(
        #     name="vegetable2",
        #     obj_groups="vegetable",
        #     placement=dict(
        #         size=(0.01, 0.01),
        #         ensure_object_boundary_in_range=False,
        #         sample_args=dict(
        #             reference="vegetable_container"
        #         )
        #     ),
        # ))

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                exclude_obj_groups=["plate", "pan", "vegetable"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.20),
                    pos=(0.5, 0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        vegetable_on_plate = OU.check_obj_in_receptacle(env, "vegetable", "plate")
        stove_locations = self.stove.check_obj_location_on_stove(env, "vegetable_container", need_knob_on=False)
        pan_on_stove = torch.tensor([loc is not None for loc in stove_locations], device=env.device)
        gripper_obj_far = OU.gripper_obj_far(env, "vegetable_container")
        return vegetable_on_plate & pan_on_stove & gripper_obj_far
