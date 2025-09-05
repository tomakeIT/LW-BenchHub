import torch
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg


class RelJointPositionAction(RelativeJointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: "RelJointPositionActionCfg"

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._processed_actions = self._processed_actions * self._scale + self._offset


@configclass
class RelJointPositionActionCfg(RelativeJointPositionActionCfg):
    class_type = RelJointPositionAction
