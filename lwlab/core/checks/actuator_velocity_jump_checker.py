from lwlab.core.checks.base_checker import BaseChecker
import torch


class VelocityJumpChecker(BaseChecker):
    type = "velocity_jump"

    def __init__(self, jump_threshold=0.3, warning_on_screen=False):
        super().__init__(warning_on_screen)
        self.jump_threshold = jump_threshold
        self._init_state()

    def _init_state(self):
        self._prev_body_poses = None
        self._prev_velocities = None
        self._velocity_jump_counts = {}
        self._velocity_jump_warning_frame_count = 0
        self._velocity_jump_warning_text = ""
        self._frame_count = 0

    def reset(self):
        self._init_state()

    def _check(self, env):
        return self._check_velocity_jump(env)

    def _check_velocity_jump(self, env):
        """
        Check if the velocity jump happens.
        Calculates the velocity jump status.

        Returns:
            dict: Dictionary containing total velocity jump times.
        """
        self.env = env
        jump_violations = {}
        self._frame_count += 1

        current_body_poses = self.env.scene.articulations['robot'].data.body_com_pose_w[0]
        body_names = self.env.scene.articulations['robot'].data.body_names

        if self._prev_body_poses is None:
            self._prev_body_poses = current_body_poses.clone()
            return {"success": True, "warning_text": "", "metrics": {}}

        velocities = []
        for i in range(len(body_names)):
            if i < len(current_body_poses) and i < len(self._prev_body_poses):
                pos_diff = current_body_poses[i][:3] - self._prev_body_poses[i][:3]
                velocity = torch.norm(pos_diff) / self.env.step_dt
                velocities.append(velocity)
            else:
                velocities.append(torch.tensor(0.0, device=current_body_poses.device))

        if self._prev_velocities is None:
            self._prev_velocities = velocities
            self._prev_body_poses = current_body_poses.clone()
            return {"success": True, "warning_text": "", "metrics": {}}

        fast_bodies = []

        for i, body_name in enumerate(body_names):
            curr_v = velocities[i]
            prev_v = self._prev_velocities[i]
            v_diff = abs(curr_v - prev_v)

            if v_diff > self.jump_threshold:
                jump_violations[body_name] = {
                    "velocity_jump": v_diff.item(),
                    "prev_velocity": prev_v.item(),
                    "curr_velocity": curr_v.item()
                }

                if body_name not in self._velocity_jump_counts:
                    self._velocity_jump_counts[body_name] = 0
                self._velocity_jump_counts[body_name] += 1

                if len(fast_bodies) < 3:
                    fast_bodies.append(body_name)

        if self._frame_count > 60:
            if fast_bodies and self._velocity_jump_warning_frame_count == 0:
                if len(fast_bodies) == 1:
                    self._velocity_jump_warning_text = f"velocity_jump Warning: Body <<{fast_bodies[0]}>> velocity jump too large"
                elif len(fast_bodies) == 2:
                    self._velocity_jump_warning_text = f"velocity_jump Warning: Bodies <<{fast_bodies[0]}>>, <<{fast_bodies[1]}>> velocity jump too large"
                else:
                    self._velocity_jump_warning_text = f"velocity_jump Warning: Bodies <<{fast_bodies[0]}>>, <<{fast_bodies[1]}>>, <<{fast_bodies[2]}>> velocity jump too large"
                self._velocity_jump_warning_frame_count = 1
            elif self._velocity_jump_warning_frame_count > 0:
                self._velocity_jump_warning_frame_count += 1
                if self._velocity_jump_warning_frame_count > 50:
                    self._velocity_jump_warning_text = ""
                    self._velocity_jump_warning_frame_count = 0

        self._prev_velocities = velocities
        self._prev_body_poses = current_body_poses.clone()

        success = len(jump_violations) == 0
        metrics = self.get_velocity_jump_metrics()
        metrics["success"] = success

        return {
            "success": success,
            "warning_text": self._velocity_jump_warning_text,
            "metrics": metrics
        }

    def get_velocity_jump_metrics(self):
        """
        Calculate the metrics for velocity jump.
        """
        metrics = {}
        if self._velocity_jump_counts:
            robot_metrics = {name: count for name, count in self._velocity_jump_counts.items() if count > 0}
            if robot_metrics:
                metrics["robot"] = robot_metrics
        return metrics
