from lwlab.core.checks.base_checker import BaseChecker
import torch
from lwlab.core.models.fixtures import FixtureType
import omni
from isaaclab.sensors import ContactSensorCfg


class KitchenClippingChecker(BaseChecker):
    type = "kitchen_clipping"

    def __init__(self, warning_on_screen=False):
        super().__init__(warning_on_screen)
        self._init_state()

    def _init_state(self, w=5, spike_th=100, mean_force_th=150, stable_var_th=5.0):
        self._clipping_warning_frame_count = 0
        self._clipping_warning_text = ""
        self._clipping_counts = 0
        self._clipping_frames = []
        self.forces = []
        self.spike_th = spike_th
        self.mean_force_th = mean_force_th
        self.stable_var_th = stable_var_th
        self.w = w
        self.last_force = 0
        self._frame_count = 0
        self._force_events = []
        self.history_metrics = {
            "force": [],
            "mean_force": [],
            "force_variance": [],
            "delta_force": [],
            "clipping_times": [],
            "clipping_frames": [],
        }

    def reset(self):
        self._init_state()

    def _check(self, env):
        return self._check_clipping(env)

    def _check_clipping(self, env):
        """
        Check if the clipping happens.
        Calculates the clipping status.

        Returns:
            dict: Dictionary containing total clipping times.
        """
        if self._clipping_counts is None:
            self._clipping_counts = 0

        if self._clipping_warning_frame_count is not None and 50 > self._clipping_warning_frame_count > 0:
            self._clipping_warning_frame_count += 1
        else:
            self._clipping_warning_frame_count = 0
            self._clipping_warning_text = ""
        try:

            left_gripper = "left_gripper"
            right_gripper = "right_gripper"

            self._frame_count += 1

            current_frame = self._frame_count

            left_force = env.cfg.calculate_contact_force(left_gripper)
            right_force = env.cfg.calculate_contact_force(right_gripper)
            # Handle both scalar and multi-environment tensors
            if left_force.dim() == 0:
                left_force_val = left_force.item()
            else:
                left_force_val = left_force[0].item()

            if right_force.dim() == 0:
                right_force_val = right_force.item()
            else:
                right_force_val = right_force[0].item()

            force = max(left_force_val, right_force_val)

            self.forces.append(force)
            if len(self.forces) > self.w:
                self.forces.pop(0)

            mu = sum(self.forces) / len(self.forces)
            var = sum((x - mu)**2 for x in self.forces) / len(self.forces)

            deltaF = abs(force - getattr(self, "last_force", 0.0))
            self.last_force = force

            # print(f"force: {force}, mean: {mu}, var: {var}, deltaF: {deltaF}")

            last_event = self._force_events[-1] if self._force_events else None
            if (last_event is None) or (last_event.get("end") is not None):
                if force > 2:
                    self._force_events.append({"start": current_frame, "end": None, "triggered": False, "candidates": []})
            else:
                if force < 2 and last_event.get("end") is None:
                    last_event["end"] = current_frame
                    for f in last_event["candidates"]:
                        if last_event["start"] + 100 <= f <= last_event["end"] - 70:
                            if (self._clipping_warning_frame_count == 0):
                                self._clipping_warning_text = "kitchen_clipping Warning: Contact forces too high, there may be << Clipping >> happens"
                                self._clipping_counts += 1
                                self._clipping_frames.append(current_frame)
                                self._clipping_warning_frame_count += 1
                                last_event["triggered"] = True
                                break
                    last_event["candidates"] = []

            condition_now = (force > 100 or deltaF > self.spike_th or var > self.stable_var_th)

            for ev in self._force_events:
                if ev.get("triggered"):
                    continue

                start = ev["start"]
                end = ev.get("end")
                window_start = start + 70

                if condition_now and current_frame >= window_start:
                    if end is not None:
                        if current_frame <= end - 70:
                            if not ev["candidates"] or ev["candidates"][-1] != current_frame:
                                ev["candidates"].append(current_frame)
                        else:
                            pass
                    else:
                        if not ev["candidates"] or ev["candidates"][-1] != current_frame:
                            ev["candidates"].append(current_frame)

                if ev["candidates"]:
                    if end is None:
                        f = ev["candidates"][0]
                        if current_frame >= f + 70:
                            if self._clipping_warning_frame_count == 0:
                                self._clipping_warning_text = "kitchen_clipping Warning: Contact forces too high, there may be << Clipping >> happens"
                                self._clipping_counts += 1
                                self._clipping_frames.append(current_frame)
                                self._clipping_warning_frame_count += 1
                                ev["triggered"] = True
                                break
                            else:
                                ev["triggered"] = True
                                break
                    else:
                        for f in ev["candidates"]:
                            if start + 70 <= f <= end - 70:
                                if self._clipping_warning_frame_count == 0:
                                    self._clipping_warning_text = "kitchen_clipping Warning: Contact forces too high, there may be << Clipping >> happens"
                                    self._clipping_counts += 1
                                    self._clipping_frames.append(current_frame)
                                    self._clipping_warning_frame_count += 1
                                    ev["triggered"] = True
                                    break
                                else:
                                    ev["triggered"] = True
                                    break
                        if ev.get("end") is not None:
                            ev["candidates"] = []

            if len(self._force_events) > 500:
                self._force_events = self._force_events[-200:]

            if self._clipping_counts > 0:
                success = False
            else:
                success = True

            self.history_metrics["force"].append(float(force))
            self.history_metrics["mean_force"].append(float(mu))
            self.history_metrics["force_variance"].append(float(var))
            self.history_metrics["delta_force"].append(float(deltaF))
            self.history_metrics["clipping_times"] = int(self._clipping_counts)
            self.history_metrics["clipping_frames"] = list(self._clipping_frames)
            self.history_metrics["success"] = success

            result = {
                "success": success,
                "warning_text": self._clipping_warning_text,
                "metrics": self.history_metrics
            }

            return result
        except Exception as e:
            import traceback
            print(f"Error in _check_clipping: {traceback.format_exc()}")
            return {"success": False, "warning_text": None, "metrics": {}}
