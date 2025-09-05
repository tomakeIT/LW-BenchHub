class MotionMetric():
    def __init__(self):
        pass

    def _compute_episode_metrics(self, env, episode_info):
        return episode_info["articulation"]["robot"]["joint_velocity"][0]

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            vel_limit=None,
    ):
        results = dict()
        robot_joint_velocity = episode_metrics["articulation"]["robot"]["joint_velocity"][0]
        for vel in robot_joint_velocity:
            if vel > vel_limit:
                results["joint_velocity"] = {"success": False, "feedback": f"Robot's joint velocity is too high ({vel}), must be <= {vel_limit}"}
                break
        if "joint_velocity" not in results:
            results["joint_velocity"] = {"success": True, "feedback": None}

        return results
