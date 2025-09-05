class BaseChecker:
    type = "base"

    def __init__(self, warning_on_screen=False):
        self.warning_on_screen = warning_on_screen

    def check(self, env):
        result = self._check(env)
        if self.warning_on_screen:
            self.show_warning(result)
        return result

    def _check(self, env):
        return {"success": True, "warning_text": None}

    def reset(self):
        pass

    def show_warning(self, result):
        if result.get("warning_text"):
            return result.get("warning_text")
        else:
            return None

    def get_metrics(self, result):
        if result.get("metrics"):
            return result.get("metrics")
        else:
            return {}
