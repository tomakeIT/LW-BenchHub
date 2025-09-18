from types import MethodType, FunctionType
import threading
from .proxy import EnvManager


def is_property(obj, attr_name):
    cls = type(obj)
    attr = getattr(cls, attr_name, None)
    return isinstance(attr, property)


def generate_env_attrs_meta_info(env):
    meta_info = {}
    attr_names = dir(env)
    for attr_name in attr_names:
        if not attr_name.startswith("_"):
            attr = getattr(env, attr_name)
            if callable(attr):
                meta_info[attr_name] = {
                    "callable": True,
                    "doc": attr.__doc__,
                    "type": f"{type(attr).__module__}.{type(attr).__name__}",
                    "is_method": isinstance(attr, MethodType),
                    "is_function": isinstance(attr, FunctionType),
                    "is_class": isinstance(attr, type),
                }
            else:
                attr_is_property = is_property(env, attr_name)
                attr_type_name = f"{type(attr).__module__}.{type(attr).__name__}"
                if attr is None and attr_is_property:
                    property_func = getattr(env.__class__, attr_name)
                    fget = property_func.fget
                    if "return" in fget.__annotations__:
                        attr_type_name = str(fget.__annotations__["return"])
                attr_is_property = is_property(env, attr_name)

                meta_info[attr_name] = {
                    "callable": False,
                    "type": attr_type_name,
                    "is_property": attr_is_property,
                }
    return meta_info


class DistributedEnvWrapper:
    def __init__(self, env):
        self._env = env
        self._manager = self._create_manager()
        self._server = self._manager.get_server()

    def serve(self):
        print(f"Waiting for connection on {self._server.listener.address}...")
        while True:
            self._server.stop_event = threading.Event()
            c = self._server.listener.accept()
            self._server.handle_request(c)

    def _create_manager(self, address=('', 50000), authkey=b'lightwheel'):
        # server
        mgr = EnvManager(address=address, authkey=authkey)
        mgr.register_for_server(self._env)
        return mgr

    def __getattr__(self, key):
        print(f"__getattr__: {key}")
        return getattr(self._env, key)

    # def __setattr__(self, key, value):
    #     if key in ("_env", "_manager", "_server"):
    #         return super().__setattr__(key, value)
    #     return setattr(self._env, key, value)
