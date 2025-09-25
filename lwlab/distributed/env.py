from types import MethodType, FunctionType
from torch.multiprocessing import Queue  # noqa: F401
import threading
import signal
import socket
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
        self._shutdown_event = threading.Event()
        self._setup_signal_handlers()

    def serve(self):
        print(f"Waiting for connection on {self._server.listener.address}...")
        print("Press Ctrl+C to stop the server")

        while not self._shutdown_event.is_set():
            try:
                # Set socket to non-blocking mode temporarily to check for shutdown
                self._server.listener._listener._socket.settimeout(1.0)  # 1 second timeout
                c = self._server.listener.accept()
                self._server.listener._listener._socket.settimeout(None)  # Reset to blocking

                self._server.stop_event = threading.Event()
                self._server.handle_request(c)
            except socket.timeout:
                # Timeout occurred, check if we should shutdown
                continue
            except OSError as e:
                if self._shutdown_event.is_set():
                    print("Server shutting down...")
                    break
                else:
                    raise e

        print("Server stopped.")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down gracefully...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _create_manager(self, address=('', 50000), authkey=b'lightwheel'):
        # server
        mgr = EnvManager(address=address, authkey=authkey)
        mgr.register_for_server(self)
        return mgr

    def __getattr__(self, key):
        print(f"__getattr__: {key}")
        return getattr(self._env, key)

    def close_connection(self):
        self._server.stop_event.set()

    def close(self):
        print("Closing environment")
        self._env.close()

    # def __setattr__(self, key, value):
    #     if key in ("_env", "_manager", "_server"):
    #         return super().__setattr__(key, value)
    #     return setattr(self._env, key, value)
