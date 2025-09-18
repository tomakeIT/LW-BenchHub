from lwlab.distributed.proxy import RemoteEnv
env = RemoteEnv.make(address=('127.0.0.1', 50000), authkey=b'lightwheel')

from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.runner.torch import Runner
import pickle
with open("agent_cfg.pkl", "rb") as f:
    agent_cfg = pickle.load(f)

# env = wrap_env(env, wrapper="isaaclab")

runner = Runner(env, agent_cfg)
runner.run()
