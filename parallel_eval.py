#!/usr/bin/env python3
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml


def resolve_path(path_str, base_dir):
    path = Path(os.path.expanduser(str(path_str)))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path)


def resolve_checkpoint(task, cfg, base_dir):
    checkpoint = (
        task.get("checkpoint")
        or task.get("model_path")
        or cfg.get("checkpoint")
        or cfg.get("model_path")
    )
    if checkpoint is None:
        return None
    return resolve_path(checkpoint, base_dir)


def wait_port(host, port, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def launch(cmd, env, log_file):
    f = open(log_file, "w", buffering=1)
    return subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
        text=True,
    )


def wrap(prefix, cmd):
    if not prefix.strip():
        return cmd
    full = prefix + " " + " ".join(shlex.quote(c) for c in cmd)
    return ["bash", "-lc", full]


def main(cfg_path):
    cfg_path = Path(cfg_path).expanduser().resolve()

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(os.path.expanduser(cfg["repo_root"]))
    gpus = cfg["gpus"]
    port_base = cfg.get("port_base", 50000)
    host = cfg.get("host", "127.0.0.1")
    authkey = cfg.get("authkey", "lightwheel")
    log_dir = Path(cfg.get("log_dir", "multi_eval_logs"))
    log_dir.mkdir(exist_ok=True)

    tasks = cfg["tasks"]

    if len(gpus) < len(tasks):
        raise ValueError(
            f"Not enough GPUs for tasks: {len(gpus)} GPUs configured for {len(tasks)} tasks."
        )

    procs = []

    def shutdown(*_):
        print("Shutting down...")
        for s, c in procs:
            for p in [c, s]:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    for i, task in enumerate(tasks):
        gpu = gpus[i]
        port = port_base + i

        server_env = os.environ.copy()
        client_env = os.environ.copy()

        server_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        client_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        server_env["LW_IPC_DEBUG"] = "1"
        client_env["LW_IPC_CLIENT_DEBUG"] = "1"

        if cfg.get("xla_preallocate_false", False):
            client_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        server_cmd = [
            "python3",
            str(repo_root / "lw_benchhub/scripts/env_server.py"),
            "--remote_protocol",
            "ipc",
            "--ipc_host",
            host,
            "--ipc_port",
            str(port),
            "--ipc_authkey",
            authkey,
        ]

        if cfg.get("enable_camera", False):
            server_cmd.append("--enable_camera")

        client_cmd = [
            "python3",
            str(repo_root / "lw_benchhub/scripts/policy/eval_policy.py"),
            "--config",
            str(repo_root / task["config"]),
            "--remote_protocol",
            "ipc",
            "--server_host",
            host,
            "--server_port",
            str(port),
            "--ipc_authkey",
            authkey,
        ]

        if cfg.get("debug_client_flow", False):
            client_cmd.append("--debug_client_flow")

        if cfg.get("debug_step_interval", 0) > 0:
            client_cmd += ["--debug_step_interval", str(cfg["debug_step_interval"])]

        checkpoint = resolve_checkpoint(task, cfg, repo_root)
        if checkpoint is not None:
            client_cmd += ["--overrides", "--checkpoint", checkpoint]

        server_cmd = wrap(cfg.get("server_prefix", ""), server_cmd)
        client_cmd = wrap(cfg.get("client_prefix", ""), client_cmd)

        print(f"Launching task {i} on GPU {gpu}, port {port}")

        server_log = log_dir / f"task{i}_server.log"
        client_log = log_dir / f"task{i}_client.log"

        server_p = launch(server_cmd, server_env, server_log)

        time.sleep(0.5)
        if not wait_port(host, port):
            print("Server failed to start.")
            shutdown()

        client_p = launch(client_cmd, client_env, client_log)

        procs.append((server_p, client_p))

    while True:
        time.sleep(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parallel_eval.py parallel_eval.yml")
        sys.exit(1)

    main(sys.argv[1])
