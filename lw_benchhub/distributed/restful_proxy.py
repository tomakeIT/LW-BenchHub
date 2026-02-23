import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Tuple
import time

import numpy as np
import requests
import torch


def _is_numeric_list(values):
    if not isinstance(values, list) or len(values) == 0:
        return False
    for item in values:
        if isinstance(item, list):
            if not _is_numeric_list(item):
                return False
        elif not isinstance(item, (int, float, bool)):
            return False
    return True


def _to_tensor_tree(data: Any):
    if isinstance(data, dict):
        return {k: _to_tensor_tree(v) for k, v in data.items()}
    if isinstance(data, list):
        if _is_numeric_list(data):
            return torch.as_tensor(np.asarray(data))
        return [_to_tensor_tree(v) for v in data]
    return data


def _normalize_image_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert image-like tensor to uint8 HWC/BHWC style values for policy preprocessing."""
    if not torch.is_tensor(x):
        return x
    if x.ndim not in (3, 4) or x.shape[-1] != 3:
        return x
    if x.dtype == torch.uint8:
        return x
    if torch.is_floating_point(x):
        max_val = float(torch.max(x).item()) if x.numel() > 0 else 0.0
        if max_val <= 1.0:
            x = x * 255.0
        return torch.clamp(x, 0.0, 255.0).to(torch.uint8)
    return torch.clamp(x, 0, 255).to(torch.uint8)


def _normalize_obs_images(data: Any):
    if isinstance(data, dict):
        return {k: _normalize_obs_images(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_normalize_obs_images(v) for v in data]
    if torch.is_tensor(data):
        return _normalize_image_tensor(data)
    return data


class RestfulRemoteEnv:
    """Client-side adapter for lw_benchhub RESTful environment server."""

    def __init__(self, address: Tuple[str, int], timeout: int = 120):
        self.host, self.port = address
        self.timeout = timeout
        self._session_id = None
        self._connected = False
        self._debug = os.getenv("LW_REST_CLIENT_DEBUG", "0") == "1"
        self._step_counter = 0
        self.action_space = SimpleNamespace(shape=())
        self.unwrapped = SimpleNamespace(cfg=SimpleNamespace(decimation=None))

    @classmethod
    def make(cls, address, authkey=None):
        # authkey is intentionally ignored to keep signature compatible with IPC RemoteEnv.make.
        return cls(address=address)

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def _post(self, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        t0 = time.time()
        response = requests.post(
            self._url(path),
            json=payload or {},
            timeout=self.timeout,
        )
        dt = time.time() - t0
        text = response.text
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = {"error": text}
        if response.status_code >= 400:
            raise RuntimeError(f"RESTful call failed: {path} [{response.status_code}] {body}")
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(f"RESTful call failed: {path} {body['error']}")
        if self._debug:
            print(f"[REST-CLIENT-DEBUG] {path} status={response.status_code} time={dt:.3f}s")
        return body

    def start_connection(self):
        self._connected = True

    def close_connection(self):
        self._connected = False

    def attach(self, env_config: Dict[str, Any]):
        if self._debug:
            print(f"[REST-CLIENT-DEBUG] attach.req env_config_keys={list(env_config.keys())}")
        body = self._post("/attach", {"env_config": env_config})
        self._session_id = body["session_id"]
        action_shape = body.get("action_space_shape")
        decimation = body.get("decimation")
        if action_shape is not None:
            self.action_space = SimpleNamespace(shape=tuple(action_shape))
        self.unwrapped = SimpleNamespace(cfg=SimpleNamespace(decimation=decimation))
        if self._debug:
            print(
                f"[REST-CLIENT-DEBUG] attach.res session_id={self._session_id} "
                f"action_space_shape={self.action_space.shape} decimation={decimation}"
            )
        self.start_connection()

    def detach(self):
        if self._session_id is None:
            return
        if self._debug:
            print(f"[REST-CLIENT-DEBUG] detach.req session_id={self._session_id}")
        self._post("/detach", {"session_id": self._session_id})
        if self._debug:
            print("[REST-CLIENT-DEBUG] detach.res ok")
        self._session_id = None

    def reset(self):
        if self._session_id is None:
            raise RuntimeError("Environment is not attached.")
        if self._debug:
            print(f"[REST-CLIENT-DEBUG] reset.req session_id={self._session_id}")
        body = self._post("/reset", {"session_id": self._session_id})
        obs_tensor = body.get("obs_tensor")
        if obs_tensor is None:
            raise RuntimeError("RESTful server did not return 'obs_tensor'.")
        obs = _normalize_obs_images(_to_tensor_tree(obs_tensor))
        info = body.get("info", {})
        if self._debug:
            print(f"[REST-CLIENT-DEBUG] reset.res obs_summary={self._summarize_obs(obs)} info_keys={list(info.keys())}")
        return obs, info

    def _encode_action(self, action: Any):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            action = action.tolist()
        return action

    def _normalize_bool_like(self, value):
        if isinstance(value, list):
            arr = np.asarray(value)
            if arr.shape == ():
                return bool(arr.item())
            return arr.astype(bool)
        return value

    def step(self, action: Any):
        if self._session_id is None:
            raise RuntimeError("Environment is not attached.")
        self._step_counter += 1
        if self._debug and (self._step_counter <= 5 or self._step_counter % 20 == 0):
            shape = tuple(action.shape) if torch.is_tensor(action) else (
                np.asarray(action).shape if isinstance(action, (list, tuple, np.ndarray)) else type(action).__name__
            )
            print(f"[REST-CLIENT-DEBUG] step.req idx={self._step_counter} action_shape={shape}")
        body = self._post(
            "/step",
            {"session_id": self._session_id, "action": self._encode_action(action)},
        )
        obs_tensor = body.get("obs_tensor")
        if obs_tensor is None:
            raise RuntimeError("RESTful server did not return 'obs_tensor'.")
        obs = _normalize_obs_images(_to_tensor_tree(obs_tensor))
        reward = body.get("reward")
        done = self._normalize_bool_like(body.get("done"))
        truncated = self._normalize_bool_like(body.get("truncated"))
        info = _to_tensor_tree(body.get("info", {}))
        if self._debug and (self._step_counter <= 5 or self._step_counter % 20 == 0):
            print(f"[REST-CLIENT-DEBUG] step.res idx={self._step_counter} done={done} truncated={truncated}")
        return obs, reward, done, truncated, info

    def close(self):
        if self._debug:
            print("[REST-CLIENT-DEBUG] close begin")
        try:
            self.detach()
        finally:
            self.close_connection()
        if self._debug:
            print("[REST-CLIENT-DEBUG] close end")

    def _summarize_obs(self, obs: Any):
        if isinstance(obs, dict):
            parts = []
            for k, v in obs.items():
                if isinstance(v, dict):
                    parts.append(f"{k}:dict({len(v)})")
                elif torch.is_tensor(v):
                    parts.append(f"{k}:tensor{tuple(v.shape)}/{v.dtype}")
                elif isinstance(v, np.ndarray):
                    parts.append(f"{k}:ndarray{v.shape}/{v.dtype}")
                else:
                    parts.append(f"{k}:{type(v).__name__}")
            return ", ".join(parts)
        return type(obs).__name__

