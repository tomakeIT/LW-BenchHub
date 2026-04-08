import contextlib
import shutil
import time
from pathlib import Path

import numpy as np


def as_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def create_episode_dir(save_data_dir: Path, task_name: str) -> Path:
    task_dir = save_data_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(16):
        episode_name = f"{task_name}_{time.time_ns() // 1000}"
        episode_dir = task_dir / episode_name
        if not episode_dir.exists():
            episode_dir.mkdir(parents=True, exist_ok=False)
            return episode_dir
        time.sleep(0.000001)

    fallback_dir = task_dir / f"{task_name}_{time.time_ns() // 1000}_fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir


def switch_remote_recorder_output(env, dataset_path: Path) -> bool:
    try:
        recorder_manager = env.unwrapped.recorder_manager
        dataset_handler = recorder_manager._dataset_file_handler
        if dataset_handler is None:
            print("[SAVE-DATA] Recorder manager has no dataset handler; skip export.")
            return False
        with contextlib.suppress(Exception):
            dataset_handler.close()

        env_name = ""
        with contextlib.suppress(Exception):
            env_name = str(env.unwrapped.cfg.env_name)
        dataset_handler.create(str(dataset_path), env_name=env_name)
        return True
    except Exception as e:
        print(f"[SAVE-DATA] Failed to switch recorder output: {e}")
        return False


def remote_get_recorded_episode_count(env) -> int:
    try:
        recorder_manager = env.unwrapped.recorder_manager
        dataset_handler = recorder_manager._dataset_file_handler
        if dataset_handler is None:
            return 0
        with contextlib.suppress(Exception):
            dataset_handler.flush()
        return int(dataset_handler.get_num_episodes())
    except Exception:
        return 0


def remote_force_export_current_episode(env) -> bool:
    try:
        recorder_manager = env.unwrapped.recorder_manager
        recorder_manager.export_episodes()
        with contextlib.suppress(Exception):
            recorder_manager._dataset_file_handler.flush()
        with contextlib.suppress(Exception):
            failed_handler = recorder_manager._failed_episode_dataset_file_handler
            if failed_handler is not None:
                failed_handler.flush()
        return True
    except Exception as e:
        print(f"[SAVE-DATA] Failed to force export current episode: {e}")
        return False


TARGET_CAMERA_ORDER = (
    "left_hand",
    "first_person",
    "right_hand",
    "left_shoulder",
    "eye_in_hand",
    "right_shoulder",
)


def _resolve_camera_reorder_indices(record_camera) -> list[int] | None:
    if not isinstance(record_camera, (list, tuple)):
        return None
    camera_names = [str(name).lower() for name in record_camera]
    if len(camera_names) < 6:
        return None

    indices = []
    used = set()
    for target in TARGET_CAMERA_ORDER:
        idx = None
        for i, camera_name in enumerate(camera_names):
            if i in used:
                continue
            if camera_name.startswith(target) or target in camera_name:
                idx = i
                break
        if idx is None:
            return None
        indices.append(idx)
        used.add(idx)
    return indices


def _copy_or_convert_video(video_src: Path, video_dst: Path, record_camera) -> None:
    reorder_indices = _resolve_camera_reorder_indices(record_camera)
    if reorder_indices is None:
        print(
            "[SAVE-DATA] Warning: camera list does not match expected 2x3 layout, "
            f"copy raw video without re-layout. record_camera={record_camera}"
        )
        shutil.copy2(video_src, video_dst)
        return

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_src))
        if not cap.isOpened():
            raise RuntimeError("failed to open source video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("source video has no frames")

        frame_h, frame_w = frame.shape[:2]
        camera_count = len(record_camera)
        if camera_count <= 0 or frame_w % camera_count != 0:
            raise RuntimeError("source video width is not divisible by camera count")

        tile_w = frame_w // camera_count
        output_size = (tile_w * 3, frame_h * 2)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_dst), fourcc, fps, output_size)
        if not writer.isOpened():
            raise RuntimeError("failed to open output video writer")

        def _compose_grid(input_frame):
            tiles = [input_frame[:, i * tile_w:(i + 1) * tile_w] for i in range(camera_count)]
            ordered_tiles = [tiles[i] for i in reorder_indices]
            top_row = np.concatenate(ordered_tiles[:3], axis=1)
            bottom_row = np.concatenate(ordered_tiles[3:6], axis=1)
            return np.concatenate([top_row, bottom_row], axis=0)

        try:
            writer.write(_compose_grid(frame))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(_compose_grid(frame))
        finally:
            writer.release()
            cap.release()
    except Exception as e:
        print(f"[SAVE-DATA] Warning: failed to convert video to 2x3 grid ({e}); fallback to raw copy.")
        if video_dst.exists():
            video_dst.unlink()
        shutil.copy2(video_src, video_dst)


def export_episode_rollout_data(
    *,
    save_data_only_success: bool,
    save_data_video_name: str,
    save_data_hdf5_success_name: str,
    save_data_hdf5_failed_name: str,
    episode_dir: Path,
    run_id: str,
    succeeded: bool,
    recorded_demo_count: int,
    eval_root: Path,
    record_camera,
) -> None:
    should_keep = recorded_demo_count > 0 and (succeeded or not save_data_only_success)
    if not should_keep:
        if episode_dir.exists():
            shutil.rmtree(episode_dir, ignore_errors=True)
        if succeeded and recorded_demo_count <= 0:
            print(f"[SAVE-DATA] Warning: {run_id} marked success but recorder exported 0 demos.")
        return

    if (
        (not save_data_only_success)
        and (not succeeded)
        and save_data_hdf5_failed_name
        and save_data_hdf5_failed_name != save_data_hdf5_success_name
    ):
        success_hdf5 = episode_dir / save_data_hdf5_success_name
        failed_hdf5 = episode_dir / save_data_hdf5_failed_name
        if success_hdf5.exists():
            with contextlib.suppress(Exception):
                success_hdf5.rename(failed_hdf5)

    if succeeded:
        video_candidates = [eval_root / f"{run_id}_success.mp4", eval_root / f"{run_id}.mp4"]
    else:
        video_candidates = [eval_root / f"{run_id}.mp4", eval_root / f"{run_id}_success.mp4"]

    copied_video = False
    for video_src in video_candidates:
        if video_src.exists():
            _copy_or_convert_video(video_src, episode_dir / save_data_video_name, record_camera)
            copied_video = True
            break
    if not copied_video:
        print(f"[SAVE-DATA] Warning: video not found for {run_id} under {eval_root}.")
