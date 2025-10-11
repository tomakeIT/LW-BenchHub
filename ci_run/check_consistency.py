#!/usr/bin/env python3
import subprocess
import time
import sys
import signal
import atexit
import os
import json
import argparse
import yaml
from pathlib import Path
import shutil

os.environ["LW_API_ENDPOINT"] = "https://api-dev.lightwheel.net"
LWLAB_ROOT = Path(__file__).parent.parent.absolute()
DATASET_PATH = f"{LWLAB_ROOT}/datasets/"


def cleanup_process(process):
    """cleanup process function"""
    if process and process.poll() is None:
        print("terminating teleop process and its children...")
        try:
            try:
                pgid = os.getpgid(process.pid)
                print(f"terminating process group {pgid}...")
                os.killpg(pgid, signal.SIGTERM)

                process.wait(timeout=15)
                print("process group terminated normally")
            except (OSError, subprocess.TimeoutExpired):
                print("process group termination failed, trying to terminate main process...")
                process.terminate()
                process.wait(timeout=15)
                print("main process terminated")
        except subprocess.TimeoutExpired:
            print("force killing process group...")
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=2)
                print("process group force killed")
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                try:
                    process.wait(timeout=2)
                    print("main process force killed")
                except subprocess.TimeoutExpired:
                    print("warning: unable to terminate main process")


def update_config_task(task_name, layout):
    """Update the task field in check_consistency.yml"""
    try:
        # Read the original check_consistency.yml
        config_path = f"{LWLAB_ROOT}/configs/data_collection/teleop/check_consistency.yml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Store original task for restoration
        original_task = config.get('task', 'OpenDishwasher')

        # Update the task field
        config['task'] = task_name
        if layout:
            config['layout'] = layout
        # Write back to the file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Updated check_consistency.yml with task: {task_name}")
        return original_task

    except Exception as e:
        print(f"Error updating config: {e}")
        return None


def send_keyboard_input(process, input_text):
    """Send keyboard input to the process"""
    try:
        if process and process.poll() is None:
            process.stdin.write(input_text + '\n')
            process.stdin.flush()
            # pyautogui.press(input_text)
            print(f"Sent keyboard input: {input_text}")
            return True
        else:
            print("Process is not running, cannot send input")
            return False
    except Exception as e:
        print(f"Error sending keyboard input: {e}")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Monitor teleop_main.py execution')
    parser.add_argument('--task', type=str, default='OpenDishwasher',
                        help='Task name to use in the config (default: OpenDishwasher)')
    parser.add_argument('--layout', type=str, default=None,
                        help='Floorplan to use in the config')
    return parser.parse_args()


def run_teleop(rerun_count, reset_count=0):
    # Parse command line arguments
    args = parse_arguments()

    print(f"starting teleop_main.py monitoring with task: {args.task}...")

    process = None
    original_task = None
    test_result = {
        "success": False,
        "desc": "init desc",
        "error": "none",
    }

    try:
        # Update the config file with the specified task
        original_task = update_config_task(args.task, args.layout)
        if original_task is None:
            print("Failed to update config file")
            return False

        print(f"\n[RUN] Start teleop: rerun={rerun_count}, reset={reset_count}")

        process = subprocess.Popen(
            ["python3", "-u", f"{LWLAB_ROOT}/lwlab/scripts/teleop/teleop_check_consistency.py", "--task_config=check_consistency", "--headless"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            preexec_fn=os.setsid
        )

        # Register cleanup functions for all exit scenarios
        atexit.register(cleanup_process, process)

        # Register signal handlers
        def signal_handler(signum, frame):  # pylint: disable=unused-argument
            print(f"received signal {signum}, cleaning up...")
            cleanup_process(process)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        signal.signal(signal.SIGHUP, signal_handler)   # Hangup signal

        teleop_begins_detected = False
        traceback_lines = []
        success_message_detected = False

        print("wait for 'Start Recording' output...")

        monitor_start_time = time.time()
        timeout = 300

        # --------------------------First run or rerun-----------------
        in_traceback = False
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                print(f"[TELEOP] {line}")

                if "Traceback (most recent call last):" in line:
                    in_traceback = True
                    traceback_lines = [line]
                elif in_traceback:
                    if line == "" or ("ms]" in line):
                        in_traceback = False
                        if traceback_lines:
                            test_result["error"] = traceback_lines
                            last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                            test_result["desc"] = "\n".join(last_three_lines)
                    else:
                        traceback_lines.append(line)

                if "Start Recording" in line and not teleop_begins_detected:
                    teleop_begins_detected = True
                    print("Detect keyword 'Start Recording', start to monitor...")
                    break

                if "Starting teleoperation" in line:
                    print("Detect keyword 'Starting teleoperation', teleop_main is ready for input...")
                    time.sleep(8)  # wait a bit to ensure teleop is fully started
                    send_keyboard_input(process, "b")

        if process.poll() is not None:
            print("Process exited during main loop, reading remaining output...")
            remaining_output = process.stdout.read()
            if remaining_output:
                print(f"[REMAINING OUTPUT] {remaining_output}")
                for line in remaining_output.split('\n'):
                    if line.strip():
                        line = line.strip()
                        print(f"[TELEOP] {line}")
                        if "Traceback (most recent call last):" in line:
                            in_traceback = True
                            traceback_lines = [line]
                        elif in_traceback:
                            if line == "" or ("ms]" in line):
                                in_traceback = False
                                if traceback_lines:
                                    test_result["error"] = traceback_lines
                                    last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                                    test_result["desc"] = "\n".join(last_three_lines)
                            else:
                                traceback_lines.append(line)

        if in_traceback and traceback_lines:
            print("Found incomplete traceback after main loop, saving it...")
            test_result["error"] = traceback_lines
            last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
            test_result["desc"] = "\n".join(last_three_lines)

        if not teleop_begins_detected:
            if process.poll() is None:
                if time.time() - monitor_start_time > timeout:
                    # If traceback_lines is non-empty, success should be False
                    if traceback_lines:
                        test_result["success"] = False
                        test_result["desc"] = "Start over 5 minutes but with errors"
                    else:
                        test_result["success"] = True
                        test_result["desc"] = "Start over 5 minutes"
                    test_result["error"] = traceback_lines
                    cleanup_process(process)
                    return test_result["success"]
            else:
                test_result["success"] = False
                if not traceback_lines:
                    test_result["desc"] = "not detect Start keyword, teleop did not start"
                else:
                    last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                    test_result["desc"] = "\n".join(last_three_lines)
                test_result["error"] = traceback_lines
            return False

        wait_duration = 30
        print(f"[STEP] Wait {wait_duration}s before first save (with monitoring)...")

        wait_end_time = time.time() + wait_duration

        process_alive = True
        while time.time() < wait_end_time:
            if process.poll() is not None:
                print(f"[ERROR] Process terminated unexpectedly during the first wait. Exit code: {process.returncode}")
                test_result["success"] = False
                test_result["desc"] = f"Process terminated unexpectedly during the first wait.{wait_duration}"
                process_alive = False
                break
            time.sleep(0.1)

        if process_alive:
            print("[STEP] Sending 't' to save dataset 1...")
            send_keyboard_input(process, "t")
            time.sleep(5)
            rename_dataset(rerun_count, 0)
        else:
            return False

        # --------------------------Reset-----------------
        if rerun_count == 0:
            print("[STEP] Sending 'r' to reset...")
            send_keyboard_input(process, "r")
            print("[STEP] Wait 15s for reset to complete...")
            time.sleep(15)
            print("[STEP] Reset done.")

            send_keyboard_input(process, "b")

            print(f"[STEP] Wait {wait_duration}s before second save (with monitoring)...")
            process_alive = True
            wait_end_time = time.time() + wait_duration
            while time.time() < wait_end_time:
                if process.poll() is not None:
                    print(f"[ERROR] Process terminated unexpectedly during the second wait. Exit code: {process.returncode}")
                    test_result["success"] = False
                    test_result["desc"] = f"Process terminated unexpectedly during the second wait.{wait_duration}"
                    process_alive = False
                    break
                time.sleep(0.1)

            if process_alive:
                print("[STEP] Sending 't' to save dataset 2...")
                send_keyboard_input(process, "t")
                time.sleep(5)
                rename_dataset(rerun_count, 1)
            else:
                return False

        print("[STEP] Terminating teleop_main ...")

        print(f"[DONE] teleop_main run {rerun_count} finished.\n")

        if "success" not in test_result:
            test_result["success"] = True
            test_result["desc"] = "Teleop run and save completed successfully."

    except Exception as e:
        test_result["error"] = f"script error: {str(e)}"
        test_result["desc"] = "script error"
        if process:
            cleanup_process(process)
        return False
    finally:
        if process and process.poll() is None:
            print("[FINALLY] Cleaning up the process...")
            cleanup_process(process)


def main():
    rerun_total = 2

    for rerun_count in range(rerun_total):
        run_teleop(rerun_count)

    print("[DONE] All runs completed successfully.")


def rename_dataset(rerun_count, reset_count):
    new_name = f"{DATASET_PATH}/dataset_{rerun_count}_{reset_count}.hdf5"
    old_name = f"{DATASET_PATH}/dataset.hdf5"
    if os.path.exists(old_name):
        shutil.move(old_name, new_name)
        print(f"[INFO] Renamed dataset.hdf5 -> {new_name}")
    else:
        print(f"[WARN] dataset.hdf5 not found, skip rename")


def generate_result_json(test_result):
    try:
        os.makedirs(f"/output", exist_ok=True)
        with open(f"/output/result.json", "w", encoding="utf-8") as fp:
            json.dump(test_result, fp, ensure_ascii=False, indent=2)
        print(f"Generated result.json at /output/result.json")

    except Exception as e:
        print(f"fail to create result.json: {e}")


import h5py
import numpy as np


def check_dataset_consistency():
    """
    Check data consistency among dataset_0_0.hdf5, dataset_0_1.hdf5, dataset_1_0.hdf5
    for specified paths under /data/demo_0/.
    Compare first 200 rows of each dataset.
    """
    datasets = [
        f"{DATASET_PATH}/dataset_0_0.hdf5",
        f"{DATASET_PATH}/dataset_0_1.hdf5",
        f"{DATASET_PATH}/dataset_1_0.hdf5"
    ]
    check_paths = [
        "/data/demo_0/actions",
        "/data/demo_0/joint_targets/joint_pos_target",
        "/data/demo_0/states/articulation/robot/joint_position"
    ]

    test_result = {
        "success": True,
        "desc": "",
        "error": ""
    }
    precision = 1e-8

    # Check if dataset files exist
    for fpath in datasets:
        if not os.path.exists(fpath):
            test_result["success"] = False
            test_result["error"] += f"missing file: {fpath}\n"
            test_result["desc"] = "One or more dataset files not found."
            generate_result_json(test_result)
            return test_result

    print("[CHECK] Start consistency checking...")
    all_ok = True
    inconsistencies = []

    try:
        # Load all three files
        files = [h5py.File(fpath, "r") for fpath in datasets]

        for h5_path in check_paths:
            print(f"[CHECK] Comparing data at {h5_path} ...")
            try:
                # Read data from each file
                data_list = []
                row_counts = []

                for f in files:
                    if h5_path not in f:
                        raise KeyError(f"path not found: {h5_path}")
                    dset = f[h5_path][()]
                    if dset.ndim == 0:  # Empty dataset
                        raise ValueError(f"empty data at {h5_path}")
                    data_list.append(dset)
                    row_counts.append(dset.shape[0])

                # Use the shortest row length among all datasets to avoid index overflow
                min_rows = min(row_counts)
                if min_rows == 0:
                    raise ValueError(f"no valid rows found at {h5_path}")

                # Align all datasets by truncating to the shortest length
                data_list = [d[:min_rows] for d in data_list]

                # Print debug info
                print(f"[INFO] Using first {min_rows} rows for {h5_path}")

                # Compare with numpy
                base = data_list[0]
                for i, d in enumerate(data_list[1:], start=1):
                    if not np.allclose(base, d, atol=precision, equal_nan=True):
                        all_ok = False

                        # Find difference locations
                        diff_mask = ~np.isclose(base, d, atol=precision, equal_nan=True)
                        diff_indices = np.argwhere(diff_mask)

                        # Extract a few sample differences
                        diff_samples = []
                        for idx in diff_indices[:10]:  # Show first 10 differences
                            idx_tuple = tuple(idx)
                            base_val = base[idx_tuple]
                            diff_val = d[idx_tuple]
                            diff_samples.append({
                                "index": idx_tuple,
                                "base_value": float(base_val),
                                "compare_value": float(diff_val)
                            })

                        diff_coords = [list(map(int, idx)) for idx in diff_indices[:10]]

                        inconsistencies.append({
                            "path": h5_path,
                            "compare": f"dataset_0_0.hdf5 vs dataset_{i//2}_{i%2}.hdf5",
                            "diff_coords": diff_coords,
                            "note": f"{len(diff_indices)} differences found (showing first 10)"
                        })

                        print(f"[DIFF] Found difference at {h5_path}, total {len(diff_indices)} differences")
                        for ex in diff_samples:
                            print(f"index {ex['index']}: {ex['base_value']} != {ex['compare_value']}")

            except Exception as e:
                all_ok = False
                inconsistencies.append({
                    "path": h5_path,
                    "error": str(e)
                })
                print(f"[ERROR] {e}")

        # Close all opened files
        for f in files:
            f.close()

        # Summarize results
        if all_ok:
            test_result["success"] = True
            test_result["desc"] = "All compared datasets are consistent."
            test_result["error"] = "none"
            print("[CHECK] All datasets are consistent.")
        else:
            test_result["success"] = False
            test_result["desc"] = "Inconsistencies found in datasets."
            test_result["error"] = inconsistencies
            print("[CHECK] Inconsistencies detected!")

    except Exception as e:
        test_result["success"] = False
        test_result["desc"] = "Error during consistency checking."
        test_result["error"] = str(e)
        print(f"[CHECK] Fatal error: {e}")

    generate_result_json(test_result)
    return test_result


if __name__ == "__main__":
    main()
    print("\n[POST] Starting dataset consistency check...")
    check_dataset_consistency()
