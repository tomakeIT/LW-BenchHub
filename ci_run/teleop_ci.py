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

os.environ["LW_API_ENDPOINT"] = "http://api-dev.lightwheel.net:30807"


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


def update_config_task(task_name):
    """Update the task field in teleop_ci.yml"""
    try:
        # Read the original teleop_ci.yml
        config_path = "/workspace/lwlab/configs/data_collection/teleop/teleop_ci.yml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Store original task for restoration
        original_task = config.get('task', 'SizeSorting')

        # Update the task field
        config['task'] = task_name

        # Write back to the file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Updated teleop_ci.yml with task: {task_name}")
        return original_task

    except Exception as e:
        print(f"Error updating config: {e}")
        return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Monitor teleop_main.py execution')
    parser.add_argument('--task', type=str, default='SizeSorting',
                        help='Task name to use in the config (default: SizeSorting)')
    return parser.parse_args()


def main():
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
        original_task = update_config_task(args.task)
        if original_task is None:
            print("Failed to update config file")
            return False

        process = subprocess.Popen(
            ["python3", "-u", "/workspace/lwlab/lwlab/scripts/teleop/teleop_main.py", "--task_config=teleop_ci", "--headless"],
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
                    print("Detect keyword 'Start Recording', start to monitor 30 seconds...")
                    break

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
                    test_result["success"] = True
                    test_result["desc"] = "Start over 5 minutes"
                    test_result["error"] = traceback_lines
                    cleanup_process(process)
                    return True
            else:
                test_result["success"] = False
                if not traceback_lines:
                    test_result["desc"] = "not detect Start keyword, teleop did not start"
                else:
                    last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                    test_result["desc"] = "\n".join(last_three_lines)
                test_result["error"] = traceback_lines
            return False

        if teleop_begins_detected:
            print("wait for 30 seconds...")
            wait_start = time.time()
            while time.time() - wait_start < 30:
                if process.poll() is not None:
                    print(f"program exit in 30 seconds, exit code: {process.returncode}")

                    if in_traceback and traceback_lines:
                        print("Found incomplete traceback, saving it...")
                        test_result["error"] = traceback_lines
                        last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                        test_result["desc"] = "\n".join(last_three_lines)

                    print("Reading remaining output after process exit...")
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        print(f"[REMAINING OUTPUT] {remaining_output}")
                        for line in remaining_output.split('\n'):
                            if line.strip():
                                line = line.strip()
                                print(f"[TELEOP] {line}")

                                if "successful demonstrations" in line.lower():
                                    success_message_detected = True
                                    print(f"Detected success message in remaining output: {line}")

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
                            print("Found incomplete traceback in remaining output, saving it...")
                            test_result["error"] = traceback_lines
                            last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                            test_result["desc"] = "\n".join(last_three_lines)

                    if success_message_detected:
                        test_result["success"] = False
                        test_result["desc"] = "Fail: Task autoreset unexpectedly"
                    elif not test_result.get("error") or test_result["error"] == "none" or not traceback_lines:
                        test_result["desc"] = f"exit in 30 seconds: {process.returncode}"
                    return False
                time.sleep(0.1)

            if process.poll() is None:
                test_result["success"] = True
                test_result["desc"] = "Teleop survive over 30 seconds"
                cleanup_process(process)
                return True
    except Exception as e:
        test_result["error"] = f"script error: {str(e)}"
        test_result["desc"] = "Error! Script running error!"
        if process:
            cleanup_process(process)
        return False
    finally:
        if process:
            cleanup_process(process)

        # Final cleanup attempt
        generate_result_json(test_result)


def generate_result_json(test_result):
    try:
        os.makedirs("/output", exist_ok=True)
        with open("/output/result.json", "w", encoding="utf-8") as fp:
            json.dump(test_result, fp, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"fail to create result.json: {e}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
