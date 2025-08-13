# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from datetime import datetime  # Import datetime module
import time  # Ensure time module is imported since it's used in main
import traceback


# (Original log_scene_rigid_objects function remains unchanged)
def log_scene_rigid_objects(env):
    """
    Retrieves USD path information for all rigid objects in the environment and logs it to a JSON file 
    in the log folder under the program root directory.
    ... (Function content omitted as it matches your original code) ...
    """
    # 1. Get all keys from env.scene.rigid_objects
    # Check if env.scene exists and if rigid_objects is a dictionary
    if not hasattr(env, 'scene') or not hasattr(env.scene, 'rigid_objects') or not isinstance(env.scene.rigid_objects, dict):
        print("Warning: env.scene.rigid_objects is unavailable or not dictionary type. Cannot log rigid object information.")
        # Return a default filename to ensure error logging functionality
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"default_scene_log_{current_time_str}.json"

    rigid_objects_dict = env.scene.rigid_objects
    key_list = list(rigid_objects_dict.keys())
    print(f"Detected {len(key_list)} rigid objects.")

    # 2. Record USD path information for each object
    objects_info = []
    for key in key_list:
        obj = rigid_objects_dict[key]
        usd_path = "N/A"
        try:
            # Attempt safe access to nested attributes
            if hasattr(obj, 'cfg') and hasattr(obj.cfg, 'spawn') and hasattr(obj.cfg.spawn, 'usd_path'):
                usd_path = obj.cfg.spawn.usd_path
            elif hasattr(obj, 'usd_path'):  # Alternative if usd_path is directly on object
                usd_path = obj.usd_path
        except AttributeError:
            usd_path = "Attribute Missing"  # If any part of attribute path is missing
        except Exception as e:
            usd_path = f"Error: {e}"  # Catch other possible errors

        objects_info.append({
            "name": key,
            "usd_path": str(usd_path)  # Ensure conversion to string
        })

    # 3. Define log folder path
    log_dir = os.path.join(os.getcwd(), "log")  # log folder under program root directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")

    # 4. Construct filename
    # Use getattr to safely get configuration info with defaults
    scene_name_full = str(getattr(env.cfg, 'scene_name', '')) if hasattr(env, 'cfg') else ''
    env_name_part = str(getattr(env.cfg, 'env_name', '')) if hasattr(env, 'cfg') else ''

    # --- Key modification: Remove "Robocasa-" prefix from env_name_part ---
    if env_name_part.startswith("Robocasa-"):
        env_name_part = env_name_part[len("Robocasa-"):]
    # --------------------------------------------------------

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    scene_name_base = ""
    # Store all '-' separated parts except base name
    additional_scene_parts = []

    # Split full scene name by '-'
    parts = scene_name_full.split('-')

    if parts:
        scene_name_base = parts[0]  # First part is base name
        additional_scene_parts = parts[1:]  # Remaining parts are additional scene identifiers

    # Build list of filename components
    filename_components_to_sanitize = []

    # # Add scene base name (commented out per instructions)
    # if scene_name_base:
    #     filename_components_to_sanitize.append(scene_name_base)

    # Add environment name part if exists
    if env_name_part:
        # Add separator only if preceding content exists
        if filename_components_to_sanitize:
            filename_components_to_sanitize.append("__")
        filename_components_to_sanitize.append(env_name_part)

    # Add additional scene parts sequentially with '__' separator
    for part in additional_scene_parts:
        if part:  # Ensure part is not empty
            # Add separator only if preceding content exists
            if filename_components_to_sanitize:
                filename_components_to_sanitize.append("__")
            filename_components_to_sanitize.append(part)

    # Add timestamp
    if current_time_str:
        # Add separator only if preceding content exists
        if filename_components_to_sanitize:
            filename_components_to_sanitize.append("__")
        filename_components_to_sanitize.append(current_time_str)

    # Sanitize each filename component
    sanitized_parts = []
    for part in filename_components_to_sanitize:
        # Replace invalid filename characters with underscore
        s_part = "".join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in part).strip()
        if s_part:  # Exclude empty string parts
            sanitized_parts.append(s_part)

    # Directly join sanitized parts
    # Provide default filename if all parts are empty
    if not sanitized_parts:
        final_filename = f"default_scene_log_{current_time_str}.json"
    else:
        final_filename = "".join(sanitized_parts) + ".json"

    log_file_path = os.path.join(log_dir, final_filename)

    # 5. Write to JSON file
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(objects_info, f, indent=4, ensure_ascii=False)  # ensure_ascii=False supports non-ASCII
        print(f"Object information successfully logged to: {log_file_path}")
    except Exception as e:
        print(f"Error writing object information file: {e}")

    return final_filename


# handle_exception_and_log function definition
def handle_exception_and_log(e, final_filename):
    """
    Handles exceptions caught in the program and logs them to the final_filename JSON file in the log folder.
    Error information is appended as a list item under the "_error_events" key.
    Ensures original data isn't overwritten by wrapping it under "data_records" key if original data is a list.

    Args:
        e (Exception): Caught exception object.
        final_filename (str): Main data filename (without path), e.g., "my_data.json".
                              Expected to be in "log" folder under current working directory.
        simulation_app: Isaac Sim application object for shutdown during exceptions.
    """
    # 1. Construct full log file path
    log_dir = os.path.join(os.getcwd(), "log")  # Consistent with log_scene_rigid_objects
    full_log_file_path = os.path.join(log_dir, final_filename)

    print(f"An error occurred: {e}. Attempting to log into {full_log_file_path}")

    # 2. Prepare error data for logging
    error_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "exception_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc().splitlines()  # Split stacktrace for JSON storage
    }

    # 3. Ensure log directory exists
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as create_dir_error:
        print(f"Error creating log directory {log_dir}: {create_dir_error}")

    # 4. Attempt to load existing data from full_log_file_path
    current_data = {}
    try:
        # Check if file exists and isn't empty to avoid JSONDecodeError
        if os.path.exists(full_log_file_path) and os.path.getsize(full_log_file_path) > 0:
            with open(full_log_file_path, "r") as f:
                try:
                    current_data = json.load(f)
                    # If top-level structure is a list (original format from log_scene_rigid_objects),
                    # wrap it in a dictionary to add top-level keys like "_error_events"
                    if isinstance(current_data, list):
                        current_data = {"data_records": current_data}
                        print(f"Info: Original data in {full_log_file_path} was a JSON list. Wrapped under 'data_records' key.")
                except json.JSONDecodeError:
                    print(f"Warning: Existing file {full_log_file_path} contains invalid JSON. Initializing with empty data.")
                    current_data = {}  # Restart if JSON is corrupted
        else:
            print(f"Info: {full_log_file_path} does not exist or is empty. Initializing as empty JSON object.")

    except Exception as read_error:
        print(f"Error reading {full_log_file_path}: {read_error}. Initializing with empty data.")
        current_data = {}

    # 5. Add new error entry to current_data
    # Ensure "_error_events" key exists and is a list
    if "_error_events" not in current_data:
        current_data["_error_events"] = []
    elif not isinstance(current_data["_error_events"], list):
        # Reset if "_error_events" exists but isn't a list
        print(f"Warning: '_error_events' in {full_log_file_path} is not a list. Reinitializing.")
        current_data["_error_events"] = []

    current_data["_error_events"].append(error_entry)

    # 6. Write updated data back to full_log_file_path
    try:
        with open(full_log_file_path, "w", encoding='utf-8') as f:  # Ensure utf-8 encoding
            json.dump(current_data, f, indent=4, ensure_ascii=False)  # indent=4 for readability
        print(f"Successfully appended error information to {full_log_file_path}.")
    except IOError as write_error:
        print(f"Error writing to {full_log_file_path}: {write_error}. Original error was: {e}")

    # 7. Close simulation_app
    print("Closing simulation application due to an error...")
