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

import json
import time
import getpass
import requests
from pathlib import Path
from .exception import ApiException
from termcolor import colored
import os
CACHE_PATH = Path("~/.cache/lwlab/login/").expanduser()
CACHE_PATH.mkdir(parents=True, exist_ok=True)


class Login:
    """
    Login to the API.

    Args:
        host (str): The host of the API
        max_workers (int, optional): The maximum number of workers for downloading USD files. Defaults to 4.
    """

    def __init__(self, host):
        self.host = host
        self.cache_path = CACHE_PATH / f"account.json"

    def login(self, force_login=False):
        if not self.cache_path.exists() or force_login:
            account_data = {}
            account_data["username"] = input(colored("\nusername: ", "green"))
            account_data["password"] = getpass.getpass(colored("\npassword: ", "green"))
            response = requests.post(f"{self.host}/api/authenticate/v1/user/login", json={"username": account_data["username"], "password": account_data["password"]}, timeout=60)
            if response.status_code != 200:
                if response.status_code == 500 and response.json().get("message", "").startswith("login failed"):
                    print(colored("Invalid username or password", "red"))
                    return self.login(force_login=True)
                raise ApiException(response)
            token = response.json()["token"]
            headers = {
                "Authorization": f"Bearer {token}",
                "UserName": account_data["username"]
            }
            account_data["headers"] = headers
            with open(self.cache_path, "w") as f:
                json.dump(account_data, f)
            return headers
        else:
            with open(self.cache_path, "r") as f:
                account_data = json.load(f)
            return account_data["headers"]

    def get_headers(self):
        if "LoaderUserName" in os.environ and "LoaderToken" in os.environ:
            return {"Authorization": f"Bearer {os.environ['LoaderToken']}", 
                    "UserName": os.environ['LoaderUserName']}
        if self.cache_path.exists() and self.cache_path.is_file():
            with open(self.cache_path, "r") as f:
                account_data = json.load(f)
            return account_data["headers"]
        return {}
