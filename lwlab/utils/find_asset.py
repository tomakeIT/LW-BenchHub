import os
from pathlib import Path


def find_folder(folder_name, assets_dir):
    folder_paths = []
    for root, dirs, files in os.walk(str(assets_dir)):
        for dir in dirs:
            if folder_name.lower() in dir.lower() \
                    or folder_name.lower().replace('_', ' ') in dir.lower() \
                    and any(char.isdigit() for char in dir):
                folder_paths.append(str((Path(root) / dir).relative_to(assets_dir) / 'model.xml'))
    return folder_paths


if __name__ == "__main__":
    """
    get the relative path of the model.yaml of the specific asset name
    """
    assets_dir = Path('/home/chaorui.zhang/Fanxiang/lwlab/third_party/robocasa/robocasa/models/assets')
    folder_paths = find_folder("Cup030", assets_dir)
    for obj in folder_paths:
        print(f'/{obj}')
