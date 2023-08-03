import os


def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if the folder is empty
                print(f"Removing empty folder: {dir_path}")
                os.rmdir(dir_path)