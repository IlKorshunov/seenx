import json
import os
import subprocess


def contains_subdirectory(directory):
    return any(os.path.isdir(os.path.join(directory, item)) for item in os.listdir(directory))


def upload_to_kaggle(dataset_folder, title, username, update=False, license_name="CC0-1.0"):
    metadata_path = os.path.join(dataset_folder, "dataset-metadata.json")
    metadata = {"title": title, "id": f"{username}/{title.replace(' ', '-').lower()}", "licenses": [{"name": license_name}]}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    command = "version" if update else "create"
    update_args = ["-m", '"update dataset"'] if update else []

    if contains_subdirectory(dataset_folder):
        print("Zipping the dataset folder...")
        try:
            subprocess.run(["kaggle", "datasets", command, "-p", dataset_folder, "--dir-mode", "zip", *update_args], check=True)
            print("Dataset uploaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
    else:
        print("Uploading the dataset folder...")
        try:
            subprocess.run(["kaggle", "datasets", command, "-p", dataset_folder, *update_args], check=True)
            print("Dataset uploaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


upload_to_kaggle(dataset_folder="checkpoints", title="seenx-checkpoints", username="zarus03", update=False)
