"""
Download video folders from Google Drive dataset.

Usage:
    python download_data.py --count 3
    python download_data.py --count 10 --data_dir data
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import gdown
from tqdm import tqdm


API_KEY = "AIzaSyDIlxBgGBQTM5ImfSmF15NhDdJ7dtPgtqQ"
GDRIVE_PARENT_URL = "https://drive.google.com/drive/folders/1aIqGRHTsO9kNBrOXRRz9XV8kD0Ru8zSV"
GDRIVE_PARENT_ID = "1aIqGRHTsO9kNBrOXRRz9XV8kD0Ru8zSV"
DRIVE_API_FILES_URL = "https://www.googleapis.com/drive/v3/files"
MIN_FILES_PER_FOLDER = 5
MAX_RETRIES = 4
RETRY_DELAY_SEC = 15
DELAY_BETWEEN_FILES_SEC = 2


def _api_get(params: dict) -> dict:
    url = f"{DRIVE_API_FILES_URL}?{urlencode(params)}"
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_iter_files(query: str, fields: str, api_key: str):
    token = None
    while True:
        data = _api_get(
            {
                "q": query,
                "fields": f"nextPageToken,files({fields})",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
                "pageSize": 1000,
                "key": api_key,
                **({"pageToken": token} if token else {}),
            }
        )
        for f in data.get("files", []):
            yield f
        token = data.get("nextPageToken")
        if not token:
            return


def fetch_file_index_gdown() -> OrderedDict[str, list]:
    print("Fetching file index from Google Drive")
    all_files = gdown.download_folder(GDRIVE_PARENT_URL, skip_download=True, quiet=True, remaining_ok=True)
    folders: OrderedDict[str, list] = OrderedDict()
    for f in all_files:
        parts = f.path.split("/")
        if len(parts) < 2:
            continue
        folder_name, filename = parts[0], parts[1]
        if folder_name not in folders:
            folders[folder_name] = []
        folders[folder_name].append((f.id, filename))
    return folders


def fetch_file_index_api(api_key: str) -> OrderedDict[str, list]:
    print("Fetching file index from Google Drive API")
    folders: OrderedDict[str, list] = OrderedDict()
    q_folders = f"'{GDRIVE_PARENT_ID}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'"
    for folder in _api_iter_files(q_folders, "id,name", api_key):
        folder_id, folder_name = folder["id"], folder["name"]
        q_files = f"'{folder_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'"
        files = [(f["id"], f["name"]) for f in _api_iter_files(q_files, "id,name", api_key)]
        folders[folder_name] = files
    return folders


def fetch_file_index(method: str, api_key: str | None) -> OrderedDict[str, list]:
    if method == "api":
        if not api_key:
            raise ValueError("method=api requires --api_key or GOOGLE_API_KEY")
        return fetch_file_index_api(api_key)
    return fetch_file_index_gdown()


def _download_single_file_gdown(file_id: str, output_path: str) -> bool:
    gdown.download(id=file_id, output=output_path, quiet=True)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 100


def _download_single_file_api(file_id: str, output_path: str, api_key: str) -> bool:
    url = f"{DRIVE_API_FILES_URL}/{quote(file_id)}?alt=media&key={quote(api_key)}"
    with urlopen(url, timeout=120) as resp, open(output_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 100


def download_single_file(file_id: str, output_path: str, method: str, api_key: str | None) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if _download_single_file_api(file_id, output_path, api_key) if method == "api" else _download_single_file_gdown(file_id, output_path):
                return True
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception as e:
            if attempt == MAX_RETRIES:
                tqdm.write(f"Error: {e}")
        if attempt < MAX_RETRIES:
            tqdm.write("Retry")
            time.sleep(RETRY_DELAY_SEC * attempt)
    return False


def download_folder_files(folder_name: str, files: list[tuple[str, str]], data_dir: str, method: str, api_key: str | None) -> bool:
    output_dir = os.path.join(data_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    all_ok = True
    for file_id, filename in files:
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            continue
        if os.path.exists(output_path):
            os.remove(output_path)
        time.sleep(DELAY_BETWEEN_FILES_SEC)
        if not download_single_file(file_id, output_path, method, api_key):
            tqdm.write(f"Failed to download {filename}")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download video dataset from Google Drive")
    parser.add_argument("--count", type=int, required=True, help="Desired number of video folders in data/")
    parser.add_argument("--data_dir", type=str, default="data", help="Local data directory (default: data)")
    parser.add_argument("--method", type=str, choices=("gdown", "api"), default="gdown", help="Download backend: old gdown or new Drive API")
    parser.add_argument("--api_key", type=str, default=None, help="Google API key (required for --method api if GOOGLE_API_KEY is unset)")
    args = parser.parse_args()

    data_dir = args.data_dir
    target_count = args.count
    method = args.method
    api_key = args.api_key
    os.makedirs(data_dir, exist_ok=True)

    remote_folders = fetch_file_index(method, api_key)
    print(f"Found {len(remote_folders)} folders on Google Drive")

    if target_count > len(remote_folders):
        print(f"Warning: requested {target_count} but only {len(remote_folders)} available. Will download all.")
        target_count = len(remote_folders)

    existing = {name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))}
    print(f"Already downloaded: {len(existing)} complete folders")

    if len(existing) >= target_count:
        print(f"Already have {len(existing)} >= {target_count} folders. Nothing to do.")
        return

    need = target_count - len(existing)
    print(f"Need to download: {need} more folders")

    to_download = [(name, files) for name, files in remote_folders.items() if name not in existing]
    for folder_name, files in tqdm(to_download[:need], desc="Downloading", unit="folder"):
        tqdm.write(f"{folder_name} ({len(files)} files)")
        if not download_folder_files(folder_name, files, data_dir, method, api_key):
            tqdm.write(f"FAILED: {folder_name}")


if __name__ == "__main__":
    main()
