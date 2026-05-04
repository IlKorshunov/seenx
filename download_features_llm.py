#!/usr/bin/env python3
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen


API_KEY = "AIzaSyDIlxBgGBQTM5ImfSmF15NhDdJ7dtPgtqQ"
ROOT_FOLDER_ID = "1aIqGRHTsO9kNBrOXRRz9XV8kD0Ru8zSV"
LOCAL_DATA_DIR = Path("/home/kolya/ilya/seenx-ml/data")
DRIVE_API_FILES_URL = "https://www.googleapis.com/drive/v3/files"

MAX_RETRIES = 4
RETRY_DELAY_SEC = 15
DELAY_BETWEEN_FILES_SEC = 2
OVERWRITE_EXISTING = False


def api_get_json(params: dict) -> dict:
    url = f"{DRIVE_API_FILES_URL}?{urlencode(params)}"
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_iter_files(query: str, fields: str):
    page_token = None
    while True:
        params = {"q": query, "fields": f"nextPageToken,files({fields})", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true", "pageSize": 1000, "key": API_KEY}
        if page_token:
            params["pageToken"] = page_token

        data = api_get_json(params)

        for f in data.get("files", []):
            yield f

        page_token = data.get("nextPageToken")
        if not page_token:
            break


def list_child_folders(parent_id: str):
    q = f"'{parent_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'"
    return list(api_iter_files(q, "id,name,mimeType"))


def find_child_folder_by_name(parent_id: str, folder_name: str):
    q = f"'{parent_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
    items = list(api_iter_files(q, "id,name,mimeType"))
    return items[0] if items else None


def find_child_file_by_name(parent_id: str, file_name: str):
    q = f"'{parent_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder' and name='{file_name}'"
    items = list(api_iter_files(q, "id,name,mimeType,size"))
    return items[0] if items else None


def download_file(file_id: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    url = f"{DRIVE_API_FILES_URL}/{quote(file_id)}?alt=media&supportsAllDrives=true&key={quote(API_KEY)}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(url, timeout=120) as resp, open(tmp_path, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            if tmp_path.exists() and tmp_path.stat().st_size > 0:
                tmp_path.replace(dest_path)
                return True

            if tmp_path.exists():
                tmp_path.unlink()

        except (HTTPError, URLError, TimeoutError, OSError) as e:
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SEC * attempt)

    return False


def main():
    print(f"Listing video folders directly inside ROOT_FOLDER_ID={ROOT_FOLDER_ID}")
    video_folders = list_child_folders(ROOT_FOLDER_ID)
    print(f"Found {len(video_folders)} candidate video folders")

    downloaded = 0
    skipped = 0
    errors = 0

    for idx, video_folder in enumerate(video_folders, start=1):
        video_id = video_folder["name"]
        video_folder_id = video_folder["id"]

        print(f"\n[{idx}/{len(video_folders)}] {video_id}")

        try:
            transcripts_folder = find_child_folder_by_name(video_folder_id, "transcripts")
            if transcripts_folder is None:
                print("  transcripts folder not found, skip")
                skipped += 1
                continue

            features_file = find_child_file_by_name(transcripts_folder["id"], "features_llm.json")
            if features_file is None:
                print("  features_llm.json not found, skip")
                skipped += 1
                continue

            local_dest = LOCAL_DATA_DIR / video_id / "features_llm.json"

            local_dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  downloading -> {local_dest}")
            ok = download_file(features_file["id"], local_dest)

            if ok:
                print("  done")
                downloaded += 1
            else:
                print("  failed to download")
                errors += 1

            time.sleep(DELAY_BETWEEN_FILES_SEC)

        except Exception as e:
            print(f"  error: {e}")
            errors += 1

    print("\nDone")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Errors:     {errors}")


if __name__ == "__main__":
    main()
