from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_PARENT_FOLDER_ID = "1aIqGRHTsO9kNBrOXRRz9XV8kD0Ru8zSV"
DRIVE_API = "https://www.googleapis.com/drive/v3/files"


@dataclass
class SourceItem:
    file_id: str
    parent_id: str


def load_env_file(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


class AccessTokenManager:
    def get(self) -> str:
        token = os.getenv("GDRIVE_ACCESS_TOKEN") or os.getenv("GOOGLE_DRIVE_ACCESS_TOKEN") or os.getenv("DRIVE_ACCESS_TOKEN")
        if not token:
            raise RuntimeError("Google Drive access token not found. Set GDRIVE_ACCESS_TOKEN or use --snapshot-dir mode.")
        return token


def _api_get(access_token: str, params: dict[str, str]) -> dict:
    qs = urlencode(params)
    req = Request(f"{DRIVE_API}?{qs}", headers={"Authorization": f"Bearer {access_token}"})
    with urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _api_file(access_token: str, file_id: str, fields: str = "id,name,parents,mimeType") -> dict:
    req = Request(f"{DRIVE_API}/{quote(file_id)}?{urlencode({'fields': fields, 'supportsAllDrives': 'true'})}", headers={"Authorization": f"Bearer {access_token}"})
    with urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _list_files(access_token: str, query: str, fields: str) -> list[dict]:
    out: list[dict] = []
    token: str | None = None
    while True:
        data = _api_get(
            access_token,
            {
                "q": query,
                "fields": f"nextPageToken,files({fields})",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
                "pageSize": "1000",
                **({"pageToken": token} if token else {}),
            },
        )
        out.extend(data.get("files", []))
        token = data.get("nextPageToken")
        if not token:
            return out


def drive_find_file_in_parent(access_token: str, parent_id: str, filename: str) -> dict | None:
    q = f"'{parent_id}' in parents and trashed=false and name='{filename}'"
    files = _list_files(access_token, q, "id,name,parents,mimeType")
    return files[0] if files else None


def download_drive_text_file(access_token: str, file_id: str) -> str:
    req = Request(f"{DRIVE_API}/{quote(file_id)}?{urlencode({'alt': 'media', 'supportsAllDrives': 'true'})}", headers={"Authorization": f"Bearer {access_token}"})
    with urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8")


def resolve_video_folder_id(access_token: str, start_parent_id: str, root_folder_id: str) -> str | None:
    cur = start_parent_id
    prev = start_parent_id
    for _ in range(12):
        if not cur:
            return None
        if cur == root_folder_id:
            return prev
        meta = _api_file(access_token, cur, fields="id,parents")
        parents = meta.get("parents") or []
        prev = cur
        cur = parents[0] if parents else ""
    return None


def load_retention_target_for_video(access_token: str, video_folder_id: str) -> dict:
    retention_file = drive_find_file_in_parent(access_token, video_folder_id, "retention_parsed.json")
    if not retention_file or not retention_file.get("id"):
        return {"status": "error", "reason": "retention_parsed.json not found"}
    try:
        txt = download_drive_text_file(access_token, str(retention_file["id"]))
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
        return {"status": "error", "reason": "invalid retention json"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def discover_transcript_files(access_token: str, root_folder_id: str, max_items: int | None = None) -> list[tuple[str, str, SourceItem, None]]:
    q_folders = f"'{root_folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'"
    video_folders = _list_files(access_token, q_folders, "id,name")
    items: list[tuple[str, str, SourceItem, None]] = []
    for vf in video_folders:
        video_id = str(vf.get("id", ""))
        video_name = str(vf.get("name", ""))
        if not video_id:
            continue
        q_transcripts = f"'{video_id}' in parents and trashed=false and name='transcripts'"
        files = _list_files(access_token, q_transcripts, "id,name,parents")
        if not files:
            continue
        f = files[0]
        parent = (f.get("parents") or [""])[0]
        items.append((video_name, f"{video_name}/transcripts", SourceItem(file_id=str(f["id"]), parent_id=str(parent)), None))
        if max_items and len(items) >= int(max_items):
            break
    return items
