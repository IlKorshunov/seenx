from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from datetime import UTC
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


_ROOT = Path(__file__).resolve().parent.parent
TOKEN_PATH = _ROOT / "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
COMMENTS_ROOT = _ROOT / "get_data" / "comments"
CHANNEL_PATH = COMMENTS_ROOT / "channel.json"
_TIMECODE_RE = re.compile(r"(?<!\d)(?:(\d+):)?([0-5]?\d):([0-5]\d)(?!\d)")


def _int_or_none(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _extract_timecodes(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in _TIMECODE_RE.finditer(text or ""):
        hh = int(m.group(1) or 0)
        mm = int(m.group(2))
        ss = int(m.group(3))
        out.append({"raw": m.group(0), "seconds": hh * 3600 + mm * 60 + ss})
    return out


def load_youtube():
    if not TOKEN_PATH.is_file():
        raise FileNotFoundError(f"Нет {TOKEN_PATH}. python get_data/auth.py")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("Токен недействителен")
    need, have = set(SCOPES), set(creds.scopes or [])
    if not need.issubset(have):
        raise RuntimeError(f"В token.json scopes {have}, нужны {need}")
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def fetch_channel_info(youtube) -> dict[str, Any]:
    items = youtube.channels().list(part="snippet,statistics,brandingSettings,contentDetails,status", mine=True).execute().get("items") or []
    if not items:
        raise RuntimeError("Канал не найден")
    it = items[0]
    sn = it.get("snippet", {})
    st = it.get("statistics", {})
    cd = it.get("contentDetails", {})
    return {
        "channel_id": it.get("id"),
        "title": sn.get("title") or "",
        "description": sn.get("description") or "",
        "subscriber_count": _int_or_none(st.get("subscriberCount")),
        "video_count": _int_or_none(st.get("videoCount")),
        "view_count": _int_or_none(st.get("viewCount")),
        "uploads_playlist_id": (cd.get("relatedPlaylists") or {}).get("uploads"),
    }


def get_uploads_playlist_id(youtube) -> str:
    items = youtube.channels().list(part="contentDetails", mine=True).execute().get("items") or []
    if not items:
        raise RuntimeError("Канал не найден")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def iter_playlists(youtube) -> Iterator[dict[str, Any]]:
    page_token: str | None = None
    while True:
        resp = youtube.playlists().list(part="snippet,contentDetails", mine=True, maxResults=50, pageToken=page_token).execute()
        for it in resp.get("items", []):
            yield it
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def _playlist_info(it: dict[str, Any], fallback_name: str | None = None) -> dict[str, Any]:
    snippet = it.get("snippet") or {}
    title = (snippet.get("title") or "").strip() or (fallback_name or it["id"])
    return {
        "playlist_id": it["id"],
        "playlist_name": title,
        "playlist_description": snippet.get("description") or "",
        "playlist_dir": title,
        "item_count": _int_or_none((it.get("contentDetails") or {}).get("itemCount")),
    }


def fetch_playlist_by_id(youtube, playlist_id: str, fallback_name: str | None = None) -> dict[str, Any]:
    items = youtube.playlists().list(part="snippet,contentDetails", id=playlist_id).execute().get("items") or []
    if not items:
        raise RuntimeError(f"Плейлист не найден: {playlist_id}")
    return _playlist_info(items[0], fallback_name=fallback_name)


def iter_all_playlists(youtube) -> Iterator[dict[str, Any]]:
    uploads_id = get_uploads_playlist_id(youtube)
    seen: set[str] = set()
    uploads = fetch_playlist_by_id(youtube, uploads_id, fallback_name="uploads")
    seen.add(uploads["playlist_id"])
    yield uploads
    for it in iter_playlists(youtube):
        playlist = _playlist_info(it)
        if playlist["playlist_id"] in seen:
            continue
        seen.add(playlist["playlist_id"])
        yield playlist


def resolve_playlist(youtube, playlist_id: str | None, playlist_name: str | None) -> dict[str, Any]:
    if playlist_id:
        return fetch_playlist_by_id(youtube, playlist_id)
    if playlist_name:
        for it in iter_playlists(youtube):
            title = (it.get("snippet") or {}).get("title") or ""
            if title.lower() == playlist_name.strip().lower():
                return _playlist_info(it, fallback_name=playlist_name.strip())
        raise RuntimeError(f"Плейлист не найден по имени: {playlist_name}")
    return fetch_playlist_by_id(youtube, get_uploads_playlist_id(youtube), fallback_name="uploads")


def iter_playlist_video_ids(youtube, playlist_id: str, max_videos: int | None) -> Iterator[str]:
    kwargs: dict[str, Any] = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
    count = 0
    while True:
        resp = youtube.playlistItems().list(**kwargs).execute()
        for it in resp.get("items", []):
            vid = it["contentDetails"]["videoId"]
            yield vid
            count += 1
            if max_videos is not None and count >= max_videos:
                return
        tok = resp.get("nextPageToken")
        if not tok:
            break
        kwargs["pageToken"] = tok


def export_playlist_membership_map(youtube, out_path: Path | None = None, *, skip_uploads: bool = True) -> dict[str, Any]:
    """Export playlist -> video_ids for every themed playlist (as in Studio).

    Output JSON keys:
      - ``playlists``: list of ``{playlist_id, playlist_name, video_count, video_ids}``
        — основное представление: один плейлист → список video_id.
      - ``by_video``: inverse index ``video_id -> [{playlist_id, playlist_name}, ...]``
        — удобно, если нужно по видео узнать все плейлисты (ролик может быть в нескольких).

    The system uploads playlist (UU...) is *all* public uploads — not the same as
    themed playlists. With skip_uploads=True it is excluded so splits match Studio.
    """
    from datetime import datetime

    uploads_id = get_uploads_playlist_id(youtube)
    playlists_out: list[dict[str, Any]] = []
    by_video: dict[str, list[dict[str, str]]] = {}

    for it in iter_playlists(youtube):
        info = _playlist_info(it)
        pid, pname = info["playlist_id"], info["playlist_name"]
        if skip_uploads and pid == uploads_id:
            continue
        vids = list(iter_playlist_video_ids(youtube, pid, max_videos=None))
        playlists_out.append({"playlist_id": pid, "playlist_name": pname, "video_count": len(vids), "video_ids": vids})
        for vid in vids:
            by_video.setdefault(vid, []).append({"playlist_id": pid, "playlist_name": pname})

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "note": ("Primary: playlists[].video_ids (playlist -> videos). by_video is the inverse. Uploads playlist excluded when skip_uploads=True."),
        "skip_uploads": skip_uploads,
        "uploads_playlist_id": uploads_id,
        "playlists": playlists_out,
        "by_video": {k: v for k, v in sorted(by_video.items())},
    }
    out_path = out_path or (COMMENTS_ROOT / "video_playlist_membership.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def fetch_video_titles(youtube, video_ids: list[str]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for i in range(0, len(video_ids), 50):
        r = youtube.videos().list(part="snippet", id=",".join(video_ids[i : i + 50])).execute()
        for it in r.get("items", []):
            titles[it["id"]] = it["snippet"].get("title") or ""
    return titles


def fetch_video_meta(youtube, video_ids: list[str]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for i in range(0, len(video_ids), 50):
        r = youtube.videos().list(part="snippet,statistics,contentDetails,status", id=",".join(video_ids[i : i + 50])).execute()
        for it in r.get("items", []):
            sn = it.get("snippet", {})
            st = it.get("statistics", {})
            meta[it["id"]] = {
                "video_id": it["id"],
                "video_title": sn.get("title") or "",
                "video_description": sn.get("description") or "",
                "video_published_at": sn.get("publishedAt"),
                "video_tags": sn.get("tags") or [],
                "video_category_id": sn.get("categoryId"),
                "video_view_count": _int_or_none(st.get("viewCount")),
                "video_like_count": _int_or_none(st.get("likeCount")),
                "video_comment_count": _int_or_none(st.get("commentCount")),
            }
    return meta


def iter_replies(youtube, parent_id: str) -> Iterator[dict[str, Any]]:
    kwargs: dict[str, Any] = {"part": "snippet", "parentId": parent_id, "maxResults": 100, "textFormat": "plainText"}
    while True:
        resp = youtube.comments().list(**kwargs).execute()
        for it in resp.get("items", []):
            tid = it["id"]
            ts = it["snippet"]
            text = (ts.get("textDisplay") or ts.get("textOriginal") or "").strip()
            yield {
                "comment_id": tid,
                "parent_id": parent_id,
                "text": text,
                "timecodes": _extract_timecodes(text),
                "author": ts.get("authorDisplayName") or "",
                "like_count": int(ts.get("likeCount") or 0),
                "updated_at": ts.get("updatedAt") or ts.get("publishedAt"),
            }
        tok = resp.get("nextPageToken")
        if not tok:
            break
        kwargs["pageToken"] = tok


def _is_comments_disabled_error(err: HttpError) -> bool:
    if err.resp.status != 403:
        return False
    try:
        body = json.loads(err.content.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError):
        return False
    for e in (body.get("error") or {}).get("errors") or []:
        if e.get("reason") == "commentsDisabled":
            return True
    return False


def iter_top_level_comments(youtube, video_id: str) -> Iterator[dict[str, Any]]:
    page_token: str | None = None
    while True:
        resp = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, textFormat="plainText", pageToken=page_token).execute()
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]
            ts = top["snippet"]
            text = (ts.get("textDisplay") or ts.get("textOriginal") or "").strip()
            yield {
                "comment_id": top["id"],
                "parent_id": None,
                "text": text,
                "timecodes": _extract_timecodes(text),
                "author": ts.get("authorDisplayName") or "",
                "like_count": int(ts.get("likeCount") or 0),
                "updated_at": ts.get("updatedAt") or ts.get("publishedAt"),
            }
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def fetch_video_threads(youtube, video_id: str) -> tuple[list[dict[str, Any]], bool]:
    """Returns (threads, comments_disabled). If API reports disabled comments, threads is []."""
    threads: list[dict[str, Any]] = []
    try:
        for top in iter_top_level_comments(youtube, video_id):
            thread = dict(top)
            thread["replies"] = sorted(list(iter_replies(youtube, top["comment_id"])), key=lambda x: x["updated_at"] or "")
            threads.append(thread)
    except HttpError as e:
        if _is_comments_disabled_error(e):
            return [], True
        raise
    return sorted(threads, key=lambda x: x["updated_at"] or ""), False


def resolve_root_dir(out: Path | None) -> Path:
    if out is None:
        return COMMENTS_ROOT
    p = Path(out).expanduser()
    if not p.is_absolute():
        p = _ROOT / p
    return p


def export_playlist(youtube, root_dir: Path, playlist: dict[str, Any], *, max_videos: int | None, video_id: str | None, video_title: str | None, force: bool) -> tuple[Path, int]:
    base_dir = root_dir / playlist["playlist_dir"]
    base_dir.mkdir(parents=True, exist_ok=True)

    if video_id:
        video_ids = [video_id.strip()]
    else:
        video_ids = list(iter_playlist_video_ids(youtube, playlist["playlist_id"], max_videos))

    pending_video_ids = []
    for vid in video_ids:
        out_path = base_dir / vid / "comments.json"
        if force or not out_path.is_file():
            pending_video_ids.append(vid)
        else:
            print(f"[{playlist['playlist_name']}] Пропуск видео: {vid}", file=sys.stderr)

    titles = fetch_video_titles(youtube, pending_video_ids) if pending_video_ids else {}
    video_meta = fetch_video_meta(youtube, pending_video_ids) if pending_video_ids else {}
    if video_id and video_title is not None and video_id.strip() in pending_video_ids:
        vid = video_id.strip()
        titles[vid] = video_title
        video_meta.setdefault(vid, {})["video_title"] = video_title

    playlist_payload = {**playlist, "video_count_exported": len(video_ids), "video_ids": video_ids}
    playlist_path = base_dir / "playlist.json"
    if force or pending_video_ids or not playlist_path.is_file():
        with playlist_path.open("w", encoding="utf-8") as f:
            json.dump(playlist_payload, f, ensure_ascii=False, indent=2)

    print(f"[{playlist['playlist_name']}] Видео: {len(video_ids)}, новых: {len(pending_video_ids)}", file=sys.stderr)
    n = 0
    result_path = base_dir

    for vid in pending_video_ids:
        out_path = base_dir / vid / "comments.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        threads, comments_disabled = fetch_video_threads(youtube, vid)
        if comments_disabled:
            print(f"[{playlist['playlist_name']}] Комментарии отключены у видео: {vid}", file=sys.stderr)
        n_vid = sum(1 + len(thread["replies"]) for thread in threads)
        n += n_vid
        meta = video_meta.get(vid, {"video_id": vid, "video_title": titles.get(vid, "")})
        video = {
            **meta,
            "playlist_id": playlist["playlist_id"],
            "playlist_name": playlist["playlist_name"],
            "thread_count": len(threads),
            "comment_count": n_vid,
            "comments_disabled": comments_disabled,
            "threads": threads,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(video, f, ensure_ascii=False, indent=2)
        result_path = out_path
    return result_path, n


def main() -> None:
    ap = argparse.ArgumentParser(description="Export YouTube comments")
    ap.add_argument("--out", type=Path, default=None, help="Корневая папка для всех данных")
    ap.add_argument("--video-id", type=str, default=None, metavar="ID")
    ap.add_argument("--playlist-id", type=str, default=None)
    ap.add_argument("--playlist-name", type=str, default=None)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--video-title", type=str, default=None)
    ap.add_argument("--force", action="store_true", help="Пересчитать уже сохраненные плейлисты и видео")
    ap.add_argument("--playlist-map-only", action="store_true", help="Только записать карту видео↔плейлисты (как в Studio), без комментариев")
    ap.add_argument("--include-uploads-playlist", action="store_true", help="Включить системный плейлист всех загрузок (UU...), иначе только тематические")
    ap.add_argument("--playlist-map-out", type=Path, default=None, help="Путь для JSON карты (по умолчанию get_data/comments/video_playlist_membership.json)")
    args = ap.parse_args()
    youtube = load_youtube()
    channel = fetch_channel_info(youtube)
    CHANNEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHANNEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(channel, f, ensure_ascii=False, indent=2)

    if args.playlist_map_only:
        out = resolve_root_dir(args.out) if args.out else None
        map_path = args.playlist_map_out
        if map_path is not None and not map_path.is_absolute():
            map_path = _ROOT / map_path
        if out is not None and map_path is None:
            map_path = out / "video_playlist_membership.json"
        elif map_path is None:
            map_path = COMMENTS_ROOT / "video_playlist_membership.json"
        payload = export_playlist_membership_map(youtube, map_path, skip_uploads=not args.include_uploads_playlist)
        n_pl = len(payload["playlists"])
        n_vid = len(payload["by_video"])
        print(f"Карта плейлистов: {n_pl} плейлистов, {n_vid} уникальных video_id -> {map_path}", file=sys.stderr)
        return

    root_dir = resolve_root_dir(args.out)
    root_dir.mkdir(parents=True, exist_ok=True)

    if args.playlist_id or args.playlist_name or args.video_id:
        playlists = [resolve_playlist(youtube, args.playlist_id, args.playlist_name)]
    else:
        playlists = list(iter_all_playlists(youtube))

    total_comments = 0
    last_result_path = root_dir
    for playlist in playlists:
        last_result_path, n = export_playlist(youtube, root_dir, playlist, max_videos=args.max_videos, video_id=args.video_id, video_title=args.video_title, force=args.force)
        total_comments += n
    print(f"Готово: {last_result_path} ({total_comments} записей)", file=sys.stderr)


if __name__ == "__main__":
    main()
