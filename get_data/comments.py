import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

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


def _extract_timecodes(text):
    return [{"raw": m.group(0), "seconds": int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))} for m in _TIMECODE_RE.finditer(text or "")]


def load_youtube():
    if not TOKEN_PATH.is_file():
        raise FileNotFoundError(f"Нет {TOKEN_PATH}. python get_data/auth.py")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("Токен недействителен")
    if not set(SCOPES).issubset(set(creds.scopes or [])):
        raise RuntimeError(f"В token.json scopes {set(creds.scopes or [])}, нужны {set(SCOPES)}")
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def fetch_channel_info(youtube):
    items = youtube.channels().list(part="snippet,statistics,brandingSettings,contentDetails,status", mine=True).execute().get("items") or []
    if not items:
        raise RuntimeError("Канал не найден")
    item = items[0]
    snippet, stats, details = item.get("snippet", {}), item.get("statistics", {}), item.get("contentDetails", {})
    return {
        "channel_id": item.get("id"),
        "title": snippet.get("title") or "",
        "description": snippet.get("description") or "",
        "subscriber_count": _int_or_none(stats.get("subscriberCount")),
        "video_count": _int_or_none(stats.get("videoCount")),
        "view_count": _int_or_none(stats.get("viewCount")),
        "uploads_playlist_id": (details.get("relatedPlaylists") or {}).get("uploads"),
    }


def get_uploads_playlist_id(youtube):
    items = youtube.channels().list(part="contentDetails", mine=True).execute().get("items") or []
    if not items:
        raise RuntimeError("Канал не найден")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def iter_playlists(youtube):
    page_token = None
    while True:
        resp = youtube.playlists().list(part="snippet,contentDetails", mine=True, maxResults=50, pageToken=page_token).execute()
        yield from resp.get("items", [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def _playlist_info(item, fallback_name=None):
    snippet = item.get("snippet") or {}
    title = (snippet.get("title") or "").strip() or (fallback_name or item["id"])
    return {
        "playlist_id": item["id"],
        "playlist_name": title,
        "playlist_description": snippet.get("description") or "",
        "playlist_dir": title,
        "item_count": _int_or_none((item.get("contentDetails") or {}).get("itemCount")),
    }


def fetch_playlist_by_id(youtube, playlist_id, fallback_name=None):
    items = youtube.playlists().list(part="snippet,contentDetails", id=playlist_id).execute().get("items") or []
    if not items:
        raise RuntimeError(f"Плейлист не найден: {playlist_id}")
    return _playlist_info(items[0], fallback_name=fallback_name)


def iter_all_playlists(youtube):
    uploads = fetch_playlist_by_id(youtube, get_uploads_playlist_id(youtube), fallback_name="uploads")
    seen = {uploads["playlist_id"]}
    yield uploads
    for item in iter_playlists(youtube):
        playlist = _playlist_info(item)
        if playlist["playlist_id"] not in seen:
            seen.add(playlist["playlist_id"])
            yield playlist


def resolve_playlist(youtube, playlist_id, playlist_name):
    if playlist_id:
        return fetch_playlist_by_id(youtube, playlist_id)
    if playlist_name:
        for item in iter_playlists(youtube):
            if ((item.get("snippet") or {}).get("title") or "").lower() == playlist_name.strip().lower():
                return _playlist_info(item, fallback_name=playlist_name.strip())
        raise RuntimeError(f"Плейлист не найден по имени: {playlist_name}")
    return fetch_playlist_by_id(youtube, get_uploads_playlist_id(youtube), fallback_name="uploads")


def fetch_playlist_video_ids(youtube, playlist_id, max_videos=None):
    kwargs = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
    video_ids = []
    while True:
        resp = youtube.playlistItems().list(**kwargs).execute()
        for item in resp.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
            if max_videos is not None and len(video_ids) >= max_videos:
                return video_ids
        kwargs["pageToken"] = resp.get("nextPageToken")
        if not kwargs["pageToken"]:
            return video_ids


def export_playlist_membership_map(youtube, out_path=None, skip_uploads=True):
    uploads_id = get_uploads_playlist_id(youtube)
    playlists_out, by_video = [], {}
    for item in iter_playlists(youtube):
        info = _playlist_info(item)
        pid, pname = info["playlist_id"], info["playlist_name"]
        if skip_uploads and pid == uploads_id:
            continue
        vids = fetch_playlist_video_ids(youtube, pid)
        playlists_out.append({"playlist_id": pid, "playlist_name": pname, "video_count": len(vids), "video_ids": vids})
        for vid in vids:
            by_video.setdefault(vid, []).append({"playlist_id": pid, "playlist_name": pname})
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "note": "Primary: playlists[].video_ids (playlist -> videos). by_video is the inverse. Uploads playlist excluded when skip_uploads=True.",
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


def fetch_video_titles(youtube, video_ids):
    titles = {}
    for i in range(0, len(video_ids), 50):
        for item in youtube.videos().list(part="snippet", id=",".join(video_ids[i : i + 50])).execute().get("items", []):
            titles[item["id"]] = item["snippet"].get("title") or ""
    return titles


def fetch_video_meta(youtube, video_ids):
    meta = {}
    for i in range(0, len(video_ids), 50):
        for item in youtube.videos().list(part="snippet,statistics,contentDetails,status", id=",".join(video_ids[i : i + 50])).execute().get("items", []):
            snippet, stats = item.get("snippet", {}), item.get("statistics", {})
            meta[item["id"]] = {
                "video_id": item["id"],
                "video_title": snippet.get("title") or "",
                "video_description": snippet.get("description") or "",
                "video_published_at": snippet.get("publishedAt"),
                "video_tags": snippet.get("tags") or [],
                "video_category_id": snippet.get("categoryId"),
                "video_view_count": _int_or_none(stats.get("viewCount")),
                "video_like_count": _int_or_none(stats.get("likeCount")),
                "video_comment_count": _int_or_none(stats.get("commentCount")),
            }
    return meta


def _comment_payload(comment, parent_id=None):
    snippet = comment["snippet"]
    text = (snippet.get("textDisplay") or snippet.get("textOriginal") or "").strip()
    return {
        "comment_id": comment["id"],
        "parent_id": parent_id,
        "text": text,
        "timecodes": _extract_timecodes(text),
        "author": snippet.get("authorDisplayName") or "",
        "like_count": int(snippet.get("likeCount") or 0),
        "updated_at": snippet.get("updatedAt") or snippet.get("publishedAt"),
    }


def fetch_replies(youtube, parent_id):
    kwargs = {"part": "snippet", "parentId": parent_id, "maxResults": 100, "textFormat": "plainText"}
    replies = []
    while True:
        resp = youtube.comments().list(**kwargs).execute()
        for item in resp.get("items", []):
            replies.append(_comment_payload(item, parent_id))
        kwargs["pageToken"] = resp.get("nextPageToken")
        if not kwargs["pageToken"]:
            return sorted(replies, key=lambda x: x["updated_at"] or "")


def _is_comments_disabled_error(err):
    if err.resp.status != 403:
        return False
    try:
        body = json.loads(err.content.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError):
        return False
    return any(e.get("reason") == "commentsDisabled" for e in (body.get("error") or {}).get("errors") or [])


def fetch_top_level_comments(youtube, video_id):
    page_token = None
    comments = []
    while True:
        resp = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, textFormat="plainText", pageToken=page_token).execute()
        for item in resp.get("items", []):
            comments.append(_comment_payload(item["snippet"]["topLevelComment"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            return sorted(comments, key=lambda x: x["updated_at"] or "")


def fetch_video_threads(youtube, video_id):
    threads = []
    try:
        for top in fetch_top_level_comments(youtube, video_id):
            thread = dict(top)
            thread["replies"] = fetch_replies(youtube, top["comment_id"])
            threads.append(thread)
    except HttpError as e:
        if _is_comments_disabled_error(e):
            return [], True
        raise
    return threads, False


def resolve_root_dir(out):
    path = Path(out).expanduser() if out is not None else COMMENTS_ROOT
    return path if path.is_absolute() else _ROOT / path


def export_playlist(youtube, root_dir, playlist, max_videos=None, video_id=None, video_title=None, force=False):
    base_dir = root_dir / playlist["playlist_dir"]
    base_dir.mkdir(parents=True, exist_ok=True)
    video_ids = [video_id.strip()] if video_id else fetch_playlist_video_ids(youtube, playlist["playlist_id"], max_videos)
    pending_video_ids = [vid for vid in video_ids if force or not (base_dir / vid / "comments.json").is_file()]
    for vid in set(video_ids) - set(pending_video_ids):
        print(f"[{playlist['playlist_name']}] Пропуск видео: {vid}", file=sys.stderr)

    titles = fetch_video_titles(youtube, pending_video_ids) if pending_video_ids else {}
    video_meta = fetch_video_meta(youtube, pending_video_ids) if pending_video_ids else {}
    if video_id and video_title is not None and video_id.strip() in pending_video_ids:
        titles[video_id.strip()] = video_title
        video_meta.setdefault(video_id.strip(), {})["video_title"] = video_title

    playlist_path = base_dir / "playlist.json"
    if force or pending_video_ids or not playlist_path.is_file():
        with playlist_path.open("w", encoding="utf-8") as f:
            json.dump({**playlist, "video_count_exported": len(video_ids), "video_ids": video_ids}, f, ensure_ascii=False, indent=2)

    print(f"[{playlist['playlist_name']}] Видео: {len(video_ids)}, новых: {len(pending_video_ids)}", file=sys.stderr)
    total, result_path = 0, base_dir
    for vid in pending_video_ids:
        out_path = base_dir / vid / "comments.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        threads, comments_disabled = fetch_video_threads(youtube, vid)
        if comments_disabled:
            print(f"[{playlist['playlist_name']}] Комментарии отключены у видео: {vid}", file=sys.stderr)
        n_vid = sum(1 + len(thread["replies"]) for thread in threads)
        total += n_vid
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **video_meta.get(vid, {"video_id": vid, "video_title": titles.get(vid, "")}),
                    "playlist_id": playlist["playlist_id"],
                    "playlist_name": playlist["playlist_name"],
                    "thread_count": len(threads),
                    "comment_count": n_vid,
                    "comments_disabled": comments_disabled,
                    "threads": threads,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        result_path = out_path
    return result_path, total


def main():
    ap = argparse.ArgumentParser(description="Export YouTube comments")
    ap.add_argument("--out", type=Path, default=None, help="Корневая папка для всех данных")
    ap.add_argument("--video-id", default=None, metavar="ID")
    ap.add_argument("--playlist-id", default=None)
    ap.add_argument("--playlist-name", default=None)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--video-title", default=None)
    ap.add_argument("--force", action="store_true", help="Пересчитать уже сохраненные плейлисты и видео")
    ap.add_argument("--playlist-map-only", action="store_true", help="Только записать карту видео↔плейлисты (как в Studio), без комментариев")
    ap.add_argument("--include-uploads-playlist", action="store_true", help="Включить системный плейлист всех загрузок (UU), иначе только тематические")
    ap.add_argument("--playlist-map-out", type=Path, default=None, help="Путь для JSON карты (по умолчанию get_data/comments/video_playlist_membership.json)")
    args = ap.parse_args()
    youtube = load_youtube()

    CHANNEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHANNEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(fetch_channel_info(youtube), f, ensure_ascii=False, indent=2)

    if args.playlist_map_only:
        root = resolve_root_dir(args.out) if args.out else None
        map_path = args.playlist_map_out
        if map_path is not None and not map_path.is_absolute():
            map_path = _ROOT / map_path
        map_path = map_path or ((root or COMMENTS_ROOT) / "video_playlist_membership.json")
        payload = export_playlist_membership_map(youtube, map_path, skip_uploads=not args.include_uploads_playlist)
        print(f"Карта плейлистов: {len(payload['playlists'])} плейлистов, {len(payload['by_video'])} уникальных video_id -> {map_path}", file=sys.stderr)
        return

    root_dir = resolve_root_dir(args.out)
    root_dir.mkdir(parents=True, exist_ok=True)
    playlists = [resolve_playlist(youtube, args.playlist_id, args.playlist_name)] if args.playlist_id or args.playlist_name or args.video_id else list(iter_all_playlists(youtube))
    total_comments, last_result_path = 0, root_dir
    for playlist in playlists:
        last_result_path, n = export_playlist(youtube, root_dir, playlist, max_videos=args.max_videos, video_id=args.video_id, video_title=args.video_title, force=args.force)
        total_comments += n
    print(f"Готово: {last_result_path} ({total_comments} записей)", file=sys.stderr)


if __name__ == "__main__":
    main()
