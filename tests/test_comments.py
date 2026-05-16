import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class FakeListResource:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return FakeRequest(self.responses.pop(0))


class FakeYoutube:
    def __init__(self, **resources):
        self.resources = resources

    def channels(self):
        return self.resources["channels"]

    def playlists(self):
        return self.resources["playlists"]

    def playlistItems(self):
        return self.resources["playlist_items"]

    def videos(self):
        return self.resources["videos"]

    def commentThreads(self):
        return self.resources["comment_threads"]

    def comments(self):
        return self.resources["comments"]


def load_comments(monkeypatch):
    for name in ["google", "google.auth", "google.auth.transport", "google.oauth2", "googleapiclient"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    requests = types.ModuleType("google.auth.transport.requests")
    credentials = types.ModuleType("google.oauth2.credentials")
    discovery = types.ModuleType("googleapiclient.discovery")
    errors = types.ModuleType("googleapiclient.errors")

    class HttpError(Exception):
        def __init__(self, status=403, content=b"{}"):
            self.resp = types.SimpleNamespace(status=status)
            self.content = content

    requests.Request = object  # type: ignore[attr-defined]
    credentials.Credentials = types.SimpleNamespace(from_authorized_user_file=lambda *args, **kwargs: None)  # type: ignore[attr-defined]
    discovery.build = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    errors.HttpError = HttpError  # type: ignore[attr-defined]
    for name, module in {
        "google.auth.transport.requests": requests,
        "google.oauth2.credentials": credentials,
        "googleapiclient.discovery": discovery,
        "googleapiclient.errors": errors,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("comments_under_test", ROOT / "get_data/comments.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def top_comment(comment_id, text, updated_at="2024-01-01T00:00:00Z"):
    return {"id": comment_id, "snippet": {"textOriginal": text, "likeCount": 2, "updatedAt": updated_at, "authorDisplayName": "a"}}


def test_extract_timecodes_and_int_conversion(monkeypatch):
    comments = load_comments(monkeypatch)

    assert comments._int_or_none("42") == 42
    assert comments._int_or_none("") is None
    assert comments._int_or_none("bad") is None
    assert comments._extract_timecodes("смотри 1:02 и 01:02:03") == [{"raw": "1:02", "seconds": 62}, {"raw": "01:02:03", "seconds": 3723}]


def test_fetch_playlist_video_ids_reads_pages_and_respects_limit(monkeypatch):
    comments = load_comments(monkeypatch)
    youtube = FakeYoutube(
        playlist_items=FakeListResource(
            [
                {"items": [{"contentDetails": {"videoId": "a"}}, {"contentDetails": {"videoId": "b"}}], "nextPageToken": "next"},
                {"items": [{"contentDetails": {"videoId": "c"}}]},
            ]
        )
    )

    assert comments.fetch_playlist_video_ids(youtube, "pl") == ["a", "b", "c"]
    assert youtube.playlistItems().calls[1]["pageToken"] == "next"

    youtube = FakeYoutube(playlist_items=FakeListResource([{"items": [{"contentDetails": {"videoId": "a"}}, {"contentDetails": {"videoId": "b"}}], "nextPageToken": "ignored"}]))
    assert comments.fetch_playlist_video_ids(youtube, "pl", max_videos=1) == ["a"]


def test_fetch_video_threads_sorts_comments_and_replies(monkeypatch):
    comments = load_comments(monkeypatch)
    youtube = FakeYoutube(
        comment_threads=FakeListResource([{"items": [{"snippet": {"topLevelComment": top_comment("late", "late", "2024-02-01")}}, {"snippet": {"topLevelComment": top_comment("early", "early 0:05", "2024-01-01")}}]}]),
        comments=FakeListResource([{"items": [{"id": "r1", "snippet": {"textOriginal": "reply", "likeCount": 0, "updatedAt": "2024-01-02"}}]}, {"items": []}]),
    )

    threads, disabled = comments.fetch_video_threads(youtube, "vid")

    assert disabled is False
    assert [thread["comment_id"] for thread in threads] == ["early", "late"]
    assert threads[0]["timecodes"] == [{"raw": "0:05", "seconds": 5}]
    assert threads[0]["replies"][0]["parent_id"] == "early"


def test_fetch_video_threads_handles_disabled_comments(monkeypatch):
    comments = load_comments(monkeypatch)
    error_body = json.dumps({"error": {"errors": [{"reason": "commentsDisabled"}]}}).encode()

    class BrokenThreads:
        def list(self, **kwargs):
            raise comments.HttpError(status=403, content=error_body)

    threads, disabled = comments.fetch_video_threads(FakeYoutube(comment_threads=BrokenThreads()), "vid")

    assert threads == []
    assert disabled is True


def test_export_playlist_membership_map_writes_forward_and_inverse_indexes(tmp_path, monkeypatch):
    comments = load_comments(monkeypatch)
    youtube = FakeYoutube(
        channels=FakeListResource([{"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "uploads"}}}]}]),
        playlists=FakeListResource(
            [
                {"items": [{"id": "uploads", "snippet": {"title": "Uploads"}, "contentDetails": {"itemCount": 1}}, {"id": "pl", "snippet": {"title": "Main"}, "contentDetails": {"itemCount": 2}}]},
            ]
        ),
        playlist_items=FakeListResource([{"items": [{"contentDetails": {"videoId": "v1"}}, {"contentDetails": {"videoId": "v2"}}]}]),
    )

    payload = comments.export_playlist_membership_map(youtube, tmp_path / "map.json", skip_uploads=True)

    assert payload["playlists"] == [{"playlist_id": "pl", "playlist_name": "Main", "video_count": 2, "video_ids": ["v1", "v2"]}]
    assert payload["by_video"] == {"v1": [{"playlist_id": "pl", "playlist_name": "Main"}], "v2": [{"playlist_id": "pl", "playlist_name": "Main"}]}
    assert json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))["skip_uploads"] is True
