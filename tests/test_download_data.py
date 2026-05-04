"""Tests for download_data.py"""

import os
import types
from collections import OrderedDict
from unittest.mock import patch

from download_data import MIN_FILES_PER_FOLDER, download_folder_files, download_single_file, fetch_file_index, is_folder_complete


class TestIsFolderComplete:
    def test_nonexistent_dir(self, tmp_path):
        assert is_folder_complete(str(tmp_path / "nope")) is False

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert is_folder_complete(str(d)) is False

    def test_too_few_files(self, tmp_path):
        d = tmp_path / "few"
        d.mkdir()
        (d / "video.mp4").write_bytes(b"x")
        (d / "meta.json").write_bytes(b"x")
        assert is_folder_complete(str(d)) is False

    def test_missing_video(self, tmp_path):
        d = tmp_path / "no_video"
        d.mkdir()
        for name in ["meta.json", "audio.mp3", "retention.csv", "retention.json", "retention.png"]:
            (d / name).write_bytes(b"x")
        assert is_folder_complete(str(d)) is False

    def test_missing_meta(self, tmp_path):
        d = tmp_path / "no_meta"
        d.mkdir()
        for name in ["video.mp4", "audio.mp3", "retention.csv", "retention.json", "retention.png"]:
            (d / name).write_bytes(b"x")
        assert is_folder_complete(str(d)) is False

    def test_complete_folder(self, tmp_path):
        d = tmp_path / "ok"
        d.mkdir()
        for name in ["video.mp4", "meta.json", "audio.mp3", "retention.csv", "retention.json", "retention.png"]:
            (d / name).write_bytes(b"x")
        assert is_folder_complete(str(d)) is True

    def test_exact_min_files(self, tmp_path):
        """Exactly MIN_FILES_PER_FOLDER files including required ones."""
        d = tmp_path / "exact"
        d.mkdir()
        names = ["video.mp4", "meta.json", "audio.mp3", "retention.csv", "retention.json"]
        assert len(names) == MIN_FILES_PER_FOLDER
        for name in names:
            (d / name).write_bytes(b"x")
        assert is_folder_complete(str(d)) is True


def _make_gdrive_file(file_id: str, path: str):
    """Create a fake GoogleDriveFileToDownload namedtuple."""
    return types.SimpleNamespace(id=file_id, path=path, local_path=f"/tmp/{path}")


class TestFetchFileIndex:
    @patch("download_data.gdown.download_folder")
    def test_basic_parsing(self, mock_dl):
        mock_dl.return_value = [_make_gdrive_file("id1", "folderA/video.mp4"), _make_gdrive_file("id2", "folderA/meta.json"), _make_gdrive_file("id3", "folderB/video.mp4")]

        result = fetch_file_index()

        assert isinstance(result, OrderedDict)
        assert list(result.keys()) == ["folderA", "folderB"]
        assert result["folderA"] == [("id1", "video.mp4"), ("id2", "meta.json")]
        assert result["folderB"] == [("id3", "video.mp4")]

    @patch("download_data.gdown.download_folder")
    def test_preserves_order(self, mock_dl):
        mock_dl.return_value = [_make_gdrive_file("1", "c_folder/f.mp4"), _make_gdrive_file("2", "a_folder/f.mp4"), _make_gdrive_file("3", "b_folder/f.mp4")]

        result = fetch_file_index()
        assert list(result.keys()) == ["c_folder", "a_folder", "b_folder"]

    @patch("download_data.gdown.download_folder")
    def test_skips_flat_paths(self, mock_dl):
        """Files without subfolder (path has no '/') should be skipped."""
        mock_dl.return_value = [_make_gdrive_file("id1", "README.md"), _make_gdrive_file("id2", "folderA/video.mp4")]

        result = fetch_file_index()
        assert list(result.keys()) == ["folderA"]

    @patch("download_data.gdown.download_folder")
    def test_empty_result(self, mock_dl):
        mock_dl.return_value = []
        result = fetch_file_index()
        assert len(result) == 0


class TestDownloadSingleFile:
    @patch("download_data.gdown.download")
    def test_success_first_attempt(self, mock_dl, tmp_path):
        out = str(tmp_path / "file.mp4")

        def fake_download(**kwargs):
            with open(kwargs["output"], "wb") as f:
                f.write(b"x" * 200)

        mock_dl.side_effect = fake_download
        assert download_single_file("some_id", out) is True
        assert mock_dl.call_count == 1

    @patch("download_data.gdown.download")
    def test_empty_file_triggers_retry(self, mock_dl, tmp_path):
        out = str(tmp_path / "file.mp4")
        call_count = 0

        def fake_download(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                with open(kwargs["output"], "wb") as f:
                    f.write(b"<html>err</html>")
            else:
                with open(kwargs["output"], "wb") as f:
                    f.write(b"x" * 200)

        mock_dl.side_effect = fake_download
        with patch("download_data.time.sleep"):
            assert download_single_file("some_id", out) is True
        assert call_count == 3

    @patch("download_data.gdown.download")
    def test_all_retries_fail(self, mock_dl, tmp_path):
        out = str(tmp_path / "file.mp4")

        def fake_download(**kwargs):
            pass

        mock_dl.side_effect = fake_download
        with patch("download_data.time.sleep"):
            assert download_single_file("some_id", out) is False

    @patch("download_data.gdown.download")
    def test_exception_triggers_retry(self, mock_dl, tmp_path):
        out = str(tmp_path / "file.mp4")
        mock_dl.side_effect = Exception("network error")

        with patch("download_data.time.sleep"):
            assert download_single_file("some_id", out) is False


class TestDownloadFolderFiles:
    def _make_files_list(self, names):
        return [(f"id_{n}", n) for n in names]

    @patch("download_data.download_single_file")
    def test_downloads_all_files(self, mock_dl, tmp_path):
        files = self._make_files_list(["video.mp4", "meta.json", "audio.mp3", "r.csv", "r.json", "r.png"])

        def fake_download(file_id, output_path):
            with open(output_path, "wb") as f:
                f.write(b"x" * 200)
            return True

        mock_dl.side_effect = fake_download
        ok = download_folder_files("test_folder", files, str(tmp_path))

        assert mock_dl.call_count == 6
        assert ok is True

    @patch("download_data.download_single_file")
    def test_skips_existing_files(self, mock_dl, tmp_path):
        folder = tmp_path / "test_folder"
        folder.mkdir()
        for name in ["video.mp4", "meta.json", "audio.mp3", "r.csv", "r.json"]:
            (folder / name).write_bytes(b"x" * 200)

        files = self._make_files_list(["video.mp4", "meta.json", "audio.mp3", "r.csv", "r.json"])

        def fake_download(file_id, output_path):
            with open(output_path, "wb") as f:
                f.write(b"x" * 200)
            return True

        mock_dl.side_effect = fake_download
        download_folder_files("test_folder", files, str(tmp_path))
        assert mock_dl.call_count == 0

    @patch("download_data.download_single_file", return_value=False)
    def test_returns_false_on_download_failure(self, mock_dl, tmp_path):
        files = self._make_files_list(["video.mp4", "meta.json", "audio.mp3", "r.csv", "r.json", "r.png"])

        ok = download_folder_files("test_folder", files, str(tmp_path))
        assert ok is False

    @patch("download_data.download_single_file")
    def test_keeps_existing_files_on_partial(self, mock_dl, tmp_path):
        """Partially downloaded folder should keep existing files and only download missing ones."""
        folder = tmp_path / "test_folder"
        folder.mkdir()
        (folder / "audio.mp3").write_bytes(b"x" * 200)

        files = self._make_files_list(["video.mp4", "meta.json", "audio.mp3", "r.csv", "r.json", "r.png"])

        def fake_download(file_id, output_path):
            with open(output_path, "wb") as f:
                f.write(b"x" * 200)
            return True

        mock_dl.side_effect = fake_download
        download_folder_files("test_folder", files, str(tmp_path))

        assert (folder / "audio.mp3").exists()
        downloaded_filenames = [os.path.basename(call.args[1]) for call in mock_dl.call_args_list]
        assert "audio.mp3" not in downloaded_filenames
        assert "video.mp4" in downloaded_filenames
