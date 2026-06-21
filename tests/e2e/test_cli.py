"""End-to-end tests for the whisper-transcribe CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from whisper_transcribe.__main__ import EXIT_BAD_USAGE, EXIT_FAILURES, EXIT_OK, main

CANNED_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world.\n"
FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEdata\x00\x00\x00\x00"


@pytest.fixture()
def _mock_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ffmpeg so CLI tests do not need a real ffmpeg binary."""
    monkeypatch.setattr("whisper_transcribe.audio.shutil.which", lambda _: "/usr/bin/ffmpeg")
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = FAKE_WAV
    monkeypatch.setattr("whisper_transcribe.audio.subprocess.run", lambda *_a, **_kw: proc)


@pytest.fixture()
def _mock_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the HTTP transport so CLI tests do not need a live whisper server."""

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, text=CANNED_VTT)

    monkeypatch.setattr(
        "whisper_transcribe.client.httpx.Client",
        lambda **_kw: httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_cli_transcribes_mp3_and_exits_zero(
    tmp_path: Path,
    _mock_ffmpeg: None,
    _mock_server: None,
) -> None:
    """CLI writes a VTT file and exits 0 when transcription succeeds."""
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    code = main([str(tmp_path)])
    assert code == EXIT_OK
    assert (tmp_path / "audio.vtt").read_text() == CANNED_VTT


def test_cli_exits_bad_usage_for_nonexistent_directory() -> None:
    """CLI exits EXIT_BAD_USAGE when the directory argument does not exist."""
    code = main(["/nonexistent/path/that/cannot/exist"])
    assert code == EXIT_BAD_USAGE


def test_cli_reports_no_mp3s_and_exits_zero(tmp_path: Path) -> None:
    """CLI exits 0 and prints a message when no mp3 files are found."""
    code = main([str(tmp_path)])
    assert code == EXIT_OK


def test_cli_exits_one_when_a_file_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI exits EXIT_FAILURES when at least one file fails to transcribe."""
    monkeypatch.setattr("whisper_transcribe.audio.shutil.which", lambda _: "/usr/bin/ffmpeg")
    proc = MagicMock()
    proc.returncode = 1
    proc.stderr = b"decode error"
    monkeypatch.setattr("whisper_transcribe.audio.subprocess.run", lambda *_a, **_kw: proc)

    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    code = main([str(tmp_path)])
    assert code == EXIT_FAILURES


def test_cli_skips_existing_transcripts_by_default(
    tmp_path: Path,
    _mock_ffmpeg: None,
    _mock_server: None,
) -> None:
    """CLI leaves existing transcripts untouched without --overwrite."""
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    vtt = tmp_path / "audio.vtt"
    vtt.write_text("existing content")
    main([str(tmp_path)])
    assert vtt.read_text() == "existing content"
