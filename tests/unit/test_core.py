"""Tests for batch transcription orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog.testing

from whisper_transcribe.audio import AudioConversionError
from whisper_transcribe.client import TranscriptionError
from whisper_transcribe.core import (
    FileResult,
    TranscribeOptions,
    find_mp3s,
    transcribe_directory,
    transcribe_file,
    transcript_path,
)

CANNED_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world.\n"


class FakeClient:
    """Stub transcriber that returns a canned VTT string."""

    def transcribe(
        self,
        audio: bytes,  # noqa: ARG002
        *,
        response_format: str = "vtt",  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
    ) -> str:
        return CANNED_VTT


class FailingClient:
    """Stub transcriber that always raises TranscriptionError."""

    def transcribe(
        self,
        audio: bytes,  # noqa: ARG002
        *,
        response_format: str = "vtt",  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
    ) -> str:
        msg = "server unreachable"
        raise TranscriptionError(msg)


def _fake_decode(path: Path) -> bytes:  # noqa: ARG001
    """Return minimal fake WAV bytes without calling ffmpeg."""
    return b"RIFF\x00\x00\x00\x00WAVEdata\x00\x00\x00\x00"


def test_find_mp3s_returns_sorted_list(tmp_path: Path) -> None:
    """find_mp3s returns lowercase .mp3 files in sorted order."""
    (tmp_path / "b.mp3").touch()
    (tmp_path / "a.mp3").touch()
    (tmp_path / "other.txt").touch()
    result = find_mp3s(tmp_path)
    assert [p.name for p in result] == ["a.mp3", "b.mp3"]


def test_find_mp3s_matches_uppercase_extension(tmp_path: Path) -> None:
    """find_mp3s includes files with .MP3 extension."""
    (tmp_path / "UPPER.MP3").touch()
    (tmp_path / "lower.mp3").touch()
    result = find_mp3s(tmp_path)
    assert len(result) == 2


def test_find_mp3s_recurses_when_requested(tmp_path: Path) -> None:
    """find_mp3s descends into subdirectories when recursive=True."""
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "nested.mp3").touch()
    assert find_mp3s(tmp_path, recursive=False) == []
    assert len(find_mp3s(tmp_path, recursive=True)) == 1


def test_transcript_path_uses_correct_extension(tmp_path: Path) -> None:
    """transcript_path returns a path with the right extension for each format."""
    mp3 = tmp_path / "recording.mp3"
    assert transcript_path(mp3, "vtt").suffix == ".vtt"
    assert transcript_path(mp3, "srt").suffix == ".srt"
    assert transcript_path(mp3, "text").suffix == ".txt"
    assert transcript_path(mp3, "json").suffix == ".json"


def test_transcribe_file_writes_only_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """transcribe_file writes the transcript and no other files."""
    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _fake_decode)
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    options = TranscribeOptions()
    result = transcribe_file(mp3, FakeClient(), options)
    files_written = list(tmp_path.iterdir())
    assert result.status == "written"
    assert result.output.suffix == ".vtt"
    assert result.output.read_text() == CANNED_VTT
    assert len(files_written) == 2  # mp3 + vtt only


def test_transcribe_file_skips_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """transcribe_file skips a file whose transcript already exists."""
    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _fake_decode)
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    vtt = tmp_path / "audio.vtt"
    vtt.write_text("existing")
    result = transcribe_file(mp3, FakeClient(), TranscribeOptions())
    assert result.status == "skipped"
    assert vtt.read_text() == "existing"


def test_transcribe_file_overwrites_when_flag_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """transcribe_file replaces an existing transcript when overwrite=True."""
    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _fake_decode)
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    vtt = tmp_path / "audio.vtt"
    vtt.write_text("old")
    result = transcribe_file(mp3, FakeClient(), TranscribeOptions(overwrite=True))
    assert result.status == "written"
    assert vtt.read_text() == CANNED_VTT


def test_transcribe_directory_continues_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """transcribe_directory marks failures and processes remaining files."""
    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _fake_decode)
    (tmp_path / "a.mp3").touch()
    (tmp_path / "b.mp3").touch()
    results = transcribe_directory(tmp_path, FailingClient(), TranscribeOptions())
    assert len(results) == 2
    assert all(r.status == "failed" for r in results)


def test_transcribe_directory_logs_start_and_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """transcribe_directory emits batch_started and file_processed log events."""
    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _fake_decode)
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    with structlog.testing.capture_logs() as logs:
        transcribe_directory(tmp_path, FakeClient(), TranscribeOptions())
    events = [entry["event"] for entry in logs]
    assert "batch_started" in events
    processed = [e for e in logs if e["event"] == "file_processed"]
    assert len(processed) == 1
    assert "elapsed_seconds" in processed[0]
