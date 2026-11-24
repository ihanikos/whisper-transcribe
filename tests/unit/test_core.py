"""Tests for batch transcription orchestration and chunk merging."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import structlog.testing

from whisper_transcribe.audio import AudioConversionError
from whisper_transcribe.client import TranscriptionError
from whisper_transcribe.core import (
    TranscribeOptions,
    find_mp3s,
    merge_transcripts,
    transcribe_directory,
    transcribe_file,
    transcript_path,
)

if TYPE_CHECKING:
    from pathlib import Path

CANNED_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world.\n"
CHUNK_VTT = "WEBVTT\n\n00:00:10.000 --> 00:00:11.000\nchunk text.\n"


class FakeClient:
    """Stub transcriber that returns a canned VTT string."""

    def transcribe(
        self,
        audio: bytes,  # noqa: ARG002
        *,
        response_format: str = "vtt",  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
    ) -> str:
        """Return the canned VTT string."""
        return CANNED_VTT


class ChunkVTTClient:
    """Stub transcriber that returns the same chunk-relative VTT per call."""

    def __init__(self, *, calls: int | None = None) -> None:
        """Count calls; raise TranscriptionError after ``calls`` successful ones."""
        self.calls = 0
        self.max_calls = calls

    def transcribe(
        self,
        audio: bytes,  # noqa: ARG002
        *,
        response_format: str = "vtt",  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
    ) -> str:
        """Return CHUNK_VTT, or raise once the call budget is spent."""
        self.calls += 1
        if self.max_calls is not None and self.calls > self.max_calls:
            msg = "server unreachable"
            raise TranscriptionError(msg)
        return CHUNK_VTT


class FailingClient:
    """Stub transcriber that always raises TranscriptionError."""

    def transcribe(
        self,
        audio: bytes,  # noqa: ARG002
        *,
        response_format: str = "vtt",  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
    ) -> str:
        """Raise TranscriptionError on every call."""
        msg = "server unreachable"
        raise TranscriptionError(msg)


def _fake_decode(path: Path, *, start: float = 0.0, end: float | None = None) -> bytes:  # noqa: ARG001
    """Return minimal fake WAV bytes without decoding audio."""
    return b"RIFF\x00\x00\x00\x00WAVEdata\x00\x00\x00\x00"


@pytest.fixture()
def _short_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch audio probing so every file needs a single chunk."""
    monkeypatch.setattr(
        "whisper_transcribe.core.probe_duration_seconds",
        lambda _p: 1.0,
    )
    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _fake_decode)


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
    _short_file: None,
    tmp_path: Path,
) -> None:
    """transcribe_file writes the transcript and no other files."""
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    result = transcribe_file(mp3, FakeClient(), TranscribeOptions())
    files_written = list(tmp_path.iterdir())
    assert result.status == "written"
    assert result.output.suffix == ".vtt"
    assert result.output.read_text() == CANNED_VTT
    assert len(files_written) == 2  # mp3 + vtt only


def test_transcribe_file_skips_existing_output(
    _short_file: None,
    tmp_path: Path,
) -> None:
    """transcribe_file skips a file whose transcript already exists."""
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    vtt = tmp_path / "audio.vtt"
    vtt.write_text("existing")
    result = transcribe_file(mp3, FakeClient(), TranscribeOptions())
    assert result.status == "skipped"
    assert vtt.read_text() == "existing"


def test_transcribe_file_overwrites_when_flag_set(
    _short_file: None,
    tmp_path: Path,
) -> None:
    """transcribe_file replaces an existing transcript when overwrite=True."""
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    vtt = tmp_path / "audio.vtt"
    vtt.write_text("old")
    result = transcribe_file(mp3, FakeClient(), TranscribeOptions(overwrite=True))
    assert result.status == "written"
    assert vtt.read_text() == CANNED_VTT


def test_transcribe_file_merges_chunks_with_shifted_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Chunk responses are merged with timestamps shifted by chunk start."""
    monkeypatch.setattr(
        "whisper_transcribe.core.probe_duration_seconds",
        lambda _p: 620.0,
    )
    monkeypatch.setattr(
        "whisper_transcribe.core.plan_chunks",
        lambda _p, _d: [(0.0, 300.0), (300.0, 620.0)],
    )
    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _fake_decode)
    mp3 = tmp_path / "long.mp3"
    mp3.touch()
    client = ChunkVTTClient()
    result = transcribe_file(mp3, client, TranscribeOptions())
    assert client.calls == 2
    assert result.status == "written"
    assert result.output.read_text() == (
        "WEBVTT\n\n"
        "00:00:10.000 --> 00:00:11.000\nchunk text.\n\n"
        "00:05:10.000 --> 00:05:11.000\nchunk text.\n"
    )


def test_transcribe_file_failure_mid_chunks_leaves_no_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A chunk failure fails the file and writes no partial transcript."""
    monkeypatch.setattr(
        "whisper_transcribe.core.probe_duration_seconds",
        lambda _p: 620.0,
    )
    monkeypatch.setattr(
        "whisper_transcribe.core.plan_chunks",
        lambda _p, _d: [(0.0, 300.0), (300.0, 620.0)],
    )
    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _fake_decode)
    mp3 = tmp_path / "long.mp3"
    mp3.touch()
    client = ChunkVTTClient(calls=1)
    with pytest.raises(TranscriptionError):
        transcribe_file(mp3, client, TranscribeOptions())
    assert not (tmp_path / "long.vtt").exists()


def test_transcribe_directory_continues_after_failure(
    _short_file: None,
    tmp_path: Path,
) -> None:
    """transcribe_directory marks failures and processes remaining files."""
    (tmp_path / "a.mp3").touch()
    (tmp_path / "b.mp3").touch()
    results = transcribe_directory(tmp_path, FailingClient(), TranscribeOptions())
    assert len(results) == 2
    assert all(r.status == "failed" for r in results)


def test_transcribe_directory_logs_start_and_duration(
    _short_file: None,
    tmp_path: Path,
) -> None:
    """transcribe_directory emits batch_started and file_processed log events."""
    mp3 = tmp_path / "audio.mp3"
    mp3.touch()
    with structlog.testing.capture_logs() as logs:
        transcribe_directory(tmp_path, FakeClient(), TranscribeOptions())
    events = [entry["event"] for entry in logs]
    assert "batch_started" in events
    processed = [e for e in logs if e["event"] == "file_processed"]
    assert len(processed) == 1
    assert "elapsed_seconds" in processed[0]


def test_merge_transcripts_single_part_is_verbatim() -> None:
    """A single chunk response is returned unchanged."""
    assert merge_transcripts([CANNED_VTT], [0.0], "vtt") == CANNED_VTT


def test_merge_transcripts_srt_shifts_and_renumbers() -> None:
    """Srt chunks get shifted timestamps and unique cue numbers."""
    part = "1\n00:00:10,000 --> 00:00:11,000\ntext\n"
    merged = merge_transcripts([part, part], [0.0, 60.0], "srt")
    assert merged == (
        "1\n00:00:10,000 --> 00:00:11,000\ntext\n\n"
        "2\n00:01:10,000 --> 00:01:11,000\ntext\n"
    )


def test_merge_transcripts_text_joins_lines() -> None:
    """Text chunks are joined with newlines."""
    merged = merge_transcripts(["first", "second"], [0.0, 300.0], "text")
    assert merged == "first\nsecond\n"


def test_merge_transcripts_json_shifts_segments() -> None:
    """Json chunk segments are shifted and concatenated."""
    first = '{"text": "hello", "segments": [{"start": 10.0, "end": 11.0}]}'
    second = '{"text": "world", "segments": [{"start": 5.0, "end": 6.0}]}'
    merged = json.loads(merge_transcripts([first, second], [0.0, 300.0], "json"))
    assert merged["text"] == "hello world"
    assert merged["segments"] == [
        {"start": 10.0, "end": 11.0},
        {"start": 305.0, "end": 306.0},
    ]


def test_merge_transcripts_verbose_json_shifts_offsets() -> None:
    """verbose_json offsets (ms) and timestamps (clock) are shifted."""
    first = (
        '{"transcription": [{"offsets": {"from": 0, "to": 1000},'
        ' "timestamps": {"from": "00:00:00.000", "to": "00:00:01.000"},'
        ' "text": "a"}]}'
    )
    second = (
        '{"transcription": [{"offsets": {"from": 500, "to": 1500},'
        ' "timestamps": {"from": "00:00:00.500", "to": "00:00:01.500"},'
        ' "text": "b"}]}'
    )
    merged = json.loads(merge_transcripts([first, second], [0.0, 60.0], "verbose_json"))
    second_entry = merged["transcription"][1]
    assert second_entry["offsets"] == {"from": 60500, "to": 61500}
    assert second_entry["timestamps"] == {
        "from": "00:01:00.500",
        "to": "00:01:01.500",
    }


def test_transcribe_directory_records_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An audio conversion failure is recorded as a failed result."""
    monkeypatch.setattr(
        "whisper_transcribe.core.probe_duration_seconds",
        lambda _p: 1.0,
    )

    def _boom(path: Path, *, start: float = 0.0, end: float | None = None) -> bytes:  # noqa: ARG001
        msg = "cannot decode"
        raise AudioConversionError(msg)

    monkeypatch.setattr("whisper_transcribe.core.decode_to_wav_bytes", _boom)
    (tmp_path / "a.mp3").touch()
    results = transcribe_directory(tmp_path, FakeClient(), TranscribeOptions())
    assert results[0].status == "failed"
