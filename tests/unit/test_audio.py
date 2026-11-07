"""Tests for in-memory decoding and silence-aligned chunking via PyAV."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pytest

from tests.mp3gen import write_mp3
from whisper_transcribe.audio import (
    WHISPER_SAMPLE_RATE,
    AudioConversionError,
    decode_to_wav_bytes,
    plan_chunks,
    probe_duration_seconds,
)

if TYPE_CHECKING:
    from pathlib import Path

# mp3 frames add ~0.15 s of slop at encode/seek boundaries.
SLOP_SECONDS = 0.15
BYTES_PER_SECOND = WHISPER_SAMPLE_RATE * 2


@pytest.fixture()
def mixed_mp3(tmp_path: Path) -> Path:
    """Provide a 6 s mp3 made of 3 s tone followed by 3 s silence."""
    path = tmp_path / "mixed.mp3"
    write_mp3(path, [(3.0, False), (3.0, True)])
    return path


def test_probe_duration_matches_written_audio(mixed_mp3: Path) -> None:
    """probe_duration_seconds returns the duration the file was written with."""
    duration = probe_duration_seconds(mixed_mp3)
    assert duration is not None
    assert abs(duration - 6.0) < SLOP_SECONDS


def test_decode_whole_file_returns_valid_wav(mixed_mp3: Path) -> None:
    """decode_to_wav_bytes returns a 16 kHz mono s16 WAV of the full file."""
    wav = decode_to_wav_bytes(mixed_mp3)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack_from("<I", wav, 4)[0] == len(wav) - 8
    assert struct.unpack_from("<H", wav, 22)[0] == 1  # mono
    assert struct.unpack_from("<I", wav, 24)[0] == WHISPER_SAMPLE_RATE
    assert struct.unpack_from("<H", wav, 34)[0] == 16  # bits per sample
    data_size = struct.unpack_from("<I", wav, 40)[0]
    assert data_size == len(wav) - 44
    assert abs(data_size - 6 * BYTES_PER_SECOND) < SLOP_SECONDS * BYTES_PER_SECOND


def test_decode_window_returns_requested_range(mixed_mp3: Path) -> None:
    """decode_to_wav_bytes decodes only [start, end) seconds."""
    wav = decode_to_wav_bytes(mixed_mp3, start=1.0, end=3.0)
    data_size = struct.unpack_from("<I", wav, 40)[0]
    assert abs(data_size - 2 * BYTES_PER_SECOND) < SLOP_SECONDS * BYTES_PER_SECOND


def test_decode_raises_for_non_audio_file(tmp_path: Path) -> None:
    """decode_to_wav_bytes raises AudioConversionError for undecodable input."""
    path = tmp_path / "broken.mp3"
    path.write_bytes(b"this is not audio")
    with pytest.raises(AudioConversionError):
        decode_to_wav_bytes(path)


def test_plan_chunks_snaps_boundary_to_silence(tmp_path: Path) -> None:
    """plan_chunks moves a grid boundary onto the nearest silent window."""
    path = tmp_path / "gap.mp3"
    duration = write_mp3(path, [(5.0, False), (4.0, True), (5.0, False)])
    chunks = plan_chunks(path, duration, chunk_seconds=6.0, tolerance=3.0)
    assert len(chunks) == 2
    boundary = chunks[0][1]
    assert boundary == pytest.approx(chunks[1][0])
    # Silence runs 5..9 s; the midpoint nearest the 6 s grid point wins.
    assert 5.5 < boundary < 6.5


def test_plan_chunks_falls_back_to_grid_without_silence(tmp_path: Path) -> None:
    """plan_chunks keeps the grid point when no silence is nearby."""
    path = tmp_path / "tone.mp3"
    duration = write_mp3(path, [(12.0, False)])
    chunks = plan_chunks(path, duration, chunk_seconds=5.0, tolerance=2.0)
    assert chunks[0] == (0.0, 5.0)
    assert chunks[1] == (5.0, duration)


def test_plan_chunks_single_chunk_for_short_file(tmp_path: Path) -> None:
    """plan_chunks returns one span covering the file when no boundary fits."""
    path = tmp_path / "short.mp3"
    duration = write_mp3(path, [(10.0, False)])
    chunks = plan_chunks(path, duration)
    assert len(chunks) == 1
    assert chunks[0] == (0.0, duration)
