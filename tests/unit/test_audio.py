"""Tests for audio decoding via ffmpeg streaming."""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from whisper_transcribe.audio import (
    AudioConversionError,
    _WAV_HEADER_SIZE,
    decode_to_wav_bytes,
    _finalize_wav,
)


def _make_wav_bytes(*, data: bytes = b"\x00" * 32, zeroed_sizes: bool = False) -> bytes:
    """Build a minimal but valid PCM WAV byte string."""
    fmt_chunk = (
        b"fmt "
        + struct.pack("<I", 16)    # chunk size
        + struct.pack("<H", 1)     # PCM
        + struct.pack("<H", 1)     # mono
        + struct.pack("<I", 16000) # sample rate
        + struct.pack("<I", 32000) # byte rate
        + struct.pack("<H", 2)     # block align
        + struct.pack("<H", 16)    # bits per sample
    )
    data_chunk = b"data" + struct.pack("<I", 0 if zeroed_sizes else len(data)) + data
    body = fmt_chunk + data_chunk
    riff_size = 0 if zeroed_sizes else (4 + len(body))
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + body


def test_decode_to_wav_bytes_calls_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """decode_to_wav_bytes invokes ffmpeg with the correct arguments."""
    fake_wav = _make_wav_bytes()
    monkeypatch.setattr("whisper_transcribe.audio.shutil.which", lambda _: "/usr/bin/ffmpeg")
    result = MagicMock()
    result.returncode = 0
    result.stdout = fake_wav
    monkeypatch.setattr(
        "whisper_transcribe.audio.subprocess.run",
        lambda cmd, **_kw: result,
    )
    mp3 = tmp_path / "test.mp3"
    mp3.touch()
    wav = decode_to_wav_bytes(mp3)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_decode_to_wav_bytes_raises_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """decode_to_wav_bytes raises AudioConversionError when ffmpeg is not on PATH."""
    monkeypatch.setattr("whisper_transcribe.audio.shutil.which", lambda _: None)
    mp3 = tmp_path / "test.mp3"
    mp3.touch()
    with pytest.raises(AudioConversionError, match="ffmpeg was not found"):
        decode_to_wav_bytes(mp3)


def test_decode_to_wav_bytes_raises_on_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """decode_to_wav_bytes raises AudioConversionError when ffmpeg exits non-zero."""
    monkeypatch.setattr("whisper_transcribe.audio.shutil.which", lambda _: "/usr/bin/ffmpeg")
    result = MagicMock()
    result.returncode = 1
    result.stderr = b"error: no such file"
    monkeypatch.setattr(
        "whisper_transcribe.audio.subprocess.run",
        lambda cmd, **_kw: result,
    )
    mp3 = tmp_path / "test.mp3"
    mp3.touch()
    with pytest.raises(AudioConversionError, match="ffmpeg failed"):
        decode_to_wav_bytes(mp3)


def test_finalize_wav_repairs_zeroed_chunk_sizes(tmp_path: Path) -> None:
    """_finalize_wav corrects placeholder chunk sizes from piped ffmpeg output."""
    source = tmp_path / "audio.wav"
    data_payload = b"\x00" * 64
    zeroed = _make_wav_bytes(data=data_payload, zeroed_sizes=True)
    fixed = _finalize_wav(zeroed, source)
    riff_size = struct.unpack_from("<I", fixed, 4)[0]
    assert riff_size == len(fixed) - 8


def test_finalize_wav_raises_on_invalid_header(tmp_path: Path) -> None:
    """_finalize_wav raises AudioConversionError on non-WAV data."""
    source = tmp_path / "audio.wav"
    with pytest.raises(AudioConversionError, match="invalid WAV"):
        _finalize_wav(b"INVALID" + b"\x00" * 40, source)


def test_finalize_wav_raises_when_data_chunk_missing(tmp_path: Path) -> None:
    """_finalize_wav raises AudioConversionError when there is no data chunk."""
    source = tmp_path / "audio.wav"
    # Valid RIFF/WAVE header but no "data" sub-chunk
    buf = bytearray(44)
    buf[0:4] = b"RIFF"
    buf[4:8] = struct.pack("<I", 36)
    buf[8:12] = b"WAVE"
    buf[12:16] = b"fmt "
    with pytest.raises(AudioConversionError, match="no data chunk"):
        _finalize_wav(bytes(buf), source)
