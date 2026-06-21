"""Decode audio files into in-memory WAV bytes for the whisper.cpp server.

No temporary file is written: ffmpeg streams the converted WAV to stdout and
the bytes are returned directly, so an interrupted run can never orphan a
wav file on disk.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# whisper.cpp models operate on 16 kHz mono PCM audio.
WHISPER_SAMPLE_RATE = 16_000

# Canonical PCM WAV header length (RIFF + fmt + data chunk headers).
_WAV_HEADER_SIZE = 44


class AudioConversionError(RuntimeError):
    """Raised when ffmpeg cannot decode a source audio file."""


def _ffmpeg_path() -> str:
    """Return the absolute path to the ffmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = "ffmpeg was not found on PATH; install it to decode mp3 files"
        raise AudioConversionError(msg)
    return ffmpeg


def decode_to_wav_bytes(
    source: Path,
    *,
    sample_rate: int = WHISPER_SAMPLE_RATE,
) -> bytes:
    """Decode ``source`` into 16 kHz mono PCM WAV bytes held in memory.

    Raises:
        AudioConversionError: If ffmpeg is missing, exits non-zero, or emits
            data that is not a usable WAV stream.

    """
    command = [
        _ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-bitexact",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        msg = f"ffmpeg failed for {source} (exit {result.returncode}): {stderr}"
        raise AudioConversionError(msg)
    return _finalize_wav(result.stdout, source)


def _finalize_wav(data: bytes, source: Path) -> bytes:
    """Repair the RIFF/data chunk sizes ffmpeg leaves unset when piping.

    Writing WAV to a non-seekable pipe leaves the RIFF and data chunk sizes as
    placeholders, which strict readers reject. With ``-bitexact`` the data
    chunk is last, so both sizes are recomputed from the actual byte length.
    """
    if len(data) < _WAV_HEADER_SIZE or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        msg = f"ffmpeg produced invalid WAV data for {source}"
        raise AudioConversionError(msg)
    data_pos = data.find(b"data", 12)
    if data_pos == -1:
        msg = f"ffmpeg WAV output for {source} has no data chunk"
        raise AudioConversionError(msg)
    buf = bytearray(data)
    struct.pack_into("<I", buf, 4, len(buf) - 8)
    struct.pack_into("<I", buf, data_pos + 4, len(buf) - (data_pos + 8))
    return bytes(buf)
