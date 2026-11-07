"""Decode and chunk audio files in memory for the whisper.cpp server.

PyAV (FFmpeg library bindings) decodes in process — no ffmpeg binary is
executed and no temporary file is written. Long recordings are split into
time chunks with boundaries snapped to nearby silence, so words are not
cut and only one chunk of PCM is ever held in memory.
"""

from __future__ import annotations

import array
import struct
from itertools import pairwise
from typing import TYPE_CHECKING

import av

if TYPE_CHECKING:
    from pathlib import Path

# whisper.cpp models operate on 16 kHz mono PCM audio.
WHISPER_SAMPLE_RATE = 16_000

# Chunking defaults: target chunk length, how far a boundary may move to
# reach silence, and the minimum spacing between boundaries / from the end.
CHUNK_SECONDS = 300.0
CHUNK_TOLERANCE = 30.0
MIN_CHUNK_SPACING = 30.0

# Silence detection: peak absolute amplitude below threshold (16-bit PCM,
# ~ -36 dBFS) sustained for at least SILENCE_MIN_SECONDS.
SILENCE_THRESHOLD = 512
SILENCE_MIN_SECONDS = 0.3

# Canonical PCM WAV header length (RIFF + fmt + data chunk headers).
_WAV_HEADER_SIZE = 44


class AudioConversionError(RuntimeError):
    """Raised when an audio file cannot be decoded or chunked."""


def probe_duration_seconds(source: Path) -> float | None:
    """Return duration in seconds, or None when the file does not report one."""
    try:
        with av.open(str(source)) as container:
            if not container.streams.audio:
                msg = f"{source} contains no audio stream"
                raise AudioConversionError(msg)
            if container.duration is not None:
                return container.duration / 1_000_000
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except av.FFmpegError as exc:
        msg = f"cannot open {source}: {exc}"
        raise AudioConversionError(msg) from exc
    return None


def decode_to_wav_bytes(
    source: Path,
    *,
    start: float = 0.0,
    end: float | None = None,
) -> bytes:
    """Decode ``[start, end)`` seconds of ``source`` into in-memory WAV bytes.

    The audio is resampled to 16 kHz mono PCM; ``end=None`` decodes to the
    end of the file. Raises AudioConversionError on undecodable input.
    """
    return _wav_bytes(_decode_pcm(source, start, end))


def plan_chunks(
    source: Path,
    duration: float,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
    tolerance: float = CHUNK_TOLERANCE,
) -> list[tuple[float, float]]:
    """Split ``[0, duration)`` into chunks, snapping boundaries to silence.

    Each boundary on the ``chunk_seconds`` grid moves to the nearest silent
    window midpoint within ``tolerance`` seconds, so chunk edges fall in
    pauses instead of mid-word. Falls back to the grid point when no
    silence is nearby.
    """
    spacing = min(MIN_CHUNK_SPACING, chunk_seconds / 2)
    boundaries: list[float] = []
    previous = 0.0
    target = chunk_seconds
    while target < duration - spacing:
        point = _snap_to_silence(source, target, tolerance)
        point = min(point, duration - spacing)
        point = max(point, previous + spacing)
        if point >= duration - spacing:
            break
        boundaries.append(point)
        previous = point
        target += chunk_seconds
    edges = [0.0, *boundaries, duration]
    return list(pairwise(edges))


def _decode_pcm(source: Path, start: float, end: float | None) -> bytes:
    """Decode ``[start, end)`` into raw 16 kHz mono s16le PCM bytes."""
    try:
        with av.open(str(source)) as container:
            if not container.streams.audio:
                msg = f"{source} contains no audio stream"
                raise AudioConversionError(msg)
            stream = container.streams.audio[0]
            if stream.time_base is None:
                msg = f"{source} reports no timebase"
                raise AudioConversionError(msg)
            container.seek(max(0, int(start / stream.time_base)), stream=stream)
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=WHISPER_SAMPLE_RATE,
            )
            chunks: list[bytes] = []
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_time = float(frame.pts * stream.time_base)
                if frame_time < start:
                    continue
                if end is not None and frame_time >= end:
                    break
                chunks.extend(_resampled_bytes(resampler, frame))
            chunks.extend(_resampled_bytes(resampler, None))
    except av.FFmpegError as exc:
        msg = f"failed to decode {source}: {exc}"
        raise AudioConversionError(msg) from exc
    return b"".join(chunks)


def _resampled_bytes(
    resampler: av.AudioResampler,
    frame: av.AudioFrame | None,
) -> list[bytes]:
    """Resample one frame (None flushes) and return its PCM byte chunks."""
    resampled = resampler.resample(frame)
    if resampled is None:
        return []
    frames = resampled if isinstance(resampled, list) else [resampled]
    # bytes(plane) over-reads buffer alignment padding, so cut to samples.
    return [bytes(rf.planes[0])[: rf.samples * 2] for rf in frames]


def _snap_to_silence(source: Path, target: float, tolerance: float) -> float:
    """Return the silence midpoint nearest ``target`` within tolerance."""
    # With no silent window nearby, the grid point itself is used.
    window_start = max(0.0, target - tolerance)
    pcm = _decode_pcm(source, window_start, target + tolerance)
    candidates = _silence_midpoints(pcm, window_start)
    if not candidates:
        return target
    return min(candidates, key=lambda t: abs(t - target))


def _silence_midpoints(pcm: bytes, window_start: float) -> list[float]:
    """Return absolute times of silent windows of SILENCE_MIN_SECONDS.

    A window is silent when its peak absolute sample stays below
    SILENCE_THRESHOLD. Samples are little-endian s16; all supported
    platforms are little-endian, so ``array("h")` reads them natively.
    """
    samples = array.array("h", pcm)
    size = int(WHISPER_SAMPLE_RATE * SILENCE_MIN_SECONDS)
    midpoints = []
    for offset in range(0, len(samples) - size + 1, size):
        window = samples[offset : offset + size]
        lowest, highest = min(window), max(window)
        if max(highest, -lowest) < SILENCE_THRESHOLD:
            midpoints.append(window_start + (offset + size / 2) / WHISPER_SAMPLE_RATE)
    return midpoints


def _wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw s16 mono 16 kHz PCM in a canonical WAV byte string."""
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        _WAV_HEADER_SIZE - 8 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM
        1,  # mono
        WHISPER_SAMPLE_RATE,
        WHISPER_SAMPLE_RATE * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        len(pcm),
    )
    return header + pcm
