"""Tiny mp3 generator for tests — encodes via PyAV, needs no external tools."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import av

if TYPE_CHECKING:
    from pathlib import Path

RATE = 44_100
AMPLITUDE = 12_000  # well above the silence threshold
_FRAME_BYTES = 2048


def write_mp3(path: Path, segments: list[tuple[float, bool]]) -> float:
    """Write an mp3 of ``(seconds, silent)`` segments and return its duration."""
    pcm = b"".join(_pcm(seconds, silent=silent) for seconds, silent in segments)
    container = av.open(str(path), "w")
    stream = container.add_stream("libmp3lame", rate=RATE)
    stream.layout = "mono"
    for offset in range(0, len(pcm), _FRAME_BYTES):
        chunk = pcm[offset : offset + _FRAME_BYTES]
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(chunk) // 2)
        frame.planes[0].update(chunk)
        frame.sample_rate = RATE
        frame.pts = offset // 2
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return len(pcm) / 2 / RATE


def _pcm(seconds: float, *, silent: bool) -> bytes:
    """Return ``seconds`` of square-wave PCM, or silence."""
    count = int(seconds * RATE)
    if silent:
        return b"\x00\x00" * count
    return _square_wave_chunk(count)


def _square_wave_chunk(count: int) -> bytes:
    return b"".join(
        struct.pack(
            "<h",
            AMPLITUDE if (i // 20) % 2 == 0 else -AMPLITUDE,
        )
        for i in range(count)
    )
