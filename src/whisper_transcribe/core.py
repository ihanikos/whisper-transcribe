"""Batch transcription orchestration: find mp3s, chunk, transcribe, merge, write."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import structlog

from whisper_transcribe.audio import (
    CHUNK_SECONDS,
    AudioConversionError,
    decode_to_wav_bytes,
    plan_chunks,
    probe_duration_seconds,
)
from whisper_transcribe.client import TranscriptionError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

log = structlog.get_logger()

# Cue timestamps in vtt/srt: optional hours, minutes, seconds, ms.
_CLOCK_RE = re.compile(r"(?:(\d+):)?(\d+):(\d{2})([.,])(\d{3})")
# Length of "HH:MM:SS.mmm" after splitting; "MM:SS.mmm" has one part less.
_CLOCK_WITH_HOURS_PARTS = 4


class Transcriber(Protocol):
    """Anything that can turn in-memory WAV bytes into a transcript string."""

    def transcribe(
        self,
        audio: bytes,
        *,
        response_format: str = ...,
        temperature: float = ...,
    ) -> str:
        """Return the transcript of ``audio`` in ``response_format``."""
        ...


# Maps a whisper.cpp response_format to the transcript file extension.
EXTENSION_BY_FORMAT: dict[str, str] = {
    "vtt": ".vtt",
    "srt": ".srt",
    "text": ".txt",
    "json": ".json",
    "verbose_json": ".json",
}

SUPPORTED_FORMATS: tuple[str, ...] = tuple(EXTENSION_BY_FORMAT)

FileStatus = Literal["written", "skipped", "failed"]


@dataclass(frozen=True)
class TranscribeOptions:
    """Settings controlling a batch transcription run."""

    response_format: str = "vtt"
    temperature: float = 0.0
    recursive: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class FileResult:
    """Outcome of transcribing a single mp3 file."""

    source: Path
    output: Path
    status: FileStatus
    detail: str = ""


def find_mp3s(directory: Path, *, recursive: bool = False) -> list[Path]:
    """Return sorted mp3 files in ``directory`` (recursively if requested)."""
    walker = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path for path in walker if path.is_file() and path.suffix.lower() == ".mp3"
    )


def transcript_path(mp3_path: Path, response_format: str) -> Path:
    """Return the transcript output path for ``mp3_path`` and ``response_format``."""
    return mp3_path.with_suffix(EXTENSION_BY_FORMAT[response_format])


def transcribe_file(
    mp3_path: Path,
    client: Transcriber,
    options: TranscribeOptions,
) -> FileResult:
    """Transcribe a single mp3 and write the transcript next to it.

    Long recordings are split into silence-aligned time chunks; each chunk
    is decoded to WAV in memory (never on disk) and posted separately, so
    memory stays bounded regardless of recording length. Chunk responses
    are merged with timestamps shifted to file-absolute positions. Existing
    transcripts are left untouched unless ``options.overwrite`` is set.
    """
    output = transcript_path(mp3_path, options.response_format)
    if output.exists() and not options.overwrite:
        return FileResult(mp3_path, output, "skipped", "output already exists")
    duration = probe_duration_seconds(mp3_path)
    spans: Sequence[tuple[float, float | None]]
    if duration is not None and duration > CHUNK_SECONDS:
        spans = plan_chunks(mp3_path, duration)
        log.info("chunking", file=str(mp3_path), chunks=len(spans), duration=duration)
    else:
        # ponytail: unknown duration -> whole file in one chunk
        spans = [(0.0, duration)]
    parts: list[str] = []
    for start, end in spans:
        wav_bytes = decode_to_wav_bytes(mp3_path, start=start, end=end)
        parts.append(
            client.transcribe(
                wav_bytes,
                response_format=options.response_format,
                temperature=options.temperature,
            ),
        )
    transcript = merge_transcripts(
        parts,
        [start for start, _ in spans],
        options.response_format,
    )
    output.write_text(transcript, encoding="utf-8")
    return FileResult(mp3_path, output, "written")


def merge_transcripts(
    parts: list[str],
    starts: list[float],
    response_format: str,
) -> str:
    """Merge per-chunk server responses into one transcript.

    Timestamps in each part are shifted forward by the chunk's start offset
    so they point at file-absolute positions. A single part is returned
    verbatim.
    """
    if len(parts) == 1:
        return parts[0]
    if response_format in ("vtt", "srt"):
        return _merge_cues(parts, starts, vtt=response_format == "vtt")
    if response_format == "text":
        return "\n".join(part.strip() for part in parts) + "\n"
    return _merge_json(parts, starts)


def _merge_cues(parts: list[str], starts: list[float], *, vtt: bool) -> str:
    """Shift, strip headers from, and join vtt/srt cue blocks."""
    blocks: list[str] = []
    for part, start in zip(parts, starts, strict=True):
        lines = part.splitlines()
        if vtt:
            lines = _drop_vtt_header(lines)
        lines = [
            _shift_clock_line(line, start) if " --> " in line else line
            for line in lines
        ]
        blocks.append("\n".join(lines).strip())
    if not vtt:
        blocks = _renumber_srt(blocks)
    body = "\n\n".join(blocks) + "\n"
    return f"WEBVTT\n\n{body}" if vtt else body


def _drop_vtt_header(lines: list[str]) -> list[str]:
    """Drop the WEBVTT header line and everything up to the first blank line."""
    if lines and lines[0].startswith("WEBVTT"):
        for index, line in enumerate(lines[1:], 1):
            if not line.strip():
                return lines[index + 1 :]
        return []
    return lines


def _renumber_srt(blocks: list[str]) -> list[str]:
    """Re-sequence srt cue indices so they stay unique after merging."""
    renumbered = []
    for number, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if lines and lines[0].isdigit():
            lines[0] = str(number)
        renumbered.append("\n".join(lines))
    return renumbered


def _shift_clock_line(line: str, offset: float) -> str:
    """Shift every clock timestamp on a cue timing line by ``offset`` seconds."""

    def replace(match: re.Match[str]) -> str:
        seconds = _clock_to_seconds(
            f"{match.group(1) or 0}:{match.group(2)}:{match.group(3)}.{match.group(5)}",
        )
        return _seconds_to_clock(seconds + offset).replace(".", match.group(4))

    return _CLOCK_RE.sub(replace, line)


def _clock_to_seconds(clock: str) -> float:
    """Convert ``[HH:]MM:SS.mmm`` to seconds."""
    parts = re.split(r"[:.,]", clock)
    hours = int(parts[0]) if len(parts) == _CLOCK_WITH_HOURS_PARTS else 0
    minutes, seconds, millis = (int(part) for part in parts[-3:])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def _seconds_to_clock(total: float) -> str:
    """Convert seconds to ``HH:MM:SS.mmm``."""
    whole, millis = divmod(round(total * 1000), 1000)
    hours, rem = divmod(whole, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _merge_json(parts: list[str], starts: list[float]) -> str:
    """Merge json / verbose_json chunk responses into one object.

    Shifts ``segments[].start/end`` (seconds), ``transcription[].offsets``
    (milliseconds), and ``transcription[].timestamps`` (clock strings), and
    concatenates ``text``.
    """
    merged: dict[str, Any] = json.loads(parts[0])
    segments: list[Any] = []
    transcriptions: list[Any] = []
    texts: list[str] = []
    for part, start in zip(parts, starts, strict=True):
        data = json.loads(part)
        texts.append(str(data.get("text", "")))
        segments.extend(
            _shift_segment(segment, start) for segment in data.get("segments") or []
        )
        transcriptions.extend(
            _shift_transcription_entry(entry, start)
            for entry in data.get("transcription") or []
        )
    if segments:
        merged["segments"] = segments
        merged["text"] = " ".join(text for text in texts if text)
    if transcriptions:
        merged["transcription"] = transcriptions
    return json.dumps(merged, ensure_ascii=False, indent=2)


def _shift_segment(segment: dict[str, Any], start: float) -> dict[str, Any]:
    """Copy a json segment with ``start``/``end`` moved forward by ``start``."""
    shifted = dict(segment)
    for key in ("start", "end"):
        if key in shifted:
            shifted[key] = float(shifted[key]) + start
    return shifted


def _shift_transcription_entry(entry: dict[str, Any], start: float) -> dict[str, Any]:
    """Shift a verbose_json entry's offsets (ms) and timestamps (clock)."""
    offsets = entry.get("offsets")
    if offsets:
        for key in ("from", "to"):
            if key in offsets:
                offsets[key] += round(start * 1000)
    timestamps = entry.get("timestamps")
    if timestamps:
        for key in ("from", "to"):
            value = timestamps.get(key)
            if isinstance(value, str):
                timestamps[key] = _seconds_to_clock(_clock_to_seconds(value) + start)
    return entry


def transcribe_directory(
    directory: Path,
    client: Transcriber,
    options: TranscribeOptions,
) -> list[FileResult]:
    """Transcribe every mp3 in ``directory``; a failure on one file is recorded.

    Logs the start of the batch, the start of each file, and the elapsed time
    once each file completes. A failure does not abort the batch: the offending
    file gets a ``failed`` result and processing continues with the next file.
    """
    mp3_paths = find_mp3s(directory, recursive=options.recursive)
    log.info(
        "batch_started",
        directory=str(directory),
        files=len(mp3_paths),
        response_format=options.response_format,
        recursive=options.recursive,
    )
    results: list[FileResult] = []
    for mp3_path in mp3_paths:
        log.info("processing", file=str(mp3_path))
        started = time.monotonic()
        try:
            result = transcribe_file(mp3_path, client, options)
        except (AudioConversionError, TranscriptionError) as exc:
            elapsed = round(time.monotonic() - started, 2)
            log.exception(
                "transcription_failed",
                file=str(mp3_path),
                error=str(exc),
                elapsed_seconds=elapsed,
            )
            output = transcript_path(mp3_path, options.response_format)
            result = FileResult(mp3_path, output, "failed", str(exc))
        else:
            elapsed = round(time.monotonic() - started, 2)
            log.info(
                "file_processed",
                file=str(mp3_path),
                status=result.status,
                elapsed_seconds=elapsed,
            )
        results.append(result)
    return results
