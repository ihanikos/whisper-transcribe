"""Batch transcription orchestration: find mp3s, convert, transcribe, write."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import structlog

from whisper_transcribe.audio import AudioConversionError, decode_to_wav_bytes
from whisper_transcribe.client import TranscriptionError

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()


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

    The decoded WAV is held in memory and never written to disk; only the
    transcript output file is created. Existing transcripts are left untouched
    unless ``options.overwrite`` is set.
    """
    output = transcript_path(mp3_path, options.response_format)
    if output.exists() and not options.overwrite:
        return FileResult(mp3_path, output, "skipped", "output already exists")
    wav_bytes = decode_to_wav_bytes(mp3_path)
    transcript = client.transcribe(
        wav_bytes,
        response_format=options.response_format,
        temperature=options.temperature,
    )
    output.write_text(transcript, encoding="utf-8")
    return FileResult(mp3_path, output, "written")


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
