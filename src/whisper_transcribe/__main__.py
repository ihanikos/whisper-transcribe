"""CLI entry point for batch mp3 transcription via a whisper.cpp server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import structlog
from rich.console import Console
from rich.table import Table

from whisper_transcribe.client import DEFAULT_SERVER_URL, WhisperClient
from whisper_transcribe.core import (
    SUPPORTED_FORMATS,
    FileResult,
    TranscribeOptions,
    transcribe_directory,
)

console = Console()
err_console = Console(stderr=True)

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_BAD_USAGE = 2


def _configure_logging(*, verbose: bool) -> None:
    """Route structlog output to stderr at INFO, or DEBUG when verbose."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if verbose else logging.INFO,
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="whisper-transcribe",
        description=(
            "Transcribe every .mp3 in a directory via a local whisper.cpp "
            "server, writing one transcript per file."
        ),
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(),
        help="directory to scan for .mp3 files (default: current directory)",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="response_format",
        choices=SUPPORTED_FORMATS,
        default="vtt",
        help="transcript format (default: vtt)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="recurse into subdirectories",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help=f"whisper.cpp server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing transcripts (default: skip them)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _render_summary(results: list[FileResult]) -> None:
    """Print a per-file results table and a one-line tally to stdout."""
    table = Table(title="Transcription results")
    table.add_column("File", overflow="fold")
    table.add_column("Status")
    table.add_column("Output / detail", overflow="fold")
    status_style = {"written": "green", "skipped": "yellow", "failed": "red"}
    for result in results:
        detail = result.detail or str(result.output)
        table.add_row(
            result.source.name,
            f"[{status_style[result.status]}]{result.status}[/]",
            detail,
        )
    console.print(table)
    tally = dict.fromkeys(status_style, 0)
    for result in results:
        tally[result.status] += 1
    console.print(
        f"[green]{tally['written']} written[/], "
        f"[yellow]{tally['skipped']} skipped[/], "
        f"[red]{tally['failed']} failed[/]",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = _build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose)

    directory: Path = args.directory
    if not directory.is_dir():
        err_console.print(f"[red]error:[/] not a directory: {directory}")
        return EXIT_BAD_USAGE

    options = TranscribeOptions(
        response_format=args.response_format,
        temperature=args.temperature,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )
    with WhisperClient(args.server) as client:
        results = transcribe_directory(directory, client, options)

    if not results:
        console.print(f"No .mp3 files found in {directory}")
        return EXIT_OK

    _render_summary(results)
    failed = any(result.status == "failed" for result in results)
    return EXIT_FAILURES if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
