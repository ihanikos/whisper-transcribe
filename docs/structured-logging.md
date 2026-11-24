# Structured logging specification

whisper-transcribe writes each application event through `structlog`. It keeps
human-readable log output on stderr and stores a JSON Lines copy on disk.

## Log file

`platformdirs` chooses a per-user log directory. The active file is named
`whisper-transcribe.log`. On Linux, the default path is
`~/.local/state/whisper-transcribe/log/whisper-transcribe.log`. On other
platforms, `platformdirs.user_log_dir("whisper-transcribe")` selects the
platform's standard user log directory. The CLI prints the resolved path in
`--help` output.

The application creates the directory when it starts. It rotates the file at
5 MiB and keeps three rotated files, using UTF-8 encoding. The active file and
backups use the `whisper-transcribe.log`, `whisper-transcribe.log.1`,
`whisper-transcribe.log.2`, and `whisper-transcribe.log.3` names.

If the directory or active file cannot be opened, the CLI keeps logging to
stderr and prints a warning with the path and filesystem error.

## Record format

Each file record is one JSON object followed by a newline. Records include:

| Field | Meaning |
|---|---|
| `event` | Stable event name, such as `batch_started` or `transcription_failed` |
| `level` | Log level, such as `info`, `warning`, or `error` |
| `timestamp` | UTC ISO-8601 timestamp |
| `logger` | Python logger name |
| other fields | Event-specific values such as file path, status, and elapsed time |
| `exception` | Formatted traceback when an event includes exception information |

Normal runs log at `INFO`. The `--verbose` option also records `DEBUG` events.
The CLI result table and tally remain on stdout. Log records remain on stderr
and in the JSON Lines file.

## Finding errors

On Linux, search the active log and its backups for error-level records with:

```bash
rg '"level": "error"' ~/.local/state/whisper-transcribe/log/whisper-transcribe.log*
```

Look at the `event`, `file`, `error`, and `exception` fields for the failed
operation and its traceback. Use the path printed by `whisper-transcribe
--help` on platforms with a different user log directory.

## Source specifications

This specification adapts the following ai-dev references for a local CLI
application with rotating per-user files:

- `docs/review-guidance/python/structured-logging.md` requires production
  Python events to use `structlog` rather than `print()` or direct stdlib
  logging calls.
- `mother/lib/logs/docs/SPEC.md` defines JSON Lines records with a log level
  and UTC ISO-8601 timestamp. Its stdout and package-specific requirements do
  not apply here.
- `private-claude-plugins/plugins/command-guards/hooks/check_forbidden_commands.py`
  and `private-claude-plugins/plugins/output-guards/hooks/check_output_guards.py`
  use a 5 MiB `RotatingFileHandler` with three backups. whisper-transcribe
  follows that size and retention policy.
