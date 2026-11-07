# whisper-transcribe

Batch-transcribes .mp3/.MP3 files via the local whisper.cpp server.

## Architecture

- `audio.py` — in-process decode via PyAV (no ffmpeg binary); silence-aligned time chunking
- `client.py` — thin httpx wrapper for the `/inference` endpoint
- `core.py` — orchestration: find, chunk, transcribe, merge timestamps, write
- `__main__.py` — argparse CLI + rich summary table

## Key invariants

**No temporary wav file is ever written to disk.**

**Memory stays bounded regardless of recording length.** PyAV decodes one
time chunk at a time (boundaries snapped to nearby silence); the chunk's
PCM is wrapped in a WAV header in memory and posted directly to the
whisper server. Only the transcript output file (`.vtt`, `.srt`, etc.) is
ever written.

## Dev commands

```bash
hatch run test        # run all tests
hatch run lint        # ruff check
hatch run format      # ruff format
hatch run typecheck   # mypy --strict
hatch run run         # run CLI
```

## Install

```bash
pipx install .          # first install
pipx install --force .  # reinstall after changes
```
