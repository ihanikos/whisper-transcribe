# whisper-transcribe

Batch-transcribes .mp3/.MP3 files via the local whisper.cpp server.

## Architecture

- `audio.py` — streaming pipeline: ffmpeg → stdout → bytes (no temp file)
- `client.py` — thin httpx wrapper for the `/inference` endpoint
- `core.py` — orchestration: find, skip, transcribe, write
- `__main__.py` — argparse CLI + rich summary table

## Key invariant

**No temporary wav file is ever written to disk.**

ffmpeg streams PCM audio to stdout; the bytes are patched in memory
(`_finalize_wav`) and posted directly to the whisper server. Only the
transcript output file (`.vtt`, `.srt`, etc.) is ever written.

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
