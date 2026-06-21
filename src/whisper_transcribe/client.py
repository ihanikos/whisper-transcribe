"""HTTP client for the local whisper.cpp ``/inference`` endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import httpx

if TYPE_CHECKING:
    from types import TracebackType

DEFAULT_SERVER_URL = "http://127.0.0.1:8178"

# Large files can take minutes to transcribe; give the server room.
DEFAULT_TIMEOUT_SECONDS = 600.0


class TranscriptionError(RuntimeError):
    """Raised when the whisper server cannot be reached or returns an error."""


class WhisperClient:
    """Thin client over the whisper.cpp ``/inference`` endpoint."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        """Create a client targeting ``server_url`` (e.g. ``http://127.0.0.1:8178``)."""
        self._server_url = server_url.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        response_format: str = "vtt",
        temperature: float = 0.0,
    ) -> str:
        """Transcribe in-memory WAV bytes and return the body in ``response_format``.

        Raises:
            TranscriptionError: If the request fails or the server errors.

        """
        url = f"{self._server_url}/inference"
        data = {"response_format": response_format, "temperature": str(temperature)}
        files = {"file": (filename, audio, "audio/wav")}
        try:
            response = self._client.post(url, files=files, data=data)
        except httpx.HTTPError as exc:
            msg = f"request to {url} failed: {exc}"
            raise TranscriptionError(msg) from exc
        if response.status_code != httpx.codes.OK:
            body = response.text.strip()
            msg = f"whisper server returned {response.status_code}: {body}"
            raise TranscriptionError(msg)
        return response.text

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> Self:
        """Enter the runtime context and return the client."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the client on context exit."""
        self.close()
