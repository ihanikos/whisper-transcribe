"""Integration tests for WhisperClient using a mock HTTP transport."""

from __future__ import annotations

import httpx
import pytest

from whisper_transcribe.client import TranscriptionError, WhisperClient

CANNED_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world.\n"
FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEdata\x00\x00\x00\x00"


def _make_client(handler: httpx.MockTransport) -> WhisperClient:
    return WhisperClient(client=httpx.Client(transport=handler))


def test_transcribe_returns_server_body() -> None:
    """WhisperClient.transcribe returns the response body on success."""

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, text=CANNED_VTT)

    client = _make_client(httpx.MockTransport(handler))
    result = client.transcribe(FAKE_WAV)
    assert result == CANNED_VTT


def test_transcribe_raises_on_non_200() -> None:
    """WhisperClient.transcribe raises TranscriptionError on a non-200 response."""

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(500, text="internal error")

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(TranscriptionError, match="500"):
        client.transcribe(FAKE_WAV)


def test_transcribe_raises_on_connection_error() -> None:
    """WhisperClient.transcribe raises TranscriptionError when the server is unreachable."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(TranscriptionError, match="failed"):
        client.transcribe(FAKE_WAV)


def test_client_context_manager_closes_on_exit() -> None:
    """WhisperClient closes the HTTP client when used as a context manager."""

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, text=CANNED_VTT)

    with WhisperClient(client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        result = client.transcribe(FAKE_WAV)
    assert result == CANNED_VTT


def test_transcribe_sends_correct_format_parameter() -> None:
    """WhisperClient.transcribe forwards response_format to the server."""
    received: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        received["body"] = body.decode("latin-1")
        return httpx.Response(200, text="1\n00:00:00,000 --> 00:00:01,000\nHi\n")

    client = _make_client(httpx.MockTransport(handler))
    client.transcribe(FAKE_WAV, response_format="srt")
    assert "srt" in received["body"]
