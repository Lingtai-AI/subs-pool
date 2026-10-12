"""Offline HTTP adapter wire smoke; injected transport never contacts Codex."""

from __future__ import annotations

import json

import httpx
import pytest

from subs_pool.modules.codex.errors import UpstreamTransportError
from subs_pool.modules.codex.upstream import CODEX_OFFICIAL_BASE_URL, CodexHTTPUpstream


def _sse_body(events: list[tuple[str, dict]]) -> bytes:
    body = b""
    for etype, data in events:
        body += f"event: {etype}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
    return body


@pytest.mark.asyncio
async def test_stream_posts_to_configured_url_with_expected_headers_and_forces_stream_true():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse_body(
                [("response.completed", {"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": []}})]
            ),
        )

    upstream = CodexHTTPUpstream(transport=httpx.MockTransport(handler))
    events = [
        event
        async for event in upstream.stream(
            access_token="secret-token-abc",
            account_id="acct-9",
            payload={"model": "gpt-5-codex", "input": [], "stream": False},
        )
    ]

    assert captured["url"] == f"{CODEX_OFFICIAL_BASE_URL}/responses"
    assert captured["headers"]["authorization"] == "Bearer secret-token-abc"
    assert captured["headers"]["chatgpt-account-id"] == "acct-9"
    # Native Codex REST wire (not the WebSocket-only beta value, and not the
    # generic SDK `text/event-stream` Accept) — see `CodexHTTPUpstream.stream`.
    assert captured["headers"]["accept"] == "application/json"
    assert "openai-beta" not in captured["headers"]
    assert captured["body"]["stream"] is True
    assert events[0]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_stream_retains_honest_codex_pool_wire_identity_never_lingtai_or_official_cli():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content=_sse_body([("response.completed", {"type": "response.completed", "response": {"status": "completed", "output": []}})]),
        )

    upstream = CodexHTTPUpstream(transport=httpx.MockTransport(handler))
    _ = [event async for event in upstream.stream(access_token="t", account_id=None, payload={"model": "m", "input": []})]

    assert captured["headers"]["originator"] == "codex-pool"
    assert captured["headers"]["user-agent"].startswith("codex-pool/")
    assert "lingtai" not in captured["headers"]["user-agent"].lower()
    assert "codex_cli_rs" not in captured["headers"]["user-agent"]


@pytest.mark.asyncio
async def test_stream_never_lets_payload_choose_an_arbitrary_upstream_url():
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            content=_sse_body([("response.completed", {"type": "response.completed", "response": {"status": "completed", "output": []}})]),
        )

    upstream = CodexHTTPUpstream(transport=httpx.MockTransport(handler))
    _ = [
        event
        async for event in upstream.stream(
            access_token="t",
            account_id=None,
            payload={"model": "m", "input": [], "base_url": "https://evil.example/steal", "url": "https://evil.example/steal"},
        )
    ]
    assert seen_urls == [f"{CODEX_OFFICIAL_BASE_URL}/responses"]


@pytest.mark.asyncio
async def test_http_error_is_structured_without_leaking_bearer_or_arbitrary_error_field():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer super-secret-token"
        return httpx.Response(
            401,
            json={
                "error": {
                    "type": "invalid_auth",
                    "code": "token_expired",
                    "message": "token super-secret-token has expired",
                }
            },
        )

    upstream = CodexHTTPUpstream(transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamTransportError) as excinfo:
        async for _ in upstream.stream(access_token="super-secret-token", account_id=None, payload={"model": "m", "input": []}):
            pass

    assert excinfo.value.status_code == 502
    message = str(excinfo.value)
    assert "super-secret-token" not in message
    assert "401" in message
    assert "token_expired" in message or "invalid_auth" in message
