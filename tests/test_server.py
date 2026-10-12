"""Offline local HTTP smoke; scripted upstream is not provider acceptance."""

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from subs_pool.modules.codex.accounts import AccountStore
from subs_pool.modules.codex.chain import ChainStore
from subs_pool.modules.codex.server import create_app
from subs_pool.modules.codex.quota_store import QuotaStore, iso
from fakes import ScriptedUpstream, write_auth_fixture

API_KEY = "test-key-do-not-log"


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://local")


def _setup_one_account(tmp_path):
    auth = tmp_path / "personal.json"
    write_auth_fixture(auth)
    store = AccountStore()
    account = store.import_account("personal", str(auth))
    now = datetime.now(timezone.utc)
    quota = QuotaStore(store.root)
    quota.claim([account], attempt_id="test-fixture", owner_id="test", deadline_at=now + timedelta(seconds=20))
    assert quota.commit_success(
        account,
        attempt_id="test-fixture",
        sample={
            "source_at": iso(now), "checked_at": iso(now), "fresh_until": iso(now + timedelta(seconds=60)),
            "allowed": True, "limit_reached": False,
            "primary": {"used_percent": 10, "remaining_percent": 90, "reset_at": None, "window_seconds": None},
            "secondary": {"used_percent": None, "remaining_percent": None, "reset_at": None, "window_seconds": None},
        },
    )
    return store


@pytest.mark.asyncio
async def test_nonstream_create_returns_full_response(tmp_path):
    accounts = _setup_one_account(tmp_path)
    upstream = ScriptedUpstream()
    upstream.queue_success(
        model="gpt-5-codex",
        output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi there"}]}],
    )
    app = create_app(accounts=accounts, chain_store=ChainStore(), upstream=upstream, api_key=API_KEY)

    async with _client(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "gpt-5-codex", "input": [{"role": "user", "content": "hello"}]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["output_text"] == "hi there"
    assert upstream.calls[0]["payload"]["stream"] is True
    assert upstream.calls[0]["account_id"] == "acct-1"


@pytest.mark.asyncio
async def test_streaming_create_yields_incremental_events(tmp_path):
    accounts = _setup_one_account(tmp_path)
    upstream = ScriptedUpstream()
    upstream.queue_success(
        model="gpt-5-codex",
        output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello world"}]}],
    )
    app = create_app(accounts=accounts, chain_store=ChainStore(), upstream=upstream, api_key=API_KEY)

    async with _client(app) as client:
        async with client.stream(
            "POST",
            "/v1/responses",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "gpt-5-codex", "input": [{"role": "user", "content": "hello"}], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            raw = b""
            async for chunk in resp.aiter_bytes():
                raw += chunk

    text = raw.decode()
    event_types = [line[len("event: "):] for line in text.splitlines() if line.startswith("event: ")]
    assert "response.created" in event_types
    assert event_types.count("response.output_text.delta") > 1
    assert "response.completed" in event_types


@pytest.mark.asyncio
async def test_partial_stream_does_not_advance_baseline_or_fake_completion(tmp_path):
    accounts = _setup_one_account(tmp_path)
    upstream = ScriptedUpstream()
    upstream.queue_partial_then_fail(
        model="gpt-5-codex",
        partial_item={"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "starting..."}]},
    )
    chain_store = ChainStore()
    app = create_app(accounts=accounts, chain_store=chain_store, upstream=upstream, api_key=API_KEY)

    async with _client(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "gpt-5-codex", "input": [{"role": "user", "content": "hello"}]},
        )
    assert resp.status_code == 502
    assert "error" in resp.json()
    assert chain_store.record_count() == 0


@pytest.mark.asyncio
async def test_transport_error_before_any_output_returns_error_and_no_commit(tmp_path):
    accounts = _setup_one_account(tmp_path)
    upstream = ScriptedUpstream()
    upstream.queue_transport_error()
    chain_store = ChainStore()
    app = create_app(accounts=accounts, chain_store=chain_store, upstream=upstream, api_key=API_KEY)

    async with _client(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "gpt-5-codex", "input": [{"role": "user", "content": "hello"}]},
        )
    assert resp.status_code == 502
    assert chain_store.record_count() == 0
    assert len(upstream.calls) == 1


@pytest.mark.asyncio
async def test_wrong_or_missing_bearer_key_is_rejected_and_key_never_echoed(tmp_path):
    accounts = _setup_one_account(tmp_path)
    upstream = ScriptedUpstream()
    app = create_app(accounts=accounts, chain_store=ChainStore(), upstream=upstream, api_key=API_KEY)

    async with _client(app) as client:
        resp_missing = await client.post("/v1/responses", json={"model": "m", "input": []})
        resp_wrong = await client.post(
            "/v1/responses", headers={"Authorization": "Bearer nope"}, json={"model": "m", "input": []}
        )
    assert resp_missing.status_code == 401
    assert resp_wrong.status_code == 401
    assert API_KEY not in resp_wrong.text
    assert len(upstream.calls) == 0


@pytest.mark.asyncio
async def test_unsupported_field_rejected_loudly_not_silently_dropped(tmp_path):
    accounts = _setup_one_account(tmp_path)
    upstream = ScriptedUpstream()
    app = create_app(accounts=accounts, chain_store=ChainStore(), upstream=upstream, api_key=API_KEY)

    async with _client(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "m", "input": [], "store": True},
        )
    assert resp.status_code == 400
    assert "store" in resp.json()["error"]["message"]
    assert len(upstream.calls) == 0


@pytest.mark.asyncio
async def test_no_eligible_account_returns_503(tmp_path):
    accounts = AccountStore()
    app = create_app(accounts=accounts, chain_store=ChainStore(), upstream=ScriptedUpstream(), api_key=API_KEY)

    async with _client(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "m", "input": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 503
