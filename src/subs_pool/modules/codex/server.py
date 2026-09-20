"""Local loopback Responses API server.

Exposes ``POST /v1/responses`` for an ordinary OpenAI Responses SDK client
(``client.responses.create(...)``). No caller session header, no pool
selection endpoint — a client only ever sees a normal Responses API. Routing
(the two rules) and one-latest-baseline bookkeeping happen here, driven by
observing request/response content itself. The pool alone owns the upstream
session identity: the matched (or freshly issued) chain id.
"""

from __future__ import annotations

import json
import logging
import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .accounts import AccountError, AccountStore
from .auth_codex import CodexTokenManager
from .chain import ChainStore
from .errors import PoolRequestError, UpstreamTransportError
from .routing import NoEligibleAccountError, select_account
from .quota_refresh import QuotaRefreshCoordinator, refresh_async
from .quota_store import QuotaStateError, QuotaStore, sanitized_error
from .sse import encode_event
from .upstream import Upstream

# Fields whose presence is always rejected: server-side response history /
# stateful conversation objects are not implemented at all.
_REJECTED_ALWAYS = {
    "previous_response_id": "server-side response history is not implemented; send full input",
    "conversation": "the conversation object is not implemented",
}

# `store=False` and `background=False` are ordinary values an OpenAI SDK may
# send. Truthy values would ask this stateless proxy to emulate unsupported
# persistence/background semantics, so those are rejected before upstream.
_REJECTED_UNLESS_FALSE = {
    "store": "response storage/retrieval is not implemented",
    "background": "background mode is not implemented",
}

# Inputs that can alter model behaviour or the shape/content of a completion.
# Transport-only fields (`input`, `stream`) are intentionally excluded; all
# other request fields are forwarded unchanged rather than silently discarded.
_CONFIG_FIELDS = (
    "model",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning",
    "text",
    "include",
    "service_tier",
    "temperature",
    "top_p",
    "max_output_tokens",
    "truncation",
)

# Native Codex REST parity: every real Codex turn asks for the raw encrypted
# reasoning item so replay can stay prompt-cache-stable (see
# reference/lingtai-kernel adapter.py, the `reasoning.encrypted_content`
# `include` default applied unconditionally before the request is built).
_REQUIRED_INCLUDE = "reasoning.encrypted_content"


logger = logging.getLogger("subs_pool.codex.server")


def _normalize_body(body: object) -> object:
    """Losslessly normalize ordinary SDK request shapes before validation.

    ``client.responses.create(input="...")`` is a documented SDK call shape;
    internally it is equivalent to one user message and is converted to the
    canonical item-array form before routing and forwarding.
    """
    if not isinstance(body, dict):
        return body
    normalized = dict(body)
    input_val = normalized.get("input")
    if isinstance(input_val, str):
        normalized["input"] = [{"role": "user", "content": input_val}]
    return normalized


def _validate_request(body: object) -> None:
    if not isinstance(body, dict):
        raise PoolRequestError("request body must be a JSON object")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise PoolRequestError("'model' is required and must be a non-empty string")
    if not isinstance(body.get("input"), list):
        raise PoolRequestError("'input' is required and must be a string or an array of items")
    include = body.get("include")
    if include is not None and not isinstance(include, (str, list)):
        raise PoolRequestError("'include' must be a string, an array, or null")
    for field, reason in _REJECTED_ALWAYS.items():
        if field in body:
            raise PoolRequestError(f"'{field}' is not supported by the subs-pool Codex module: {reason}")
    for field, reason in _REJECTED_UNLESS_FALSE.items():
        if field in body and body[field] is not False:
            raise PoolRequestError(f"'{field}' is not supported by the subs-pool Codex module except as false: {reason}")


def _with_encrypted_reasoning_include(body: dict) -> dict:
    """Apply the native ``reasoning.encrypted_content`` include default.

    Preserves any caller-supplied ``include`` values and their order; only
    appends the encrypted-content entry when the caller did not already ask
    for it. Must run before ``_extract_config`` so affinity/config hashing
    keys off the same effective ``include`` list this server forwards
    upstream, rather than the raw caller value.
    """
    existing = body.get("include")
    if isinstance(existing, str):
        existing = [existing]
    elif isinstance(existing, list):
        existing = list(existing)
    else:
        existing = []
    if _REQUIRED_INCLUDE not in existing:
        existing = [*existing, _REQUIRED_INCLUDE]
    return {**body, "include": existing}


def _meter(outcome: "_RunOutcome", decision) -> None:
    """One JSON line per upstream response, to the file named by SUBS_POOL_USAGE_LOG, if any.

    Counts only: tokens in, tokens served from the upstream prompt cache, tokens out, which chain and
    account, and whether the chain was matched or fresh. No conversation content, no credential, no
    request body. It exists because the relay's one job -- keeping a conversation on the account that
    holds its prompt cache -- was failing for weeks with nothing to show it: 54 per cent of input
    cached through the relay against 98 direct, found only by reading finished traces. A relay that
    cannot report its own cache hit rate cannot be trusted to have one. Never raises.
    """
    path = os.environ.get("SUBS_POOL_USAGE_LOG")
    if not path:
        return
    try:
        usage = ((outcome.final_response or {}).get("usage") or {}) if outcome else {}
        details = usage.get("input_tokens_details") or {}
        line = {"t": round(time.time(), 3), "chain": str(decision.chain_id)[-8:],
                "account": decision.account_ref, "matched": bool(decision.matched),
                "rerouted_from": getattr(decision, "rerouted_from", None),
                "input": usage.get("input_tokens"), "cached": details.get("cached_tokens"),
                "output": usage.get("output_tokens"),
                "status": (outcome.final_response or {}).get("status") if outcome else None,
                "error": bool(outcome and outcome.error is not None)}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    except Exception:                                   # a meter must never cost a turn
        return


def _extract_config(body: dict) -> dict:
    return {k: body.get(k) for k in _CONFIG_FIELDS}


def _forward_payload(body: dict) -> dict:
    rejectable = {**_REJECTED_ALWAYS, **_REJECTED_UNLESS_FALSE}
    payload = {k: v for k, v in body.items() if k not in rejectable}
    # The server always talks to Upstream in streaming mode so it can relay
    # real incremental events to streaming clients; a non-streaming client's
    # response is an aggregate of that same stream. The client's `stream`
    # value only decides our response shape, not this upstream setting.
    payload["stream"] = True
    # Codex explicitly requires store=false on every request. This is an
    # owning-layer translation, not a silent acceptance of store=true (which
    # validation rejects above).
    payload["store"] = False
    return payload


class _RunOutcome:
    __slots__ = (
        "output_items",
        "final_response",
        "error",
        "output_observed",
        "_items_by_index",
        "_item_order",
    )

    def __init__(self) -> None:
        self.output_items: list = []
        self.final_response: dict | None = None
        self.error: UpstreamTransportError | None = None
        # True once actual output was observed either in a terminal response
        # list or a response.output_item.done event. A completed response that
        # has neither is returned honestly but cannot advance affinity.
        self.output_observed = False
        self._items_by_index: dict[int, dict] = {}
        self._item_order: list[int] = []

    def _record_done_item(self, output_index: object, item: dict) -> None:
        idx = output_index if isinstance(output_index, int) and not isinstance(output_index, bool) else len(self._item_order)
        if idx not in self._items_by_index:
            self._item_order.append(idx)
        self._items_by_index[idx] = item
        self.output_observed = True

    def _accumulated_output(self) -> list:
        return [self._items_by_index[i] for i in sorted(self._item_order)]


def _terminal_event(event: dict, outcome: _RunOutcome) -> dict:
    """Record a terminal response and return the client-visible normalized event."""
    response = event.get("response")
    if not isinstance(response, dict):
        return event

    items = response.get("output")
    accumulated = outcome._accumulated_output()
    # A non-empty terminal list is authoritative. An empty trailer is common
    # when the real items arrived as response.output_item.done events, so never
    # let that empty trailer discard observed output. An absent/non-list output
    # likewise falls back only to genuine done events. An explicit empty list
    # with no done items is still unobservable and must not create an
    # input-only affinity baseline.
    if isinstance(items, list) and items:
        output_items = items
        outcome.output_observed = True
    elif accumulated:
        output_items = accumulated
    else:
        output_items = items if isinstance(items, list) else []
    outcome.output_items = output_items
    outcome.final_response = {**response, "output": output_items}
    return {**event, "response": outcome.final_response}


async def _drive_upstream(
    upstream: Upstream,
    *,
    access_token: str,
    account_id: str | None,
    payload: dict,
    session_id: str | None = None,
    thread_id: str | None = None,
) -> AsyncIterator[tuple[dict, _RunOutcome]]:
    """Yield ``(event, outcome)`` pairs and accumulate terminal output state."""
    outcome = _RunOutcome()
    try:
        async for event in upstream.stream(
            access_token=access_token,
            account_id=account_id,
            payload=payload,
            session_id=session_id,
            thread_id=thread_id,
        ):
            if not isinstance(event, dict):
                raise UpstreamTransportError("upstream returned malformed event data")
            etype = event.get("type")
            if etype == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    outcome._record_done_item(event.get("output_index"), item)
                visible_event = event
            elif etype in ("response.completed", "response.failed", "response.incomplete"):
                visible_event = _terminal_event(event, outcome)
            else:
                visible_event = event
            yield visible_event, outcome
    except UpstreamTransportError as exc:
        outcome.error = exc
        yield {"type": "error", "error": exc.to_body()["error"]}, outcome
        return

    # A clean EOF without a terminal response is still a failed/incomplete
    # turn. Surface it as a typed error rather than allowing a streaming caller
    # to mistake an abruptly ended SSE stream for success.
    if outcome.final_response is None and outcome.error is None:
        outcome.error = UpstreamTransportError("upstream stream ended without a terminal event")
        yield {"type": "error", "error": outcome.error.to_body()["error"]}, outcome


def _auth_failure_response() -> JSONResponse:
    # Do not echo file paths, token errors, or provider response details.
    return JSONResponse(
        {
            "error": {
                "message": "account authentication is unavailable",
                "type": "account_auth_error",
                "code": "account_auth_error",
            }
        },
        status_code=502,
    )


def create_app(
    *,
    accounts: AccountStore,
    chain_store: ChainStore,
    upstream: Upstream,
    api_key: str,
    quota_coordinator: QuotaRefreshCoordinator | None = None,
) -> Starlette:
    quota_store = quota_coordinator.store if quota_coordinator is not None else QuotaStore(accounts.root)
    coordinator = quota_coordinator or QuotaRefreshCoordinator(accounts, quota_store=quota_store)

    async def periodic_quota_refresh(stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    await refresh_async(coordinator, force=False, timeout_seconds=22.0)
                except Exception:  # noqa: BLE001 - periodic work must not kill the server
                    continue

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        stop = asyncio.Event()
        task: asyncio.Task | None = None
        try:
            # Startup refresh is bounded and best-effort: an unavailable pool
            # still starts so clients receive a clear local 503 instead of an
            # absent service. No request is opened upstream during refresh.
            try:
                await refresh_async(coordinator, force=True, timeout_seconds=22.0)
            except Exception:
                pass
            task = asyncio.create_task(periodic_quota_refresh(stop))
            yield
        finally:
            stop.set()
            if task is not None:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

    async def responses_endpoint(request: Request) -> JSONResponse | StreamingResponse:
        auth_header = request.headers.get("authorization", "")
        provided = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
        if not provided or not hmac.compare_digest(provided, api_key):
            return JSONResponse(
                {
                    "error": {
                        "message": "invalid api key",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
                status_code=401,
            )

        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(PoolRequestError("request body must be valid JSON").to_body(), status_code=400)

        body = _normalize_body(body)
        try:
            _validate_request(body)
        except PoolRequestError as exc:
            return JSONResponse(exc.to_body(), status_code=exc.status_code)

        # Validation established the dict/list shapes above.
        assert isinstance(body, dict)
        body = _with_encrypted_reasoning_include(body)
        cfg = _extract_config(body)
        input_items = body["input"]
        try:
            pool_accounts, quota_snapshot = quota_store.read_consistent(accounts)
        except (AccountError, QuotaStateError, OSError):
            return JSONResponse(
                {
                    "error": {
                        "message": "quota state is unavailable",
                        "type": "quota_unavailable",
                        "code": "quota_unavailable",
                    }
                },
                status_code=503,
            )

        # Resolve affinity independently of eligibility. A bound continuation
        # may be refreshed in place, but it is never rewritten onto B.
        bound = chain_store.find_bound(input_items, cfg)
        bound_account = next((a for a in pool_accounts if a.ref == bound.account_ref), None)
        bound_unavailable = bound.account_ref is not None and (
            bound_account is None or not quota_store.is_eligible(bound_account, snapshot=quota_snapshot)
        )
        no_current_account = not any(quota_store.is_eligible(account, snapshot=quota_snapshot) for account in pool_accounts)
        # JIT refresh joins an owner and never retries generation upstream.
        if bound_unavailable or no_current_account:
            outcome = await refresh_async(coordinator, force=True, timeout_seconds=22.0)
            if outcome.snapshot is None:
                return JSONResponse(
                    {
                        "error": {
                            "message": "no current quota-eligible account is available",
                            "type": "quota_unavailable",
                            "code": "quota_unavailable",
                        }
                    },
                    status_code=503,
                )
            try:
                pool_accounts, quota_snapshot = quota_store.read_consistent(accounts)
            except (AccountError, QuotaStateError, OSError):
                return JSONResponse(
                    {
                        "error": {
                            "message": "quota state is unavailable",
                            "type": "quota_unavailable",
                            "code": "quota_unavailable",
                        }
                    },
                    status_code=503,
                )

        if bound.account_ref is not None:
            bound_account = next((a for a in pool_accounts if a.ref == bound.account_ref), None)
            if bound_account is None or not quota_store.is_eligible(bound_account, snapshot=quota_snapshot):
                return JSONResponse(
                    {"error": {"message": "bound account quota is unavailable", "type": "quota_unavailable", "code": "quota_unavailable"}},
                    status_code=503,
                )
        try:
            decision = select_account(
                accounts=pool_accounts,
                input_items=input_items,
                cfg=cfg,
                chain_store=chain_store,
                quota_store=quota_store,
                snapshot=quota_snapshot,
                # The server opts in where the library will not decide for itself: if a continuation's
                # account has been retired (exhausted, or at/below the exhaust threshold), the caller's
                # alternative is a dead turn, and a harness that cannot see why will record that as the
                # agent's failure. Moving it costs the upstream prompt cache for this prefix, which is
                # why the move is logged rather than silent.
                reroute_retired=True,
            )
        except NoEligibleAccountError as exc:
            return JSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "quota_unavailable",
                        "code": "quota_unavailable",
                    }
                },
                status_code=503,
            )

        account = accounts.get(decision.account_ref)
        token_manager = CodexTokenManager(str(account.resolved_auth_path(accounts.root)))
        try:
            # Token read/refresh may perform blocking file and HTTP I/O. Keep
            # it off the event loop so concurrent requests can stream together.
            access_token = await run_in_threadpool(token_manager.get_access_token)
        except Exception:  # noqa: BLE001 - convert all auth failures to safe typed output
            return _auth_failure_response()
        try:
            account_id = await run_in_threadpool(token_manager.get_account_id)
        except Exception:  # noqa: BLE001 - account id is optional; never expose read details
            account_id = None

        payload = _forward_payload(body)
        # Native Codex parity: one stable per-conversation identity, sent
        # byte-identically as body `prompt_cache_key` and the `session_id` /
        # `thread_id` upstream headers. The pool is the sole owner: it is
        # always the routed chain id (reused on continuation, fresh on
        # no-match). A caller's body `prompt_cache_key` is replaced and caller
        # `session_id` / `thread_id` headers are never read.
        if decision.rerouted_from:
            logger.warning(
                "chain %s rerouted off retired account %s onto %s: the prefix's prompt cache is lost "
                "on the new account", decision.chain_id, decision.rerouted_from, decision.account_ref)
        identity = decision.chain_id
        payload["prompt_cache_key"] = identity
        want_stream = bool(body.get("stream", False))

        def commit_on_success(outcome: _RunOutcome) -> None:
            _meter(outcome, decision)
            if outcome.error is not None or outcome.final_response is None:
                return
            if outcome.final_response.get("status") != "completed":
                return
            if not outcome.output_observed:
                return  # no fake input-only baseline for unobservable output
            chain_store.commit(
                chain_id=decision.chain_id,
                prefix_hashes=decision.prefix_hashes,
                input_length=len(input_items),
                output_items=outcome.output_items,
                cfg=cfg,
                account_ref=decision.account_ref,
                fresh=not decision.matched,
            )

        if want_stream:
            # Pre-fetch exactly one event before committing to a 200 response.
            # A failure before any bytes gets a proper HTTP error; later errors
            # remain stream-level events because headers are already committed.
            driver = _drive_upstream(
                upstream,
                access_token=access_token,
                account_id=account_id,
                payload=payload,
                session_id=identity,
                thread_id=identity,
            )
            try:
                first_event, first_outcome = await driver.__anext__()
            except StopAsyncIteration:
                return JSONResponse(
                    {
                        "error": {
                            "message": "upstream stream ended without any events",
                            "type": "upstream_error",
                            "code": "upstream_error",
                        }
                    },
                    status_code=502,
                )
            if first_outcome.error is not None:
                return JSONResponse(first_outcome.error.to_body(), status_code=first_outcome.error.status_code)

            async def event_stream() -> AsyncIterator[bytes]:
                outcome = first_outcome
                yield encode_event(first_event.get("type", "message"), first_event)
                async for event, outcome in driver:
                    if await request.is_disconnected():
                        return  # cancellation: no fake completion, no commit
                    yield encode_event(event.get("type", "message"), event)
                commit_on_success(outcome)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        outcome = _RunOutcome()
        async for _event, outcome in _drive_upstream(
            upstream,
            access_token=access_token,
            account_id=account_id,
            payload=payload,
            session_id=identity,
            thread_id=identity,
        ):
            pass

        if outcome.error is not None:
            return JSONResponse(outcome.error.to_body(), status_code=outcome.error.status_code)
        if outcome.final_response is None:
            return JSONResponse(
                {
                    "error": {
                        "message": "upstream stream ended without a terminal event",
                        "type": "upstream_error",
                        "code": "upstream_error",
                    }
                },
                status_code=502,
            )

        commit_on_success(outcome)
        return JSONResponse(outcome.final_response, status_code=200)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return Starlette(
        routes=[
            Route("/v1/responses", responses_endpoint, methods=["POST"]),
            Route("/health", health),
        ],
        lifespan=lifespan,
    )
