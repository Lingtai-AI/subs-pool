from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from subs_pool.modules.codex.accounts import Account
from subs_pool.modules.codex.accounts import AccountStore
from subs_pool.modules.codex.chain import ChainStore
from subs_pool.modules.codex.quota_store import QuotaStore, iso
from subs_pool.modules.codex.routing import NoEligibleAccountError, select_account, weighted_choice
from fakes import write_auth_fixture

CFG = {"model": "gpt-5-codex", "instructions": None, "tools": None}


def _account(ref, tmp_path, *, enabled=True, weight=1, authenticated=True):
    auth = tmp_path / f"{ref}.json"
    if authenticated:
        write_auth_fixture(auth)
    else:
        auth.write_text("{}")
    return Account(ref=ref, auth_path=str(auth), enabled=enabled, weight=weight)


def _seed(accounts, root: Path, *, used: float = 10.0) -> QuotaStore:
    config = AccountStore(root / "pool.json")
    for account in accounts:
        stored = config.import_account(account.ref, account.auth_path, weight=account.weight)
        if stored.enabled != account.enabled:
            stored = config.set_enabled(account.ref, account.enabled)
        account.quota_epoch = stored.quota_epoch
    now = datetime.now(timezone.utc)
    store = QuotaStore(root)
    eligible = [account for account in accounts if account.enabled and account.local_authenticated(root=root)]
    store.claim(eligible, attempt_id="fixture", owner_id="test", deadline_at=now + timedelta(seconds=20))
    for account in eligible:
        store.commit_success(
            account,
            attempt_id="fixture",
            sample={
                "source_at": iso(now), "checked_at": iso(now), "fresh_until": iso(now + timedelta(seconds=60)),
                "allowed": True, "limit_reached": False,
                "primary": {"used_percent": used, "remaining_percent": 100 - used, "reset_at": None, "window_seconds": None},
                "secondary": {"used_percent": None, "remaining_percent": None, "reset_at": None, "window_seconds": None},
            },
        )
    return store


def test_weighted_choice_is_proportional_and_reproducible(tmp_path):
    a = _account("a", tmp_path, weight=1)
    b = _account("b", tmp_path, weight=3)
    counter = iter([0, 1, 2, 3, 0, 1, 2, 3])
    picks = [weighted_choice([a, b], randbelow=lambda n: next(counter) % n) for _ in range(8)]
    assert picks.count("a") == 2
    assert picks.count("b") == 6


def test_weighted_choice_raises_when_no_accounts():
    with pytest.raises(NoEligibleAccountError):
        weighted_choice([])


def test_routing_requires_fresh_sidecar_and_skips_disabled(tmp_path):
    a = _account("a", tmp_path)
    b = _account("b", tmp_path, enabled=False, weight=2)
    quota = _seed([a, b], tmp_path)
    decision = select_account(
        accounts=[a, b], input_items=[{"role": "user", "content": "hi"}], cfg=CFG,
        chain_store=ChainStore(), quota_store=quota, snapshot=quota.read(), randbelow=lambda n: 0,
    )
    assert decision.account_ref == "a"


def test_unknown_quota_is_fail_closed(tmp_path):
    a = _account("a", tmp_path)
    with pytest.raises(NoEligibleAccountError):
        select_account(accounts=[a], input_items=[{"role": "user", "content": "hi"}], cfg=CFG, chain_store=ChainStore(), quota_store=QuotaStore(tmp_path))


def test_malformed_auth_account_is_skipped_and_valid_account_selected(tmp_path):
    bad_auth = tmp_path / "bad.json"
    bad_auth.write_text("[]", encoding="utf-8")
    bad = Account(ref="bad", auth_path=str(bad_auth), weight=100)
    good = _account("good", tmp_path)
    quota = _seed([bad, good], tmp_path)
    decision = select_account(
        accounts=[bad, good], input_items=[{"role": "user", "content": "hi"}], cfg=CFG,
        chain_store=ChainStore(), quota_store=quota, snapshot=quota.read(), randbelow=lambda n: n - 1,
    )
    assert decision.account_ref == "good"


def test_full_prefix_match_sticks_to_same_current_account(tmp_path):
    a = _account("a", tmp_path, weight=1)
    b = _account("b", tmp_path, weight=100)
    quota = _seed([a, b], tmp_path)
    store = ChainStore()
    u1 = [{"role": "user", "content": "hi"}]
    first = select_account(accounts=[a, b], input_items=u1, cfg=CFG, chain_store=store, quota_store=quota, snapshot=quota.read(), randbelow=lambda n: n - 1)
    output = [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]
    store.commit(chain_id=first.chain_id, prefix_hashes=first.prefix_hashes, input_length=len(u1), output_items=output, cfg=CFG, account_ref=first.account_ref)
    second = select_account(accounts=[a, b], input_items=u1 + output + [{"role": "user", "content": "again"}], cfg=CFG, chain_store=store, quota_store=quota, snapshot=quota.read(), randbelow=lambda n: n - 1)
    assert second.matched is True
    assert second.account_ref == first.account_ref


def test_bound_account_stale_or_exhausted_does_not_fail_over(tmp_path):
    a = _account("a", tmp_path)
    b = _account("b", tmp_path)
    quota = _seed([a, b], tmp_path)
    store = ChainStore()
    first_input = [{"role": "user", "content": "hi"}]
    first = select_account(accounts=[a, b], input_items=first_input, cfg=CFG, chain_store=store, quota_store=quota, snapshot=quota.read(), randbelow=lambda n: 0)
    output = [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]
    store.commit(chain_id=first.chain_id, prefix_hashes=first.prefix_hashes, input_length=1, output_items=output, cfg=CFG, account_ref=first.account_ref)
    now = datetime.now(timezone.utc)
    quota.claim([a], attempt_id="exhaust", owner_id="test", deadline_at=now + timedelta(seconds=20))
    quota.commit_success(a, attempt_id="exhaust", sample={
        "source_at": iso(now), "checked_at": iso(now), "fresh_until": iso(now + timedelta(seconds=60)),
        "allowed": False, "limit_reached": True,
        "primary": {"used_percent": 100, "remaining_percent": 0, "reset_at": None, "window_seconds": None},
        "secondary": {"used_percent": None, "remaining_percent": None, "reset_at": None, "window_seconds": None},
    })
    with pytest.raises(NoEligibleAccountError):
        select_account(accounts=[a, b], input_items=first_input + output + [{"role": "user", "content": "again"}], cfg=CFG, chain_store=store, quota_store=quota, snapshot=quota.read())


def _retire(quota, account, *, remaining: float):
    """Commit a fresh sample that leaves `account` with this much of its window."""
    now = datetime.now(timezone.utc)
    quota.claim([account], attempt_id="retire", owner_id="test", deadline_at=now + timedelta(seconds=20))
    quota.commit_success(account, attempt_id="retire", sample={
        "source_at": iso(now), "checked_at": iso(now), "fresh_until": iso(now + timedelta(seconds=60)),
        "allowed": True, "limit_reached": False,
        "primary": {"used_percent": 100 - remaining, "remaining_percent": remaining,
                    "reset_at": None, "window_seconds": None},
        "secondary": {"used_percent": None, "remaining_percent": None, "reset_at": None, "window_seconds": None},
    })


def test_account_under_the_exhaust_threshold_is_retired_before_it_fails(tmp_path):
    """A window read as 0.4 per cent is too thin to finish a turn, so it leaves the routing pool
    while it can still be left. Zero is not the useful boundary."""
    a = _account("a", tmp_path)
    quota = _seed([a], tmp_path)
    assert quota.is_eligible(a) is True
    _retire(quota, a, remaining=0.4)
    assert quota.exhaust_threshold == 0.5
    assert quota.is_eligible(a) is False
    view = quota.view_for(a, quota.read())
    assert view["exclusion_reason"] == "exhausted"


def test_threshold_is_overridable_by_environment(tmp_path, monkeypatch):
    a = _account("a", tmp_path)
    quota = _seed([a], tmp_path)
    _retire(quota, a, remaining=3.0)
    assert quota.is_eligible(a) is True                      # 3% clears the 0.5% default
    monkeypatch.setenv("SUBS_POOL_EXHAUST_THRESHOLD", "5")
    assert quota.exhaust_threshold == 5.0
    assert quota.is_eligible(a) is False


def test_retired_bound_account_reroutes_only_when_asked_and_says_so(tmp_path):
    """The library still refuses to move a continuation on its own (the contract test above). With
    reroute_retired the move happens and names the account it left, so it is never silent."""
    a, b = _account("a", tmp_path), _account("b", tmp_path)
    quota = _seed([a, b], tmp_path)
    store = ChainStore()
    first_input = [{"role": "user", "content": "hi"}]
    first = select_account(accounts=[a, b], input_items=first_input, cfg=CFG, chain_store=store,
                          quota_store=quota, snapshot=quota.read(), randbelow=lambda n: 0)
    output = [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]
    store.commit(chain_id=first.chain_id, prefix_hashes=first.prefix_hashes, input_length=1,
                 output_items=output, cfg=CFG, account_ref=first.account_ref)
    _retire(quota, a if first.account_ref == "a" else b, remaining=0.2)
    cont = first_input + output + [{"role": "user", "content": "again"}]

    with pytest.raises(NoEligibleAccountError):               # default: the promise holds
        select_account(accounts=[a, b], input_items=cont, cfg=CFG, chain_store=store,
                       quota_store=quota, snapshot=quota.read())

    moved = select_account(accounts=[a, b], input_items=cont, cfg=CFG, chain_store=store,
                           quota_store=quota, snapshot=quota.read(), reroute_retired=True)
    assert moved.rerouted_from == first.account_ref
    assert moved.account_ref != first.account_ref
    assert moved.chain_id == first.chain_id                   # the same conversation, a new account


def test_chain_cache_survives_a_restart_and_is_safe_to_delete(tmp_path):
    """The cache holds hashes and an account ref, never conversation content, and a lost file is a
    cold start rather than an error."""
    path = tmp_path / "chain-cache-v1.json"
    a, b = _account("a", tmp_path), _account("b", tmp_path)
    quota = _seed([a, b], tmp_path)
    store = ChainStore(path=path)
    items = [{"role": "user", "content": "hi"}]
    first = select_account(accounts=[a, b], input_items=items, cfg=CFG, chain_store=store,
                           quota_store=quota, snapshot=quota.read(), randbelow=lambda n: 0)
    output = [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]
    store.commit(chain_id=first.chain_id, prefix_hashes=first.prefix_hashes, input_length=1,
                 output_items=output, cfg=CFG, account_ref=first.account_ref)
    store.flush()
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "hi" not in body and "output_text" not in body    # no conversation content on disk

    revived = ChainStore(path=path)                           # a restart
    cont = items + output + [{"role": "user", "content": "again"}]
    again = select_account(accounts=[a, b], input_items=cont, cfg=CFG, chain_store=revived,
                           quota_store=quota, snapshot=quota.read(), randbelow=lambda n: 1)
    assert again.matched is True
    assert again.account_ref == first.account_ref             # affinity, and its prompt cache, kept

    path.unlink()                                            # deleting the cache is allowed
    cold = ChainStore(path=path)
    assert cold.find_bound(cont, CFG).account_ref is None


def test_chain_cache_drops_expired_records(tmp_path):
    path = tmp_path / "chain-cache-v1.json"
    a = _account("a", tmp_path)
    quota = _seed([a], tmp_path)
    clock = [1000.0]
    store = ChainStore(path=path, ttl_seconds=60.0, now=lambda: clock[0], min_write_interval=0.0)
    items = [{"role": "user", "content": "hi"}]
    d = select_account(accounts=[a], input_items=items, cfg=CFG, chain_store=store,
                       quota_store=quota, snapshot=quota.read())
    store.commit(chain_id=d.chain_id, prefix_hashes=d.prefix_hashes, input_length=1,
                 output_items=[{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
                 cfg=CFG, account_ref=d.account_ref)
    store.flush()
    clock[0] += 3600.0                                       # an hour later, well past the TTL
    assert ChainStore(path=path, ttl_seconds=60.0, now=lambda: clock[0])._records == {}


def test_agent_turn_keeps_its_chain_when_the_echo_differs_from_the_response(tmp_path):
    """An agent's turn ends in a tool call, and the client echoes it back in its own shape.

    Codex replays a ``function_call`` without the ``status`` the response carried, so the hash of
    (input + our output) never matches the next request. Before input-prefix affinity that made every
    tool-call turn a fresh chain: a new prompt_cache_key and a new weighted draw, 172 times in 462
    turns on a real run, and the upstream prompt cache went from 98 per cent to 54. The input the
    client sent last turn is a byte-identical prefix of this turn's, and that is what must match.
    """
    from subs_pool.modules.codex.chain import ChainStore
    store = ChainStore(max_records=16)
    cfg = {"model": "m"}
    user = {"type": "message", "role": "user", "content": "go"}
    call_as_returned = {"type": "function_call", "call_id": "c1", "name": "sh", "arguments": "{}",
                        "status": "completed"}
    call_as_echoed = {k: v for k, v in call_as_returned.items() if k != "status"}
    result = {"type": "function_call_output", "call_id": "c1", "output": "ok"}

    first = store.find_match([user], cfg, {"a", "b"})
    assert first.account_ref is None
    store.commit(chain_id=first.chain_id, prefix_hashes=first.prefix_hashes, input_length=1,
                 output_items=[call_as_returned], cfg=cfg, account_ref="a", fresh=True)

    nxt = store.find_match([user, call_as_echoed, result], cfg, {"a", "b"})
    assert nxt.account_ref == "a" and nxt.chain_id == first.chain_id

    # a different conversation must still be a fresh chain: the prefix is what is matched, not luck
    other = store.find_match([{"type": "message", "role": "user", "content": "else"}], cfg, {"a", "b"})
    assert other.account_ref is None and other.chain_id != first.chain_id
