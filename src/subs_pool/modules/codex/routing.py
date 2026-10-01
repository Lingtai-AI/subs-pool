"""The two scheduling rules: full-prefix affinity, else weighted load balance.

Rule A — no match: weighted pick among eligible pool accounts.
Rule B — full-prefix match + account still eligible: sticky to that account.
Rule C — opt-in (``reroute_retired=True``): a full-prefix match whose bound account
has been retired (exhausted, or at/below the exhaust threshold) is rerouted to a
weighted pick, and the move is reported in ``rerouted_from``. Off by default, so the
library keeps its promise never to move a continuation on its own; the server turns it
on because a caller's alternative there is a dead turn.

Eligibility here means "in the pool, enabled, and authenticated" (an
account with a missing/invalid auth file is explicitly unavailable — never
excluded merely because its quota is unknown, and a token refresh never
counts as an account swap since it does not change which account is bound).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path

from .accounts import Account
from .auth_codex import CodexTokenManager
from .chain import ChainStore, MatchResult
from .quota_store import QuotaStore


class NoEligibleAccountError(Exception):
    """No pool member is enabled and authenticated."""


@dataclass(frozen=True)
class RoutingDecision:
    account_ref: str
    chain_id: str
    matched: bool
    prefix_hashes: list
    rerouted_from: str | None = None   # set when a continuation was moved off a retired account


def _is_authenticated(account: Account, *, root: Path | None = None) -> bool:
    try:
        path = account.resolved_auth_path(root or Path.cwd())
        return CodexTokenManager(str(path)).is_authenticated()
    except (FileNotFoundError, OSError, ValueError):
        return False


def eligible_refs(
    accounts: list[Account],
    *,
    quota_store: QuotaStore | None = None,
    snapshot: dict | None = None,
) -> set[str]:
    """Return only accounts with a current, committed quota sample.

    A missing sidecar is intentionally not an eligible state. The optional
    store argument is useful to callers that already hold a consistent
    snapshot; production routing always supplies the shared store.
    """
    if quota_store is None:
        quota_store = QuotaStore()
    return {
        account.ref
        for account in accounts
        if account.enabled
        and _is_authenticated(account, root=quota_store.root)
        and quota_store.is_eligible(account, snapshot=snapshot)
    }


def weighted_choice(accounts: list[Account], *, randbelow=None) -> str:
    """Unbiased weighted draw among eligible accounts, via ``secrets``.

    ``randbelow`` defaults to :func:`secrets.randbelow`; tests inject a
    deterministic stand-in to make weight-proportional selection
    reproducible without weakening the real (cryptographic) draw.

    Static weights only in this pass: dynamic (quota-scaled) weighting needs
    real quota data, which this pass does not implement (see
    IMPLEMENTATION_REPORT.md).
    """
    if not accounts:
        raise NoEligibleAccountError("no enabled, authenticated pool accounts")
    randbelow = randbelow or secrets.randbelow
    total = sum(a.weight for a in accounts)
    r = randbelow(total)
    upto = 0
    for a in accounts:
        upto += a.weight
        if r < upto:
            return a.ref
    return accounts[-1].ref  # unreachable in practice; defensive fallback


def select_account(
    *,
    accounts: list[Account],
    input_items: list,
    cfg: dict,
    chain_store: ChainStore,
    randbelow=None,
    quota_store: QuotaStore | None = None,
    snapshot: dict | None = None,
    reroute_retired: bool = False,
) -> RoutingDecision:
    elig = eligible_refs(accounts, quota_store=quota_store, snapshot=snapshot)
    if not elig:
        raise NoEligibleAccountError("no current quota-eligible account")

    # Continuation affinity is held as hard as it can be: a matching chain stays on its account for as
    # long as that account is eligible, because moving it costs the upstream prompt cache for the whole
    # prefix (tens of millions of tokens in a long agent run) and that is the cache the chain exists to
    # earn. But when the bound account has been retired -- exhausted, or under the exhaust threshold --
    # refusing is worse than moving: the caller's alternative is a dead turn, and a harness that cannot
    # see why records it as the agent's failure. So the continuation is rerouted, and `rerouted_from`
    # carries which account it left, so the move is announced rather than silent (CONTRACT.md).
    bound = chain_store.find_bound(input_items, cfg)
    rerouted_from = None
    if bound.account_ref is not None and bound.account_ref not in elig:
        if not reroute_retired:
            raise NoEligibleAccountError("the bound account has no current quota")
        rerouted_from = bound.account_ref

    match: MatchResult = chain_store.find_match(input_items, cfg, elig)
    if match.account_ref is not None:
        return RoutingDecision(
            account_ref=match.account_ref,
            chain_id=match.chain_id,
            matched=True,
            prefix_hashes=match.prefix_hashes,
        )

    eligible_accounts = [a for a in accounts if a.ref in elig]
    chosen = weighted_choice(eligible_accounts, randbelow=randbelow)
    return RoutingDecision(
        account_ref=chosen,
        chain_id=(bound.chain_id if rerouted_from else match.chain_id),
        rerouted_from=rerouted_from,
        matched=False,
        prefix_hashes=match.prefix_hashes,
    )
