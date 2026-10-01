"""Bounded in-memory content-chain affinity table.

At most one current record per inferred conversation chain: digest, length,
model-effective config digest, and the account that produced it. A request
whose input full-prefix-matches a stored record's content gets routed back
to that record's account (affinity); a successful completion then replaces
that one record with the new baseline. There is no branch/lineage concept:
if two requests race to extend the same record, whichever commits last
simply becomes the new current record — the other's output is not
preserved or merged. A request built on older, already-superseded history
just misses and falls back to ordinary weighted load balancing. Failed,
partial, or cancelled turns never call ``commit()``, so they never replace
the current record.

Each chain's ``chain_id`` is the pool-owned session identity: generated once
per new inferred chain from time plus CSPRNG bits (never from content), then
carried unchanged by every sticky continuation. The server sends it upstream
as the conversation's cache/session label.
"""

from __future__ import annotations

import secrets
import json
import os
import threading
from pathlib import Path
import time
import uuid
from dataclasses import dataclass

from .hashing import config_hash as compute_config_hash
from .hashing import extend_hash, rolling_hashes

# Default number of most-recently-committed session records retained;
# `subs-pool codex serve` overrides it from CODEX_POOL_MAX_SESSIONS at startup.
DEFAULT_MAX_SESSIONS = 100000


def new_session_id() -> str:
    """Return a fresh session label independent of conversation content.

    UUIDv7 layout: 48-bit Unix milliseconds from ``time.time_ns()`` plus 74
    bits from ``secrets``, in the same 36-character UUID text shape the
    previous random chain ids used. Uniqueness is probabilistic (random bits),
    not guaranteed by the clock; ``ChainStore`` additionally refuses to reuse
    an id it currently retains.
    """
    ms = time.time_ns() // 1_000_000
    rand = secrets.randbits(74)
    value = (
        (ms & 0xFFFF_FFFF_FFFF) << 80
        | 0x7 << 76  # version 7
        | (rand >> 62) << 64
        | 0b10 << 62  # RFC 4122 variant
        | (rand & ((1 << 62) - 1))
    )
    return str(uuid.UUID(int=value))


@dataclass(frozen=True)
class Baseline:
    chain_id: str
    content_hash: str
    length: int
    config_hash: str
    account_ref: str
    last_seen: float = 0.0
    # The hash and length of the request's INPUT alone, before the model's output was folded in.
    # content_hash needs the client to echo our output items back hash-identically, and an agent does
    # not: Codex replays a ``function_call`` without the ``status`` the response carried, and replays
    # ``reasoning`` in its own shape. Only a plain assistant message is canonicalised (hashing.py), so
    # on agentic traffic -- where nearly every turn ends in a tool call -- content_hash almost never
    # matches, every turn becomes a fresh chain with a fresh prompt_cache_key and a fresh weighted
    # draw, and the upstream prompt cache is lost. Measured on gpt-6-astra ultra, 2026-09-20: 98 per
    # cent of input tokens cached direct, 54 per cent through this relay, about a hundred times the
    # uncached input for the same task. The input the client sent last turn IS a byte-identical prefix
    # of what it sends this turn, because the client built both; that needs no canonicalisation.
    input_hash: str = ""
    input_length: int = 0


@dataclass(frozen=True)
class MatchResult:
    """A full-prefix affinity match, or the absence of one (a fresh chain)."""

    chain_id: str
    account_ref: str | None  # None when no match — caller load-balances
    prefix_hashes: list[str]  # rolling hashes over the candidate input items


class ChainStore:
    """Thread-safe; a single table (max_records bounded), optionally cached on disk.

    With ``path`` set the table is written after each commit and read back at startup. That file is a
    **cache and never an authority**: it is safe to delete, it is rebuilt by ordinary traffic, a missing
    or malformed one is a cold start rather than an error, and nothing reads it to decide whether an
    account may be used (``quota-v1.json`` remains the sole persisted quota authority).

    It holds no conversation content -- a record is a chain id, two hashes, a length, an account ref and
    a timestamp -- so persisting it does not make the server stateful in the protocol sense: the caller
    still carries the whole conversation on every request and the upstream is still called with
    ``store=false``. What it preserves across a restart is *affinity*, and with it prompt-cache
    locality: one long agent run can read tens of millions of cached input tokens, and a lost binding
    re-draws the account and pays for that prefix again.
    """

    def __init__(self, max_records: int = DEFAULT_MAX_SESSIONS, *, path: "str | Path | None" = None,
                 ttl_seconds: float = 86400.0, now=time.time, min_write_interval: float = 2.0) -> None:
        self._records: dict[str, Baseline] = {}
        self._lock = threading.Lock()
        self._max_records = max_records
        self._path = Path(path) if path else None
        self._ttl = float(ttl_seconds)
        self._now = now
        self._min_write_interval = float(min_write_interval)
        self._last_write = 0.0
        self._dirty = False
        if self._path is not None:
            self._load()

    # ---------------------------------------------------------------- the cache on disk
    def _load(self) -> None:
        """Read the cache, dropping anything expired or malformed. Never raises on a bad file."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict) or raw.get("kind") != "chain-cache" or raw.get("version") != 1:
            return
        cutoff = self._now() - self._ttl
        for rec in raw.get("records") or []:
            try:
                b = Baseline(chain_id=str(rec["chain_id"]), content_hash=str(rec["content_hash"]),
                             length=int(rec["length"]), config_hash=str(rec["config_hash"]),
                             account_ref=str(rec["account_ref"]), last_seen=float(rec.get("last_seen", 0.0)),
                             # absent in a file written before input-prefix affinity: such a record
                             # simply falls back to the full-content match, as it always did
                             input_hash=str(rec.get("input_hash", "")),
                             input_length=int(rec.get("input_length", 0)))
            except (KeyError, TypeError, ValueError):
                continue                                   # one bad record never costs the rest
            if b.last_seen >= cutoff:
                self._records[b.chain_id] = b
        self._evict_if_needed()

    def _persist_locked(self, *, force: bool = False) -> None:
        """Write the table atomically, at most once every ``min_write_interval`` unless forced."""
        if self._path is None:
            return
        self._dirty = True
        now = self._now()
        if not force and now - self._last_write < self._min_write_interval:
            return
        cutoff = now - self._ttl
        payload = {"kind": "chain-cache", "version": 1, "written_at": now,
                   "records": [{"chain_id": b.chain_id, "content_hash": b.content_hash,
                                "length": b.length, "config_hash": b.config_hash,
                                "account_ref": b.account_ref, "last_seen": b.last_seen,
                                "input_hash": b.input_hash, "input_length": b.input_length}
                               for b in self._records.values() if b.last_seen >= cutoff]}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._path)
            os.chmod(self._path, 0o600)
        except OSError:
            return                                         # a cache that cannot be written is still a cache
        self._last_write = now
        self._dirty = False

    def flush(self) -> None:
        """Write the cache now, whatever the debounce says (shutdown, or a test)."""
        with self._lock:
            self._persist_locked(force=True)

    @staticmethod
    def _matched_length(rec: "Baseline", prefix_hashes: list[str], n: int) -> int:
        """How much of the candidate input this record accounts for, or 0 if it is not a prefix.

        The full record (input + our output) is the better match when the client echoes output
        hash-identically, so it is tried first; the input-only prefix is what holds for an agent."""
        if rec.length <= n and prefix_hashes[rec.length] == rec.content_hash:
            return rec.length
        # STRICTLY longer: a continuation carries last turn's input plus at least the echoed output. A
        # request that merely equals a committed input is a second conversation opening the same way
        # (or a retry), and identity is never derived from content alone -- it gets a chain of its own.
        if rec.input_hash and 0 < rec.input_length < n and prefix_hashes[rec.input_length] == rec.input_hash:
            return rec.input_length
        return 0

    def find_match(self, input_items: list, cfg: dict, eligible_refs: set[str]) -> MatchResult:
        """Find the longest full-prefix baseline match still eligible.

        Only ``input_items`` need hashing (O(n)); comparison against every
        stored record is then O(1) per record.
        """
        cfg_hash = compute_config_hash(cfg)
        prefix_hashes = rolling_hashes(input_items)
        n = len(input_items)

        with self._lock:
            candidates = [
                (self._matched_length(rec, prefix_hashes, n), rec)
                for rec in self._records.values()
                if rec.config_hash == cfg_hash and rec.account_ref in eligible_refs
            ]
            candidates = [(m, rec) for m, rec in candidates if m > 0]
            if not candidates:
                return MatchResult(chain_id=self._unused_id_locked(), account_ref=None, prefix_hashes=prefix_hashes)

        # Longest match wins; deterministic tie-break by chain_id.
        best = sorted(candidates, key=lambda mr: (-mr[0], mr[1].chain_id))[0][1]
        return MatchResult(chain_id=best.chain_id, account_ref=best.account_ref, prefix_hashes=prefix_hashes)

    def find_bound(self, input_items: list, cfg: dict) -> MatchResult:
        """Find affinity without applying live eligibility.

        Routing must learn that a continuation is bound even when its owner
        is currently disabled, stale, or unauthenticated; otherwise it could
        silently rewrite the conversation onto another account.
        """
        cfg_hash = compute_config_hash(cfg)
        prefix_hashes = rolling_hashes(input_items)
        with self._lock:
            n = len(input_items)
            candidates = [(self._matched_length(rec, prefix_hashes, n), rec)
                          for rec in self._records.values() if rec.config_hash == cfg_hash]
            candidates = [(m, rec) for m, rec in candidates if m > 0]
            if not candidates:
                return MatchResult(self._unused_id_locked(), None, prefix_hashes)
            best = sorted(candidates, key=lambda mr: (-mr[0], mr[1].chain_id))[0][1]
            return MatchResult(best.chain_id, best.account_ref, prefix_hashes)

    def commit(
        self,
        *,
        chain_id: str,
        prefix_hashes: list[str],
        input_length: int,
        output_items: list,
        cfg: dict,
        account_ref: str,
        fresh: bool = False,
    ) -> None:
        """Replace the current record for ``chain_id`` with this completion.

        No CAS, no versioning: if another commit lands on the same
        ``chain_id`` concurrently, the last write simply wins as the new
        current record. Callers must not call this for failed, partial, or
        cancelled turns.

        ``fresh=True`` marks the first commit of a new (unmatched) chain. If
        its id became retained by another chain after ``find_match`` issued
        it, a new unused id is stored instead of overwriting that session.
        """
        cfg_hash = compute_config_hash(cfg)
        full_hash = extend_hash(prefix_hashes[input_length], output_items)
        full_length = input_length + len(output_items)

        with self._lock:
            if fresh and chain_id in self._records:
                chain_id = self._unused_id_locked()
            self._records.pop(chain_id, None)  # re-insert at MRU end
            self._records[chain_id] = Baseline(
                chain_id=chain_id,
                content_hash=full_hash,
                length=full_length,
                config_hash=cfg_hash,
                account_ref=account_ref,
                last_seen=self._now(),
                input_hash=prefix_hashes[input_length],
                input_length=input_length,
            )
            self._evict_if_needed()
            self._persist_locked()

    def _unused_id_locked(self) -> str:
        # Caller holds the lock. Regenerate on the (improbable) collision
        # with a retained session so an existing record is never overwritten.
        chain_id = new_session_id()
        while chain_id in self._records:
            chain_id = new_session_id()
        return chain_id

    def _evict_if_needed(self) -> None:
        # Bounded table: drop the least-recently-committed record on overflow
        # (commit re-inserts at the end). Restart or eviction transparently
        # loses affinity per contract (not a bug).
        while len(self._records) > self._max_records:
            oldest_id = next(iter(self._records))
            del self._records[oldest_id]

    def record_count(self) -> int:
        with self._lock:
            return len(self._records)
