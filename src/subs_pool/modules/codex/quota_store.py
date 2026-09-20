"""Versioned, file-backed Codex quota authority.

This module deliberately contains no provider transport. It owns validation,
freshness, atomic persistence, and the short shared state lock. Network work
is performed by :mod:`quota_refresh` outside the lock.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock, Timeout

from .accounts import Account, AccountStore, LEGACY_QUOTA_EPOCH

SCHEMA_VERSION = 1
MODULE = "codex"
# An account is treated as exhausted at this many percent remaining, not at zero. A window read as
# 0.4 per cent is already too thin to finish a long turn, and the failure it produces is expensive and
# ugly: the request dies upstream mid-run, and a harness that cannot see the cause records it as the
# agent's failure (measured 2026-09-20: three trials filed as reward 0.0 for a missing answer file
# after a workspace ran dry). Retiring the account a little early costs a fraction of a window and
# turns a hard failure into a routing decision. Override per store, or with
# SUBS_POOL_EXHAUST_THRESHOLD.
EXHAUST_THRESHOLD_PERCENT = 0.5

MAX_AGE_SECONDS = 60.0
REFRESH_TARGET_SECONDS = 30.0
STATE_LOCK_SECONDS = 1.0
REFRESH_LOCK_TIMEOUT_SECONDS = 0.2
_STATUSES = {"never", "checking", "ok", "failed"}
_ERROR_CODES = {
    "auth_unavailable",
    "quota_http_error",
    "quota_timeout",
    "invalid_quota_response",
    "sample_expired",
    "clock_changed",
    "refresh_deadline",
    "refresh_interrupted",
    "state_unavailable",
}


class QuotaStateError(Exception):
    """The quota sidecar is unavailable, corrupt, or unsupported."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _finite_percent(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 100.0
    )


def _window(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QuotaStateError("invalid quota window")
    used = value.get("used_percent")
    remaining = value.get("remaining_percent")
    if used is not None and not _finite_percent(used):
        raise QuotaStateError("invalid quota percentage")
    if remaining is not None and not _finite_percent(remaining):
        raise QuotaStateError("invalid quota percentage")
    for key in ("reset_at",):
        if value.get(key) is not None and parse_time(value.get(key)) is None:
            raise QuotaStateError("invalid quota timestamp")
    duration = value.get("window_seconds")
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0
    ):
        raise QuotaStateError("invalid quota duration")
    calculated_remaining = (100.0 - float(used)) if used is not None else None
    if used is None and remaining is not None:
        raise QuotaStateError("unknown quota cannot have remaining percentage")
    if remaining is not None and calculated_remaining is not None and abs(float(remaining) - calculated_remaining) > 1e-6:
        raise QuotaStateError("inconsistent quota percentage")
    return {
        "used_percent": float(used) if used is not None else None,
        "remaining_percent": calculated_remaining,
        "reset_at": value.get("reset_at"),
        "window_seconds": float(duration) if duration is not None else None,
    }


def _sample(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QuotaStateError("invalid quota sample")
    source_at = parse_time(value.get("source_at"))
    checked_at = parse_time(value.get("checked_at"))
    fresh_until = parse_time(value.get("fresh_until"))
    if source_at is None or checked_at is None or fresh_until is None:
        raise QuotaStateError("invalid quota sample timestamps")
    if not source_at <= checked_at <= fresh_until:
        raise QuotaStateError("quota sample timestamps are out of order")
    if (fresh_until - source_at).total_seconds() > MAX_AGE_SECONDS + 1e-6:
        raise QuotaStateError("quota sample exceeds freshness bound")
    primary = _window(value.get("primary"))
    # A primary-only provider result is normalized to an all-unknown
    # secondary window.  An explicitly supplied null/non-object is malformed
    # and must not be silently treated as absent.
    secondary = _window({}) if "secondary" not in value else _window(value.get("secondary"))
    for window in (primary, secondary):
        reset = parse_time(window.get("reset_at"))
        if reset is not None and reset < source_at:
            raise QuotaStateError("invalid quota reset boundary")
        if reset is not None and reset <= checked_at:
            raise QuotaStateError("quota reset boundary has passed")
        if reset is not None and fresh_until > reset:
            raise QuotaStateError("quota freshness exceeds reset boundary")
    allowed = value.get("allowed", True)
    limit_reached = value.get("limit_reached", False)
    if not isinstance(allowed, bool) or not isinstance(limit_reached, bool):
        raise QuotaStateError("invalid quota sample flags")
    return {
        "source_at": iso(source_at),
        "checked_at": iso(checked_at),
        "fresh_until": iso(fresh_until),
        "allowed": allowed,
        "limit_reached": limit_reached,
        "primary": primary,
        "secondary": secondary,
    }


def _error(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise QuotaStateError("invalid quota error")
    code = value.get("code")
    message = value.get("message")
    http_status = value.get("http_status")
    retryable = value.get("retryable")
    if not isinstance(code, str) or not code or not isinstance(message, str):
        raise QuotaStateError("invalid quota error")
    if http_status is not None and (isinstance(http_status, bool) or not isinstance(http_status, int)):
        raise QuotaStateError("invalid quota error status")
    if not isinstance(retryable, bool):
        raise QuotaStateError("invalid quota error retryability")
    return {
        "code": code,
        "message": " ".join(message.split())[:256],
        "http_status": http_status,
        "retryable": retryable,
    }


def _record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QuotaStateError("invalid quota account record")
    status = value.get("status", "never")
    if status not in _STATUSES:
        raise QuotaStateError("invalid quota status")
    epoch = value.get("quota_epoch", LEGACY_QUOTA_EPOCH)
    if not isinstance(epoch, str) or not epoch:
        raise QuotaStateError("invalid quota epoch")
    result: dict[str, Any] = {
        "quota_epoch": epoch,
        "status": status,
        "attempt_id": value.get("attempt_id"),
        "attempted_at": value.get("attempted_at"),
        "checked_at": value.get("checked_at"),
        "refresh": value.get("refresh"),
        "last_success": None,
        "error": _error(value.get("error")),
    }
    for key in ("attempt_id",):
        if result[key] is not None and not isinstance(result[key], str):
            raise QuotaStateError("invalid quota attempt id")
    for key in ("attempted_at", "checked_at"):
        if result[key] is not None and parse_time(result[key]) is None:
            raise QuotaStateError("invalid quota attempt timestamp")
    refresh = result["refresh"]
    if refresh is not None:
        if not isinstance(refresh, Mapping) or not isinstance(refresh.get("owner_id"), str) or parse_time(refresh.get("deadline_at")) is None:
            raise QuotaStateError("invalid quota refresh marker")
        result["refresh"] = {
            "owner_id": refresh["owner_id"],
            "deadline_at": iso(parse_time(refresh["deadline_at"]) or utc_now()),
        }
    if value.get("last_success") is not None:
        result["last_success"] = _sample(value["last_success"])
    return result


def _empty_snapshot() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "module": MODULE, "revision": 0, "written_at": None, "accounts": {}}


def _validate_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QuotaStateError("invalid quota snapshot")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("module") != MODULE:
        raise QuotaStateError("unsupported quota snapshot")
    revision = value.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise QuotaStateError("invalid quota revision")
    written_at = value.get("written_at")
    if not isinstance(written_at, str) or parse_time(written_at) is None:
        raise QuotaStateError("invalid quota written_at")
    accounts = value.get("accounts")
    if not isinstance(accounts, Mapping):
        raise QuotaStateError("invalid quota accounts")
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "module": MODULE,
        "revision": revision,
        "written_at": iso(parse_time(written_at) or utc_now()),
        "accounts": {},
    }
    for ref, record in accounts.items():
        if not isinstance(ref, str) or not ref:
            raise QuotaStateError("invalid quota account id")
        normalized["accounts"][ref] = _record(record)
    return normalized


def sanitized_error(code: str, message: str, *, http_status: int | None = None, retryable: bool = True) -> dict[str, Any]:
    if code not in _ERROR_CODES:
        code = "state_unavailable"
    return {
        "code": code,
        "message": " ".join(str(message).split())[:256],
        "http_status": http_status if isinstance(http_status, int) else None,
        "retryable": bool(retryable),
    }


class ClockGuard:
    """Process-local wall/monotonic rollback guard used by live consumers."""

    def __init__(self, *, wall=time.time, monotonic=time.monotonic) -> None:
        self.wall = wall
        self.monotonic = monotonic
        self._last_wall: float | None = None
        self._last_mono: float | None = None
        self.invalidated = False
        self.generation = 0
        self.invalidated_generation: int | None = None

    def observe(self, wall: float | None = None) -> float:
        wall = float(self.wall() if wall is None else wall)
        mono = float(self.monotonic())
        if self._last_wall is not None and self._last_mono is not None:
            wall_delta = wall - self._last_wall
            mono_delta = mono - self._last_mono
            if (wall_delta < -1.0 and mono_delta >= -0.001) or (mono_delta < -1.0 and wall_delta >= -0.001):
                self.generation += 1
                self.invalidated = True
                self.invalidated_generation = self.generation
            elif abs(wall_delta - mono_delta) > 1.0:
                self.generation += 1
                self.invalidated = True
                self.invalidated_generation = self.generation
        self._last_wall = wall
        self._last_mono = mono
        return wall


class QuotaStore:
    """Shared quota sidecar authority for one direct Codex root."""

    def __init__(self, root: Path | None = None, *, now=utc_now, monotonic=time.monotonic) -> None:
        if root is None:
            from .home import data_home

            root = data_home()
        self.root = Path(root).expanduser().resolve()
        self.quota_path = self.root / "quota-v1.json"
        self.state_lock_path = self.root / "state.lock"
        self.refresh_lock_path = self.root / "quota-v1.refresh.lock"
        self._now = now
        self._clock_guard = ClockGuard(monotonic=monotonic)
        self._last_view_now: datetime | None = None
        self._clock_invalidated_at: datetime | None = None

    def now(self) -> datetime:
        value = self._now()
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        return datetime.fromtimestamp(float(value), timezone.utc)

    def _observe_clock(self, now: datetime) -> None:
        self._clock_guard.wall = lambda: now.timestamp()
        self._clock_guard.observe(now.timestamp())

    def begin_check(self, started: datetime | None = None) -> int:
        """Record the process-local clock generation at check start."""
        started = (started or self.now()).astimezone(timezone.utc)
        self._observe_clock(started)
        return self._clock_guard.generation

    def ensure_layout(self) -> None:
        existed = self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        if not existed and os.name != "nt":
            self.root.chmod(0o700)
        for path in (self.state_lock_path, self.refresh_lock_path):
            if not path.exists():
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    os.close(fd)
            if os.name != "nt":
                path.chmod(0o600)

    @contextmanager
    def state_lock(self, timeout: float = STATE_LOCK_SECONDS) -> Iterator[None]:
        self.ensure_layout()
        try:
            with FileLock(str(self.state_lock_path), timeout=timeout):
                yield
        except Timeout as exc:
            raise QuotaStateError("state lock unavailable") from exc

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.quota_path.exists():
            return _empty_snapshot()
        try:
            raw = json.loads(self.quota_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QuotaStateError("quota sidecar is unreadable or corrupt") from exc
        return _validate_snapshot(raw)

    def read(self) -> dict[str, Any]:
        with self.state_lock():
            return self._read_unlocked()

    def read_consistent(self, accounts: AccountStore) -> tuple[list[Account], dict[str, Any]]:
        with self.state_lock():
            return sorted(accounts._load_unlocked().values(), key=lambda a: a.ref), self._read_unlocked()

    def _write_unlocked(self, snapshot: dict[str, Any]) -> None:
        snapshot = dict(snapshot)
        snapshot["revision"] = int(snapshot.get("revision", 0)) + 1
        snapshot["written_at"] = iso(self.now())
        snapshot = _validate_snapshot(snapshot)
        self.ensure_layout()
        fd, raw_tmp = tempfile.mkstemp(prefix=".quota-v1.json.tmp.", dir=self.root)
        tmp = Path(raw_tmp)
        try:
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(snapshot, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.quota_path)
            if os.name != "nt":
                self.quota_path.chmod(0o600)
            try:
                directory_fd = os.open(self.root, os.O_RDONLY)
            except OSError:
                directory_fd = -1
            if directory_fd >= 0:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def invalidate_unlocked(self, refs: set[str]) -> None:
        """Remove records while the caller already owns ``state.lock``."""
        snapshot = self._read_unlocked()
        changed = False
        for ref in refs:
            if snapshot["accounts"].pop(ref, None) is not None:
                changed = True
        if changed:
            self._write_unlocked(snapshot)

    def invalidate(self, refs: Iterable[str]) -> None:
        with self.state_lock():
            self.invalidate_unlocked(set(refs))

    def claim(self, accounts: Iterable[Account], *, attempt_id: str, owner_id: str, deadline_at: datetime) -> list[Account]:
        selected = list(accounts)
        with self.state_lock():
            snapshot = self._read_unlocked()
            attempted_at = iso(self.now())
            for account in selected:
                snapshot["accounts"][account.ref] = {
                    "quota_epoch": account.quota_epoch,
                    "status": "checking",
                    "attempt_id": attempt_id,
                    "attempted_at": attempted_at,
                    "checked_at": snapshot["accounts"].get(account.ref, {}).get("checked_at"),
                    "refresh": {"owner_id": owner_id, "deadline_at": iso(deadline_at)},
                    "last_success": snapshot["accounts"].get(account.ref, {}).get("last_success"),
                    "error": None,
                }
            if selected:
                self._write_unlocked(snapshot)
        return selected

    def _commit_record_unlocked(self, ref: str, attempt_id: str, record: dict[str, Any]) -> bool:
        snapshot = self._read_unlocked()
        current = snapshot["accounts"].get(ref)
        if not isinstance(current, Mapping) or current.get("attempt_id") != attempt_id:
            return False
        snapshot["accounts"][ref] = record
        self._write_unlocked(snapshot)
        return True

    def commit_success(
        self,
        account: Account,
        *,
        attempt_id: str,
        sample: dict[str, Any],
        clock_generation: int | None = None,
    ) -> bool:
        with self.state_lock():
            current_accounts = AccountStore(self.root / "pool.json")._load_unlocked()
            current = current_accounts.get(account.ref)
            if current is None or current.quota_epoch != account.quota_epoch:
                return False
            snapshot = self._read_unlocked()
            old = snapshot["accounts"].get(account.ref)
            if not isinstance(old, Mapping) or old.get("attempt_id") != attempt_id:
                return False
            current_now = self.now()
            now = iso(current_now)
            self._observe_clock(current_now)
            can_restore_clock = not self._clock_guard.invalidated or (
                clock_generation is not None
                and clock_generation == self._clock_guard.invalidated_generation
            )
            if can_restore_clock:
                self._clock_guard.invalidated = False
                self._clock_guard.invalidated_generation = None
                self._clock_invalidated_at = None
            self._last_view_now = current_now
            return self._commit_record_unlocked(
                account.ref,
                attempt_id,
                {
                    "quota_epoch": account.quota_epoch,
                    "status": "ok",
                    "attempt_id": attempt_id,
                    "attempted_at": old.get("attempted_at"),
                    "checked_at": now,
                    "refresh": None,
                    "last_success": sample,
                    "error": None,
                },
            )

    def commit_failure(self, account: Account, *, attempt_id: str, failure: dict[str, Any]) -> bool:
        with self.state_lock():
            current_accounts = AccountStore(self.root / "pool.json")._load_unlocked()
            current = current_accounts.get(account.ref)
            if current is None or current.quota_epoch != account.quota_epoch:
                return False
            snapshot = self._read_unlocked()
            old = snapshot["accounts"].get(account.ref)
            if not isinstance(old, Mapping) or old.get("attempt_id") != attempt_id:
                return False
            return self._commit_record_unlocked(
                account.ref,
                attempt_id,
                {
                    "quota_epoch": account.quota_epoch,
                    "status": "failed",
                    "attempt_id": attempt_id,
                    "attempted_at": old.get("attempted_at"),
                    "checked_at": iso(self.now()),
                    "refresh": None,
                    "last_success": old.get("last_success"),
                    "error": failure,
                },
            )

    def repair_abandoned(self) -> int:
        count = 0
        with self.state_lock():
            snapshot = self._read_unlocked()
            for record in snapshot["accounts"].values():
                if isinstance(record, dict) and record.get("status") == "checking":
                    record["status"] = "failed"
                    record["checked_at"] = iso(self.now())
                    record["refresh"] = None
                    record["error"] = sanitized_error("refresh_interrupted", "previous quota refresh stopped", retryable=True)
                    count += 1
            if count:
                self._write_unlocked(snapshot)
        return count

    def _sample_fresh(self, record: Mapping[str, Any], now: datetime) -> bool:
        if record.get("status") != "ok":
            return False
        sample = record.get("last_success")
        if not isinstance(sample, Mapping):
            return False
        source = parse_time(sample.get("source_at"))
        checked = parse_time(sample.get("checked_at"))
        fresh_until = parse_time(sample.get("fresh_until"))
        if source is None or checked is None or fresh_until is None:
            return False
        return source <= checked <= now < fresh_until

    def view_for(
        self,
        account: Account,
        snapshot: Mapping[str, Any],
        *,
        now: datetime | None = None,
        unsatisfied: bool = False,
    ) -> dict[str, Any]:
        now = now or self.now()
        now = now.astimezone(timezone.utc)
        self._observe_clock(now)
        if self._clock_guard.invalidated:
            self._clock_invalidated_at = now
        self._last_view_now = now
        record = snapshot.get("accounts", {}).get(account.ref)
        if not isinstance(record, Mapping):
            record = {"quota_epoch": account.quota_epoch, "status": "never", "last_success": None, "error": None, "attempt_id": None, "attempted_at": None, "checked_at": None, "refresh": None}
        if record.get("quota_epoch", LEGACY_QUOTA_EPOCH) != account.quota_epoch:
            freshness = "never"
            reason = "account_state_changed"
        elif not account.enabled:
            freshness = "never"
            reason = "disabled"
        elif not account.local_authenticated(root=self.root):
            freshness = "never"
            reason = "unauthenticated"
        elif record.get("status") == "checking":
            freshness = "checking"
            reason = "checking"
        elif record.get("status") == "failed":
            freshness = "failed"
            reason = "quota_failed"
        elif record.get("status") == "ok" and self._sample_fresh(record, now):
            sample = record.get("last_success")
            exhausted = bool(sample.get("limit_reached")) if isinstance(sample, Mapping) else False
            allowed = bool(sample.get("allowed", True)) if isinstance(sample, Mapping) else False
            primary = sample.get("primary", {}) if isinstance(sample, Mapping) else {}
            secondary = sample.get("secondary", {}) if isinstance(sample, Mapping) else {}
            primary_remaining = primary.get("remaining_percent")
            secondary_remaining = secondary.get("remaining_percent")
            thr = self.exhaust_threshold
            usable_secondary = secondary_remaining is None or (isinstance(secondary_remaining, (int, float)) and secondary_remaining > thr)
            if not allowed or exhausted or primary_remaining is None or primary_remaining <= thr or not usable_secondary:
                reason = "exhausted" if (not allowed or exhausted or (primary_remaining is not None and primary_remaining <= thr) or (secondary_remaining is not None and secondary_remaining <= thr)) else "never_checked"
            else:
                reason = None
            freshness = "fresh"
        elif record.get("status") == "ok":
            freshness = "stale"
            reason = "stale"
        else:
            freshness = "never"
            reason = "never_checked"
        sample = record.get("last_success")
        age = None
        if isinstance(sample, Mapping):
            source = parse_time(sample.get("source_at"))
            if source is not None and source <= now:
                age = max(0.0, (now - source).total_seconds())
        if self._clock_guard.invalidated and isinstance(sample, Mapping):
            # The guard is process-local and remains latched until a check
            # started after the discontinuity commits successfully.  A wall
            # rollback can otherwise make a previously expired sample look
            # young again.
            freshness = "stale"
            reason = "clock_changed"
        if unsatisfied:
            # A forced caller must not mistake a previous green sample for
            # satisfaction when this demand could not obtain a new result.
            current = None
            eligible = False
            reason = "refresh_unavailable"
            freshness = "failed"
        else:
            current = sample if freshness == "fresh" else None
            eligible = reason is None and freshness == "fresh"
        return {
            "id": account.ref,
            "ref": account.ref,
            "enabled": account.enabled,
            "authenticated": account.local_authenticated(root=self.root),
            "status": record.get("status", "never"),
            "attempt_id": record.get("attempt_id"),
            "attempted_at": record.get("attempted_at"),
            "checked_at": record.get("checked_at"),
            "freshness": freshness,
            "age_seconds": age,
            "max_age_seconds": MAX_AGE_SECONDS,
            "current": current,
            "last_success": sample,
            "error": record.get("error"),
            "eligible": eligible,
            "exclusion_reason": reason,
        }

    def rows(
        self,
        accounts: Iterable[Account],
        snapshot: Mapping[str, Any],
        *,
        now: datetime | None = None,
        unsatisfied_refs: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        unsatisfied_refs = unsatisfied_refs or set()
        return [self.view_for(account, snapshot, now=now, unsatisfied=account.ref in unsatisfied_refs) for account in accounts]

    @property
    def exhaust_threshold(self) -> float:
        """Percent remaining at or below which an account is retired from routing."""
        v = getattr(self, "_exhaust_threshold", None)
        if v is not None:
            return v
        raw = os.environ.get("SUBS_POOL_EXHAUST_THRESHOLD")
        try:
            return float(raw) if raw is not None else EXHAUST_THRESHOLD_PERCENT
        except ValueError:
            return EXHAUST_THRESHOLD_PERCENT

    def is_eligible(self, account: Account, *, snapshot: Mapping[str, Any] | None = None, now: datetime | None = None) -> bool:
        snapshot = snapshot if snapshot is not None else self.read()
        return bool(self.view_for(account, snapshot, now=now)["eligible"])


__all__ = [
    "ClockGuard",
    "EXHAUST_THRESHOLD_PERCENT",
    "MAX_AGE_SECONDS",
    "MODULE",
    "QuotaStateError",
    "QuotaStore",
    "REFRESH_TARGET_SECONDS",
    "SCHEMA_VERSION",
    "iso",
    "parse_time",
    "sanitized_error",
]
