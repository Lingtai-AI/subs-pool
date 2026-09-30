# Codex module contract

Codex is the only built-in provider module. It owns auth, account
configuration, WHAM parsing, quota sidecar/freshness/refresh, eligibility,
routing, affinity, upstream wire behavior, SSE, and the local Responses
server. Generic `subs_pool` code does not inspect any of these facts.

The module keeps the baseline two-rule scheduler: a full input-prefix and
effective-config match stays on its current in-memory chain account; a miss is
a weighted choice. Full-prefix hashing, UUID session labels, chain capacity,
successful-only commits, and response normalization remain in the relocated
baseline files. A failed, partial, cancelled, or incomplete generation never
commits and is never retried or replayed.

Before selection, the module reads account configuration and
`quota-v1.json` together under `state.lock`. A candidate must be enabled,
locally authenticated, on the current quota
epoch, `status=ok`, fresh under the fixed 60-second source-time policy, and
have known positive remaining capacity. A nullable secondary window remains
nullable according to the baseline WHAM parser; if it is present and
exhausted it excludes the account. Unknown quota is never turned into zero or
green capacity. No candidate means a local HTTP 503 with code
`quota_unavailable`, before any upstream generation request.

The sidecar is the sole quota authority. Its top-level schema is version 1,
module `codex`, monotonic revision, replacement timestamp, and account-keyed
records containing epoch, status (`never`/`checking`/`ok`/`failed`), attempt
metadata, refresh marker, last successful sample, and sanitized error. It
contains no tokens or raw provider bodies. Failed refreshes preserve the
historical `last_success` but never expose it as current.

Sidecar writes use same-directory exclusive temporary files, flush/fsync,
atomic replace, directory fsync where supported, and mode 0600 state files.
`filelock` backs the persistent `state.lock` and
`quota-v1.refresh.lock` inodes. State locking is short; network work never
holds it. One pool-wide owner performs at most two account checks concurrently
and foreground coordination is bounded. Followers join completed overlapping
waves or fail closed; they do not steal a live kernel lock.

The direct Codex root is explicit `CODEX_POOL_HOME`, otherwise
`${SUBS_POOL_HOME:-~/.subs-pool}/codex`. The root itself contains `pool.json`,
`quota-v1.json`, `state.lock`, and `quota-v1.refresh.lock`; there is no
`accounts/` relocation or hidden migration. Existing `pool.json` entries
without `quota_epoch` mean `legacy-v1` for compatibility. Mutations that
change enablement, auth, or identity advance the epoch and remove its quota
record.

The local Responses server forwards request fields unchanged, with three
owning-layer translations: `stream` is always true and `store` is always false
upstream, and the standard sampling/length controls Codex rejects with HTTP 400
(`max_output_tokens`, `temperature`, `top_p`, `truncation`) are omitted from
the upstream request. Unsupported stateful fields (`previous_response_id`,
`conversation`, truthy `store`/`background`) are rejected locally with 400.

The foreground server remains `--listen 127.0.0.1:8765`, loopback-only, and
user-managed. It refreshes on startup, at the 30-second target while running,
and just-in-time when no current candidate exists. It does not install a
service, auto-start from the TUI, persist affinity, or perform generation
failover.
