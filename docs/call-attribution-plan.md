# Implementation Plan — Correct call attribution (kill the shared-endpoint corruption path)

## Status — all four phases implemented

| Phase | State |
|---|---|
| 1 — degraded state is visible | **Done.** `degraded` flag on `/credentials`, ERROR log, `DegradedCallingBanner` app-wide. |
| 2 — refuse to share | **Done.** One retry, then a single-holder claim; the second user gets 503 `plivo_endpoint_unavailable`. |
| 3 — attribute by dial token | **Done.** `calls.dial_token` → `X-PH-DialToken` → exact-match UPDATE. Legacy username fallback retained and logged. |
| 4 — provisioning robustness | **Done.** `_read()` separates "DB down" from "no row"; orphaned endpoints adopted by alias. |

**Two things remain open and both need a live call:**

1. **The `X-PH-` forwarding is still unverified on this account.** Everything is built so this failing is
   survivable — the backend logs `[DialAttribution] Falling back to username matching` and the old path
   still runs — but until a real dial is placed we do not know whether the token arrives, i.e. whether
   D1 is *actually* fixed in production or only in the tests. Place one call and
   `grep DialTokenProbe` / `grep DialAttribution` in the backend log. `Matched call … by dial token`
   means it works.
2. **The legacy username fallback must be deleted** once the logs show it is never used. Left in place it
   preserves the exact bug this plan removes.

Unverified and untouched: the Plivo account's endpoint limit (Phase 4's third bullet) — it is the other
way `ensure_endpoint_for_user` can start failing for everyone at once.

## Context

Surfaced while adding busy-endpoint filtering for concurrent recruiters. `GET /api/plivo/credentials`
falls back to the **shared** SIP endpoint when per-user provisioning fails
(`backend/api/routes/plivo.py:293`). The fallback exists for a good reason — a Plivo hiccup should
degrade dialling, not kill it — but it fails **silently and destructively**: two recruiters on one
endpoint username get each other's recordings and transcripts written onto the wrong candidate.

The fallback is the trigger, but it is not the root cause. The root cause is that **call attribution
is keyed on the endpoint username plus "most recent row wins"**, which is only correct when exactly
one call per username is ever in flight. Fixing the fallback alone leaves the same bug reachable by
a single recruiter redialling quickly.

## Current state

| Step | Where | What it does |
|---|---|---|
| 1 | `Calls.jsx:2285` | `initiateCall(call.id, { plivoUsername: endpointUsername })` |
| 2 | `calls.py:2181` | writes `calls.plivo_endpoint_username` on that row |
| 3 | `VoIPContext.jsx:792` | `client.call(dialNumber)` — **no extra headers** |
| 4 | `plivo.py:65` | `/dial` webhook parses the username out of `From: sip:<user>@…` |
| 5 | `plivo_service.py:249` | `UPDATE calls … WHERE plivo_endpoint_username = %s ORDER BY updated_at DESC LIMIT 1` |
| 6 | `plivo.py:195` | recording callback keyed on the `plivo_call_uuid` stamped in step 5 |

Step 5 is the defect. Nothing in the chain carries an identifier for *which dial attempt* the webhook
belongs to, so the row is guessed by recency.

## The three defects, ranked

### D1 — Shared endpoint misattributes across recruiters (silent data corruption)

Recruiter A initiates for candidate X at T0; recruiter B initiates for candidate Y at T1 > T0. Both
`calls` rows now carry the same `plivo_endpoint_username`. A's `/dial` webhook lands at T2 and step 5
picks **B's** row (most recently updated). A's `plivo_call_uuid` is stamped on B's call, so A's
recording, transcript and AI insights attach to **candidate Y**.

No error is raised. The only trace is `logger.warning` at `plivo.py:295`. A recruiter reading
candidate Y's call notes has no way to know they belong to someone else.

Two further consequences of a shared username, both introduced or worsened by the concurrency work:

- `mark_endpoint_busy(username=…)` marks **every** recruiter on the shared endpoint busy, so one
  person dialling removes all of them from inbound ring-all.
- Inbound `answered_by_user_id` cannot be resolved at all — the winning B-leg maps to a username, not
  a person.

### D2 — Stale `call_uuid` satisfies the dial handshake (single recruiter, no sharing needed)

`last_calls[username]` (`plivo_service.py:230`) is **never cleared**. `waitForPlivoDial`
(`VoIPContext.jsx:815`) accepts *any* `call_uuid` present for the username and ignores the
`to_number` and `seen_at` that `/api/plivo/call-state/{username}` already returns. So a recruiter's
second dial passes the handshake instantly on the **previous** call's UUID, before the new webhook
has arrived. Today that only produces a false-pass gate rather than a wrong row, but it hides genuine
webhook failures and is one refactor away from becoming a second misattribution path.

### D3 — Endpoint provisioning leaks Plivo endpoints when the DB is down

`ensure_endpoint_for_user._read()` (`plivo_service.py:424`) returns `None` both when the row is
absent **and** when `get_db_connection` fails. The two are indistinguishable, so with the DB down the
function proceeds to `client.endpoints.create(...)` on **every login**, then fails at `_write` and
returns `None`. Each attempt orphans a live endpoint in the Plivo account, and the caller silently
takes the shared fallback — i.e. a DB blip both leaks resources and arms D1.

---

## Phase 1 — Make the degraded state loud (do first, ~1h)

Does not fix anything; stops it being invisible, which is the property that makes D1 dangerous.

- `/api/plivo/credentials` returns `degraded: true` and a `reason` alongside the shared credentials.
- `plivo.py:295` logs at **ERROR**, not `warning`.
- `VoIPContext` stores it and exposes it; the Calls UI shows a persistent (non-dismissible) warning
  strip: *"Calling is running in degraded mode — call recordings may be attached to the wrong
  candidate. Contact support before making calls."* Reuse the existing `voipError` / `voipMeta` /
  `voipActionLabel` plumbing rather than adding new state.

## Phase 2 — Refuse to share rather than corrupt (~2h)

Silent corruption is worse than a blocked dial. Recruiters can retry a dial; they cannot un-mix two
candidates' transcripts.

- Retry `ensure_endpoint_for_user` once with a short backoff before considering the fallback —
  transient Plivo 5xx is the common case and a retry resolves it.
- Track which `user_id` currently holds the shared endpoint (in-memory is sufficient; it is a
  degraded path). Grant the fallback to the **first** user only. Any second user gets **HTTP 503**
  with `code: "plivo_endpoint_unavailable"` and an actionable message, not a shared username.
- Because a single holder is guaranteed, `mark_endpoint_busy(username=…)` on the shared endpoint is
  no longer ambiguous.

Skippable if Phase 3 ships immediately — but keep it if Phase 3 slips, since it is what actually
stops the corruption.

## Phase 3 — Attribute by an explicit per-attempt token (the real fix, ~1 day)

Removes the whole class of bug: attribution stops depending on username uniqueness or recency, so
D1 and D2 both die regardless of whether the fallback is in play.

**Verified mechanism:** Plivo's browser SDK takes `client.call(dest, extraHeaders)` where headers
named `X-PH-*` are forwarded to the Application's answer URL as request parameters
([Browser SDK reference](https://www.plivo.com/docs/voice/client/browser/reference),
[SIP headers](https://www.plivo.com/docs/voice/xml/request/sip-headers/)).

1. `POST /api/calls/{id}/initiate` generates a `dial_token` (uuid4) and stores it on the `calls` row.
   New nullable column `dial_token VARCHAR(64)` + index, added through the existing idempotent
   `ensure_calls_schema_ready()` pattern — **remember to update the sentinel column check**
   (`calls.py:813`), or the migration silently never runs.
2. `initiate` returns the token; `placeCall` passes it as
   `client.call(dialNumber, { 'X-PH-DialToken': token })`.
3. `/api/plivo/dial` reads the token from the webhook payload and `record_browser_dial` matches
   `WHERE dial_token = %s` — an exact row, no `ORDER BY … LIMIT 1`.
4. Keep the username path as a fallback **for one release** for calls placed by a client that has not
   picked up the new bundle, logging whenever it is used so we can confirm the token path covers
   everything before deleting it.
5. `waitForPlivoDial` polls by **token** rather than username, which fixes D2 for free — a stale UUID
   from a previous attempt no longer matches.

## Phase 4 — Provisioning robustness (~2h)

- `_read()` distinguishes "no connection" from "no row": raise or return a sentinel on connection
  failure so `ensure_endpoint_for_user` aborts **before** calling `client.endpoints.create`.
- Reconcile orphans: on startup, list Plivo endpoints whose `alias` matches `recruiter_<id>` and
  adopt any that exist in Plivo but not in `plivo_endpoints`, instead of creating a duplicate.
- Confirm the account's endpoint limit — the inbound plan flagged it as unverified and it is the
  other way `ensure_endpoint_for_user` can start failing for everyone at once.

---

## Verification

1. **D1 regression test** — two `calls` rows sharing one `plivo_endpoint_username`, dial webhooks
   arriving out of order; assert each `plivo_call_uuid` lands on its **own** row. This test must fail
   against today's code (confirm it does before fixing) and pass after Phase 3.
2. **Phase 2** — force `ensure_endpoint_for_user` to return `None`; assert user 1 gets the shared
   endpoint with `degraded: true` and user 2 gets 503, not the same username.
3. **D2** — dial, hang up, dial again; assert the second `waitForPlivoDial` does not return the first
   call's UUID.
4. **D3** — simulate `get_db_connection` returning `None`; assert `client.endpoints.create` is
   **never** called.
5. **No outbound regression** — the whole point is that normal dialling is untouched. Re-run
   `tests/plivo_regressions_test.py` and `tests/calls_initiate_test.py` and compare against a
   freshly-measured baseline (currently 5 pre-existing failures in `calls_initiate_test.py`, all
   `inbound_pending` assertions — **the baseline drifts, always stash and re-measure, never assume**).
6. Live dialling verification uses **only the Nethranand P S profile**, per the standing rule.

## Risks

- **`X-PH-` header forwarding is verified from Plivo's docs but not yet against this account's SDK
  version.** Prove it end-to-end with a single logged webhook before building Phase 3 on top of it;
  if headers do not arrive, the fallback is to pass the token in the dialled URI.
- **Phase 2 turns a silent degrade into a hard 503** for the second user. That is the intended
  trade, but it is a visible behaviour change — ship it with Phase 1's messaging so the failure is
  explained rather than mysterious.
- **The one-release username fallback in Phase 3.4 must actually be removed.** Left in, it preserves
  the exact bug being fixed. Log every use and delete it once the count is zero.
- **Historical data is already wrong.** Any misattribution that has already happened is not repaired
  by this plan, and there is no reliable way to detect it retroactively — the correct call UUID for a
  given row was never recorded. Worth deciding whether to flag calls placed during known
  shared-endpoint windows.
