# Implementation Plan — Inbound Calls (ring everyone, live pickup, voicemail fallback)

## Context

Source: a Zoom screen-share of a Google AI Studio prototype ("Remix: RecruitDash") mocking a
redesigned Calls Workspace. It is a **design spec, not working code**. The capability is capturing
calls candidates make *to us* after outreach, so nothing is lost and a recruiter can talk to them
immediately.

Specified in the prototype:

- `INBOUND CALLBACKS` stat card with a `1-Click` badge — "3 pending candidate call-back(s)".
- `Inbound Callbacks` tab beside Due Today / Upcoming / Completed / Call Lists.
- **"Inbound Missed Callbacks"** panel — *"Candidates who called back after receiving outreach
  voicemails or emails. One-click callback automatically completes call tasks."*
- Sub-filters `All / Pending / Resolved` and a **`+ Log Incoming Call`** action.
- Row: avatar, name, `Title at Company`, amber **Missed Callback** badge, phone, relative time,
  a **Current Status** chip, a free-text note, and **View History** + **1-Click Call Back**.
- **View History** opens the existing `CandidateConversationModal` (LinkedIn/Email/Calls/Tasks/Resume).

From the call audio (transcribed), which **overrides** two things visible in the mock:

- *"On the right-hand top, let us not do that. It will crowd the whole UX."* — the
  **`Inbound Calls 3` pill at top-right of Talent Pool is explicitly rejected.** Do not build it,
  even though the prototype shows it. The **only** indicator is a small bubble on the **left
  sidebar** nav: *"a small bubble of number of unanswered or uncalled back calls."*
- *"The moment you complete the callback, **irrespective of whether it is connected or not**, it
  should be marked as callback completed and the number here will reduce."* — resolution does
  **not** depend on the call connecting. Completing the callback attempt resolves the row and
  decrements the badge.

Also confirmed verbally: matching is **by phone number** (*"This is matched based on the number.
If the number is present, it will automatically show which person"*), and the trigger is
*"the number which is associated with the recruiter — whenever somebody calls back, that should be
tracked."*

Decisions from the founder:
- **Ring everyone**, first to answer wins.
- A candidate calling while a recruiter is working must surface as a **quiet in-app banner** they
  can answer on the spot — **no ringtone**, nothing interrupting what they're doing.
- If nobody answers, play a **short voicemail** and leave the callback pending.

Sources: extracted video frames **plus the call audio transcribed via Whisper** (transcript
retained in the session scratchpad). Note the audio describes **logging and managing missed
inbound callbacks**; the *live silent pickup while a recruiter is working* (Phase 4) is an
additional requirement given separately by the founder, not something stated in the recording.

## Current state

| Area | Today |
|---|---|
| `backend/api/routes/plivo.py` | Entirely outbound — `/dial`, `/recording`, `/credentials`, `/insights`. **No inbound route.** |
| `plivo_service.setup_plivo()` | Provisions **one** Application (`answer_url = {public}/api/plivo/dial`) and **one** endpoint, state in `data/plivo_softphone_state.json`; re-points the app when the tunnel URL rotates. |
| `GET /api/plivo/credentials` | Returns the **same** endpoint username/password to every caller, and **has no authentication dependency**. |
| `calls` table | Outbound cadence tasks only (`list_id`, `due_date`, `task_title`, sequencing). **No `direction`, no inbound concept.** |
| `frontend/src/context/VoIPContext.jsx` | Registers only outbound handlers (`onCallRemoteRinging`/`onCallAnswered`/`onCallTerminated`/`onCallFailed`). Already disables SDK tones (`setConnectTone(false)`, `setRingToneBack(false)`) and sweeps Plivo audio via `stopPlivoManagedAudio()`. |

## Verified Plivo mechanics

- A number is bound to an **Application**; inbound hits its **Answer URL**, which returns XML.
  Optional **Hangup URL** fires at the end.
- Parameters: `CallUUID`, `From`, `To`, `Direction=inbound`, `CallStatus`
  (`ringing`/`in-progress`/`completed`/`no-answer`), plus `HangupCause`, `Duration` on hangup.
  Plivo **retries callbacks and may duplicate them** → `CallUUID` must be an idempotency key.
- `<Dial>` accepts **multiple `<User>` elements** and rings them **simultaneously, first to answer
  wins** — exactly the "ring everyone" requirement:
  ```xml
  <Dial timeout="25" action="{public}/api/plivo/incoming-unanswered">
    <User>sip:alice@phone.plivo.com</User>
    <User>sip:bob@phone.plivo.com</User>
  </Dial>
  ```
  The `action` URL receives `DialStatus` (`timeout`/`no-answer`), `DialALegUUID`, `DialBLegUUID`
  (empty if unanswered), `DialRingStatus`.
- Browser SDK: `client.on('onIncomingCall', callInfo => …)` → `client.answer(callInfo.callUUID)`,
  `client.reject(uuid)`, `client.ignore(uuid)` (stops the ring). With a call already live, use
  `answer(uuid, 'ignore')` so a second inbound never rings over a conversation.

**Critical constraint:** the existing Application's answer URL is `/api/plivo/dial`, which expects
a `To` field to place an outbound leg. Binding `PLIVO_NUMBER` to that same app would push inbound
calls into the outbound dialer and break them. **Inbound needs its own Application.**

---

## Phase 1 — Per-user Plivo endpoints (prerequisite)

"Ring everyone" needs one SIP endpoint per recruiter: with a single shared endpoint we cannot fork
the call, and cannot tell *who* answered. This phase also closes a real hole — `/credentials`
currently hands SIP credentials to any unauthenticated caller.

- New table `plivo_endpoints`: `user_id UNIQUE`, `endpoint_id`, `username`, `password`,
  `app_id`, `created_at`, `last_registered_at`.
- `plivo_service.ensure_endpoint_for_user(user_id)` — create-or-fetch, reusing the existing
  create/persist logic. Keep the current shared endpoint working as a fallback so outbound dialling
  never regresses mid-migration.
- **Add `Depends(deps.get_current_user)` to `/api/plivo/credentials`** and return that user's
  endpoint. This is a behaviour change for an endpoint that is currently public — verify the
  frontend always calls it authenticated.
- Track liveness: `POST /api/plivo/registered` on SDK `onLogin`, clearing on `onLogout`, so the
  webhook rings only endpoints plausibly online (stale registrations just don't answer, which the
  `timeout` already handles).

## Phase 2 — Data model

Add `inbound_calls` through the existing idempotent pattern in `ensure_calls_schema_ready()`:

```
id, candidate_id (nullable — unknown callers), from_number, to_number,
plivo_call_uuid UNIQUE, received_at, answered_by_user_id (nullable), answered_at,
duration, hangup_cause, call_status, dial_status,
status ('pending'|'answered'|'resolved'), note,
resolved_at, resolved_by, resolved_call_id, recording_url, transcript, created_by
```

Reusing `calls` was considered and rejected: cadence sequencing, Due Today counts, call-list counts
and next-attempt logic all assume outbound task rows, so inbound rows would corrupt them.
`UNIQUE(plivo_call_uuid)` provides the idempotency Plivo's retries demand.

## Phase 3 — Inbound webhooks

- **Inbound Application** provisioned in `setup_plivo()` alongside the endpoint app:
  answer `{public}/api/plivo/incoming`, hangup `{public}/api/plivo/incoming-hangup`. Bind
  `PLIVO_NUMBER` to it and re-point on tunnel rotation exactly as the endpoint app already does.
- **`POST /api/plivo/incoming`** — match `From` via the existing `normalize_number()` against
  `candidates.mobile_phone`/`phone`; upsert the `inbound_calls` row on `CallUUID`; return the
  multi-`<User>` `<Dial>` above, one `<User>` per registered endpoint. If none are registered, go
  straight to voicemail.
- **`POST /api/plivo/incoming-unanswered`** (the `action` URL) — on `DialStatus` of
  `timeout`/`no-answer`, return a short voicemail leg (`<Speak>` greeting + `<Record>`) and leave
  the row `pending`. If answered, record `answered_by_user_id` from the winning `DialBLegUUID`.
- **`POST /api/plivo/incoming-hangup`** — write duration / hangup cause / final status.
- All three are **unauthenticated** (Plivo carries no bearer token) and must **validate Plivo's
  signature**, mirroring how `/api/plivo/recording` is already exposed.

## Phase 4 — Silent in-app incoming banner

In `frontend/src/context/VoIPContext.jsx` (the provider, **not** `Calls.jsx`, so it works on
whatever page the recruiter is on, changes no route and steals no focus):

- Register **`onIncomingCall`**; immediately suppress any ring — `setRingTone(false)` where
  available plus the existing `stopPlivoManagedAudio()` sweep — so **no sound ever plays**.
- Render a **non-blocking corner banner** (not a modal): candidate name/company when matched, raw
  number when not, with **Accept** and **Dismiss**. It must not pause polling, close drawers, or
  interrupt typing.
- Accept → `client.answer(callUUID)`, reusing the existing in-call UI (timer, notes, wrap-up) built
  for outbound. If a call is already live, `answer(uuid, 'ignore')`.
- On hangup, mark the `inbound_calls` row answered/resolved and complete the candidate's pending
  outbound task.
- Because every browser now registers a **distinct** endpoint, all online recruiters see the banner
  and the first to accept takes the call; Plivo cancels the rest.

## Phase 5 — Inbound Callbacks workspace

- `frontend/src/pages/Calls.jsx`: `Inbound Callbacks` tab, the stat card, the panel with
  All/Pending/Resolved, and the prototype's row layout. **1-Click Call Back** reuses the outbound
  dial path then calls resolve; **View History** opens the existing `CandidateConversationModal`.
- APIs: `GET /api/calls/inbound` (status filter, joined to candidate name/title/company/status),
  `POST /api/calls/inbound/{id}/resolve` (completes the pending outbound task — the prototype's
  "one-click callback automatically completes call tasks"), `POST /api/calls/inbound/manual`
  (backs `+ Log Incoming Call`).
- **Resolution is outcome-independent.** Per the call: completing the callback marks the row
  resolved *whether or not it connected*. So `resolve` fires when the callback attempt ends — do
  **not** gate it on `outcome`/`DialStatus`. A no-answer callback still clears the row.
- Pending count folded into the existing `/api/calls/stats` payload — `Calls.jsx` already polls it
  every 15s; do **not** add another poll (this codebase has a history of request storms).
- **Left sidebar bubble only.** The count feeds the badge on the Calls nav item and decrements as
  rows resolve. **No top-right pill on Talent Pool** — explicitly rejected as UX clutter.

---

## Verification

1. **Webhook unit tests** with replayed Plivo payloads: known caller matches a candidate; unknown
   caller stores `candidate_id = NULL`; a **duplicate `CallUUID` does not create a second row**;
   `DialStatus=no-answer` produces the voicemail branch and leaves the row `pending`.
2. **Ring-everyone**: two browsers logged in as different users → dial `PLIVO_NUMBER` → both banners
   appear silently, one accepts, the other clears. Confirm `answered_by_user_id` is the accepter.
3. **Silence**: with the OS volume up, confirm no tone plays on inbound — the regression risk is
   Plivo's own ringtone element slipping past `stopPlivoManagedAudio()`.
4. **Non-disruption**: trigger an inbound call while typing in a Roles filter and while a drawer is
   open — neither may be interrupted.
5. **No outbound regression**: existing browser dialling still works; the new Application must not
   disturb the endpoint app. Re-run the suite against the **31-failure baseline** (stash and compare
   — the baseline drifts; never assume).
6. Per the founder's standing rule, live dialling tests use **only the Nethranand P S profile**.

## Risks

- **`/credentials` becoming authenticated** is a behaviour change to a currently public route —
  the highest-risk item for breaking existing dialling. Ship Phase 1 behind the shared-endpoint
  fallback and verify outbound before removing it.
- **Tunnel rotation**: inbound breaks the moment the cloudflared/ngrok URL changes unless the new
  Application is re-pointed like the existing one. Never pin `NGROK_URL`.
- **Per-user endpoint cost/limits** on the Plivo account are unverified.
- **"The number which is associated with the recruiter"** (from the call) implies a per-recruiter
  *phone number*, but the system has a single `PLIVO_NUMBER`. Ringing everyone works off one shared
  number, so this is not blocking — but if each recruiter is meant to have their own DID, that is a
  separate provisioning change. Confirm before Phase 3.
- Unknown/unmatched numbers are shown flagged rather than dropped (founder: not a concern that most
  candidates lack numbers).
