# Candidate history linking — plan

**Ask.** When a candidate is added (by LinkedIn profile), every previous Hayasa
interaction with that person — calls, inbound callbacks, LinkedIn and email
threads, status changes, notes — must show on the new profile automatically:
when we contacted them, the exact conversation, the timeline.

**Why it does not happen today (verified 2026-09-25).** Every history store is
keyed on `candidates.id`, and the same person legitimately exists as several
rows: one master-library row (`owner_user_id IS NULL`) plus one copy per
recruiter pool (`uq_candidates_recruiter_li` on `(owner_user_id,
normalized_linkedin)`). A new row is a new `id`, so it starts with empty
history even when another row for the same LinkedIn profile has months of it.
Nothing merges or links rows, and the Activity endpoint is further limited to
call lists the current user created.

---

## 1. Facts the design rests on

| Store | Key | Owner scoping today | File |
|---|---|---|---|
| `calls` (outbound, cadence) | `candidate_id` | via `call_lists.created_by` | `backend/api/routes/calls.py:1066` |
| `inbound_calls` | `candidate_id` (nullable), `from_number` | none | `calls.py:1211` |
| `deleted_calls` | `call_row->>'candidate_id'` (JSONB) | none, never read back | `calls.py:1194` |
| `candidate_outreach` (LinkedIn + email) | `(candidate_id, recruitment_role_id)`; threads in `li_chat_history_cache`, `email_chat_history_cache` JSONB | none | `backend/db/migrations/add_outreach_tracking.sql`, `backend/api/routes/outreach.py` |
| `candidate_status_history` | `candidate_id` | none; no API reads it | `backend/services/candidate_status_log.py` |
| `candidates.notes` | the row | — | overwritten, not appended |

Identity:

* `normalize_linkedin()` (`backend/services/linkedin_normalize.py`) returns the
  lower-cased URL **path** only, e.g. `/in/jane-doe`. It does not reject
  non-LinkedIn URLs, keeps sub-paths (`/in/x/details/...`), and does not
  canonicalise `/pub/` or locale hosts.
* The column holds **three formats** in production: `/in/slug` (app), bare
  `slug` (legacy pipeline), and raw URLs (`scripts/ingest_*`). The HeyReach
  poller already matches all three by hand (`heyreach_reply_sync.py:121`).
* `PATCH /candidates/{id}` can change `linkedin` without recomputing
  `normalized_linkedin` (`candidates.py:697-727`).
* Email matching on import is case-sensitive and untrimmed; phone is never an
  identity key except for inbound-call matching (last 10 digits).

Provider APIs (for history that predates our records or was lost):

* HeyReach — base `https://api.heyreach.io/api/public`, header `X-API-KEY`,
  **300 req/min shared**. `POST /inbox/GetConversationsV2` filters by
  `leadProfileUrl` (also `linkedInAccountIds`, `campaignIds`, `offset/limit`);
  `GetChatroom(accountId, conversationId)` returns the full thread;
  `POST /lead/GetLead` takes `profileUrl`. Already wrapped in
  `integrations/heyreach.py:612-685` (`get_li_chat_history`).
* Smartlead — `GET /api/v1/leads/?api_key&email=` returns the lead with
  `lead_campaign_data` (all campaigns); `GET
  /api/v1/campaigns/{campaign_id}/leads/{lead_id}/message-history`
  (`event_time_gt`, `show_plain_text_response`). Production already hits
  Smartlead's **200 req/min** ceiling ("Account rate limit exceeded"), so any
  backfill must be queued and throttled. Wrapped in
  `integrations/smartlead.py:209-245` (`get_chat_history`).

---

## 2. Design: link, don't merge

Merging rows is off the table: ownership, unique indexes, cadences and role
links all hang off `candidates.id`, and two recruiters *should* be able to hold
their own copy. Instead:

1. **A person key.** `normalized_linkedin` becomes a true canonical key (one
   format everywhere), with email and phone as secondary keys.
2. **A link table** recording which rows are the same person and how we know.
3. **One read model** — a unified timeline — that answers for a *person*, not a
   row, and is what every history surface renders.
4. **An on-create hook** that links a new row to its prior rows immediately
   (optimistic: the UI shows "known — N previous interactions" the moment the
   row is saved) and queues a provider backfill in the background.

### 2.1 Schema

```sql
-- Same person, different candidate rows.
CREATE TABLE IF NOT EXISTS candidate_person_links (
    candidate_id        INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    person_key          TEXT    NOT NULL,          -- canonical /in/slug
    matched_on          VARCHAR(16) NOT NULL,      -- 'linkedin' | 'email' | 'phone' | 'manual'
    linked_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    linked_by           VARCHAR(255),              -- email or 'system'
    PRIMARY KEY (candidate_id)
);
CREATE INDEX IF NOT EXISTS ix_person_links_key ON candidate_person_links (person_key);

-- Threads pulled from HeyReach/Smartlead for a person before any row existed,
-- or recovered for a new row. Keyed on the person, not the row, so a third
-- copy of the same candidate does not trigger a third fetch.
CREATE TABLE IF NOT EXISTS person_provider_threads (
    person_key   TEXT NOT NULL,
    provider     VARCHAR(16) NOT NULL,             -- 'heyreach' | 'smartlead'
    thread_ref   TEXT NOT NULL,                    -- conversationId / campaign:lead
    messages     JSONB NOT NULL,                   -- same shape as *_chat_history_cache
    fetched_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (person_key, provider, thread_ref)
);

-- Backfill queue (rate-limited workers drain it).
CREATE TABLE IF NOT EXISTS person_history_jobs (
    id           SERIAL PRIMARY KEY,
    person_key   TEXT NOT NULL,
    provider     VARCHAR(16) NOT NULL,
    status       VARCHAR(16) NOT NULL DEFAULT 'queued', -- queued|running|done|failed
    attempts     INTEGER NOT NULL DEFAULT 0,
    next_run_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_error   TEXT,
    UNIQUE (person_key, provider)
);
```

Add to the existing schema function (`ensure_calls_schema_ready` or a new
`ensure_person_history_migrations`) **and to its fast-path sentinel** — the
sentinel omission is how `plivo_hangup_cause` went missing on production.

### 2.2 Canonical identity (prerequisite, its own PR)

* `normalize_linkedin()`:
  * host must be `linkedin.com` or `*.linkedin.com` (locale subdomains
    collapse); otherwise `None`;
  * keep only `/in/<slug>` (drop `/details/...`, query, fragment, trailing
    slash); `/pub/<slug>/a/b/c` → `/in/<slug>`; percent-decode and lower-case;
  * a bare `slug` input → `/in/slug`.
* `PATCH /candidates/{id}` recomputes `normalized_linkedin` when `linkedin`
  changes and re-links the person (see 2.4).
* One backfill (`backend/db/candidate_pool_migrate.py` style, idempotent):
  rewrite bare slugs and raw URLs to `/in/slug`. Dedupe collisions the way the
  existing master dedupe does (`_legacy_<id>` suffix), then log them for a
  human decision.
* Email keys: `LOWER(TRIM(email))` in all match SQL; phone key: last 10 digits
  (the inbound-call rule).

### 2.3 Person resolution (`backend/services/person_identity.py`)

```python
def person_key_for(candidate_row) -> str | None            # canonical /in/slug
def resolve_person_rows(cur, candidate_id, *, include_archived=True) -> list[int]
    # rows sharing person_key (via candidate_person_links, else on-the-fly by
    # normalized_linkedin); union rows sharing LOWER(email) or phone tail,
    # but only when the LinkedIn key is absent on either side (email/phone are
    # weaker keys — never let a shared "info@" address glue two people).
def link_candidate(cur, candidate_id, *, by) -> LinkResult
    # upsert candidate_person_links; return prior row ids + counts of prior
    # calls / threads / status changes (for the optimistic UI badge)
```

Archived rows are *included* for history: an archived row's calls are still
"we contacted this person".

### 2.4 On-create hook (all three write paths)

After the INSERT in:

* `create_candidate` (`candidates.py:950`)
* `_bulk_insert_new_import_rows` and the slow path in
  `candidate_imports.py`
* `assign_master_to_recruiter` (`candidate_pool.py:607`)

call `link_candidate()` in the same transaction, then **after commit** enqueue
`person_history_jobs` for `heyreach` (if a LinkedIn key exists) and
`smartlead` (if an email exists). The single-add response gains:

```json
{"success": true, "data": {...}, "known_person": {
   "prior_candidate_ids": [2519, 13302], "calls": 3, "linkedin_replies": 1,
   "emails": 4, "last_interaction_at": "2026-09-24T13:47:00Z"}}
```

so the modal can show the badge immediately (optimistic), while the provider
backfill fills in anything we never stored.

Also fix the single-add duplicate check to match import semantics: ignore
archived rows (or offer "restore" instead of 409).

### 2.5 Unified timeline read model

`GET /candidates/{id}/timeline?scope=person|row` → chronological events for
the *person* (default) from all linked rows:

| type | source | dedupe key |
|---|---|---|
| `call` | `calls` (completed), `deleted_calls` (flagged `archived: true`) | `plivo_call_uuid` or `calls.id` |
| `inbound_call` | `inbound_calls` | `plivo_call_uuid` |
| `linkedin_message` | every linked row's `li_chat_history_cache` ∪ `person_provider_threads(heyreach)` | message `id` |
| `email_message` | `email_chat_history_cache` ∪ `person_provider_threads(smartlead)` | message `id` |
| `status_change` | `candidate_status_history` | row id |
| `note` | `candidates.notes` per row (current value, attributed to owner) | row id |
| `link` | `candidate_person_links` ("linked to existing profile #2519 by LinkedIn") | — |

Each event carries `candidate_id`, `owner` (recruiter email or "master"),
`role` when known, and `you: bool`. Dedupe first, then sort by `occurred_at`.
Keep `GET /candidates/{id}/activity` as a thin alias (`scope=row`, calls only)
so existing UI keeps working during rollout.

**Visibility policy (decision needed):** today Activity shows only the current
user's own call lists. The ask is company-wide ("see when *we* previously
contacted"). Proposed: admins see everything; recruiters see all events but
transcripts/recordings of *other* recruiters' calls are summarised
(outcome, date, recruiter) unless `permissions.view_all_history`. This is a
product call — flagging, not deciding.

### 2.6 Provider backfill worker

A single background loop per process is what the HeyReach/Smartlead pollers do
today; do the same but **claim jobs in the DB** (`UPDATE ... WHERE status='queued'
AND next_run_at <= now() ... FOR UPDATE SKIP LOCKED LIMIT 1`) so the four
gunicorn workers never fetch the same person twice — the same lesson as the
insights claim (`claim_insights_run`).

* HeyReach: `GetConversationsV2(leadProfileUrl=<full URL>)` → for each
  conversation `GetChatroom(accountId, conversationId)` → normalise via the
  existing formatter (`heyreach.py:687-740`) → upsert
  `person_provider_threads`. Budget: ≤ 60 req/min from this worker (20% of the
  shared 300).
* Smartlead: `leads/?email=` → `lead_campaign_data[]` → `message-history` per
  campaign → upsert. Budget ≤ 40 req/min (production already brushes the
  200/min limit from the reply poller).
* Backoff on 429: `next_run_at = now() + 2^attempts min`, give up after 5.
* Idempotent: re-running a job only adds missing messages.

Also pre-warm on **lookup** (2.7) so that by the time the recruiter clicks
"Add", the provider history is usually already there.

### 2.7 Frontend

1. **Add Candidate modal** (`AddCandidateModal.jsx`): on LinkedIn-field blur,
   `GET /candidates/lookup?linkedin=<url>` → if known, render "Already in
   Hayasa: 3 calls · 1 LinkedIn reply · last contact 24 Sep (Aman)" with a
   link to the timeline, and change the CTA to "Add to my pool & link
   history". Same lookup for email. Optimistic: the badge renders from the
   lookup response; no waiting on providers.
2. **Roles / Talent Pool rows**: a small "history" chip
   (`known_person.counts`) sourced from the list endpoints via a cheap
   aggregate (`person_history_summary` view: person_key → counts, last
   interaction). Cached per worker with the existing drift cadence.
3. **CandidateConversationModal**: new first tab **Timeline** =
   `/timeline?scope=person`; Email/LinkedIn/Calls tabs unchanged.
4. **Calls modal** (`Calls.jsx`): "Previous attempts" strip above the notes
   box, from the same timeline (calls only), so a recruiter sees last
   outcome/reason before dialing — this is exactly the "Not Reachable — Incoming
   freeze" note the team asked about.
5. **Import summary** (`candidate_uploads`): "N of M rows are people we have
   contacted before" using the link results.

### 2.8 Optimistic-UI contract

* Create → respond with `known_person` synchronously (one indexed query).
* Timeline endpoint answers from local rows immediately and includes
  `"pending_backfill": ["heyreach"]` when a job is queued; the UI shows a
  subtle "fetching older LinkedIn messages…" line and re-fetches once
  (`X-Poll-After: 15`), never a spinner that blocks.
* Never block a create/import on a provider call.

---

## Status (2026-09-25)

* **PR 1 — identity** shipped (`aaf8692`): canonical `normalize_linkedin`,
  `person_key`, batched key backfill, PATCH recompute, lower-cased emails.
  Note: hosted skips startup migrations (`RUN_STARTUP_MIGRATIONS` unset), so
  the backfill runs from `ensure_calls_schema_ready` on first request, with
  a sentinel that checks for non-canonical keys.
* **PR 2 — link + timeline** shipped: `candidate_person_links`,
  `person_identity.py` (resolve / link / summary / lookup),
  `person_timeline.py`, `GET /candidates/lookup`,
  `GET /candidates/{id}/timeline`, `known_person` on create, hooks in every
  pool write path; frontend: Add-Candidate banner, History tab, "Previous
  attempts" in the Calls modal. Timeline cost on hosted ≈ 4 s (8 statements);
  candidate for a single CTE later.
* **PR 3 — provider backfill** shipped, **off by default**:
  `backend/services/person_history_backfill.py` — `person_history_jobs`
  (claimed with `FOR UPDATE SKIP LOCKED`, exponential backoff, 5 attempts),
  `person_provider_threads`, one daemon worker per process (tick 12 s, one
  job per tick ⇒ ≤ 20 jobs/min across 4 workers), HeyReach by profile URL,
  Smartlead by email across every campaign of the lead, last 12 months.
  Jobs are enqueued on candidate create and on Add-modal lookup; the
  timeline merges provider threads (deduped by message id) and reports
  `pending_backfill`, which the UI shows as "Fetching older … messages" and
  re-reads once after 15 s. **To enable on hosted:** app setting
  `ENABLE_PERSON_HISTORY_BACKFILL=true` (HEYREACH_API_KEY / SMARTLEAD_API_KEY
  already present); optional `PERSON_HISTORY_TICK_SECONDS`.

## 3. Work breakdown (incremental, each shippable)

| # | Deliverable | Files | Tests |
|---|---|---|---|
| 0 | Canonical `normalize_linkedin`, PATCH recompute, key backfill + report | `services/linkedin_normalize.py`, `routes/candidates.py`, `db/candidate_pool_migrate.py` | unit table of URL forms (locale hosts, `/pub/`, `/details`, bare slug, non-LinkedIn → None); PATCH recompute; backfill idempotency |
| 1 | Schema (3 tables) + sentinel; `person_identity.py` resolver + `link_candidate` | `routes/calls.py` (schema fn), `services/person_identity.py` | resolver on fixture rows (linkedin / email-only / phone-only / archived); no cross-person glue on shared email |
| 2 | On-create hook in the 3 write paths; `known_person` in responses; single-add 409 vs archived fix | `routes/candidates.py`, `services/candidate_imports.py`, `services/candidate_pool.py` | each path links; import counts; assign merges keep link |
| 3 | `GET /candidates/{id}/timeline` + `GET /candidates/lookup` | `routes/candidates.py` (+ `services/person_timeline.py`) | union/dedupe/sort; visibility policy per role; alias parity with `/activity` |
| 4 | Backfill jobs + worker (claim-by-DB, rate budgets, backoff) | `services/person_history_backfill.py`, `main.py` lifespan | claim exclusivity across two fake workers; 429 backoff; idempotent upsert |
| 5 | Frontend: lookup badge in Add modal, Timeline tab, Calls "previous attempts", import summary | `AddCandidateModal.jsx`, `CandidateConversationModal.jsx`, `Calls.jsx`, `TalentPool.jsx` | (no JS test infra) manual checklist + `npm run build` |
| 6 | Backfill existing data: link all current rows by person_key; enqueue provider jobs for people with outreach rows lacking threads | `scripts/backfill_person_links.py` | dry-run report first |

Feature flag `ENABLE_PERSON_HISTORY` (env) gates 2–5 so 0–1 can ship first.

## 4. Risks and mitigations

* **False links via shared email/phone** (agency mailboxes, family phones):
  email/phone only link when no LinkedIn key contradicts, and every link
  records `matched_on` so it can be undone (`DELETE FROM candidate_person_links`
  + a "Not the same person" action later).
* **Provider rate limits**: DB-claimed queue with per-provider budgets; the
  existing pollers keep priority.
* **Per-worker caches**: the summary chip is served from a DB view with the
  60 s drift sync — no new module-level dicts (see
  `docs/…` and the 2026-09-25 incident: dial state and profile cache went
  stale across gunicorn workers).
* **Privacy across recruiters**: policy decision in 2.5; default to summaries
  for others' transcripts.
* **Key backfill collisions**: report, don't auto-merge; the `_legacy_<id>`
  suffix keeps the unique index intact.

## 4b. Call-modal issues found while testing (tracked here, fixed separately)

Found 2026-09-25 during the Plivo work; each is fixed in the calls modal,
listed so the "previous attempts" strip (2.7 §4) is built on a modal that
behaves:

* A modal opened right after a hangup handled the *previous* call's
  terminated event on mount and jumped to "Call ended → Log Call Details"
  while the new dial rang underneath. Events older than the modal are now
  ignored (`modalMountedAtMsRef`, `lastHandledCallEventRef` seeded).
* "Cancel" in the log form re-entered the ended state instead of closing.
* "Ringing" with no live SDK call froze the modal (watchdog added).
* Redials within ~2 s of a hangup died in Plivo's DELAYED NEGOTIATION
  (cooldown added); two browsers on one SIP line displaced each other
  (per-device endpoints).

## 5. Open decisions for the product owner

1. Cross-recruiter visibility of full transcripts/recordings (2.5).
2. Should an archived row's history count as "previous contact"? (proposed: yes)
3. Should adding a known person auto-copy their latest notes/status into the
   new row, or only show them in the timeline? (proposed: show only; copying
   overwrites the recruiter's own context)
4. Backfill depth for providers: all time, or last 12 months?
