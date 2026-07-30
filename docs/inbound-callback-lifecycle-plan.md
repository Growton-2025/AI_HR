# Implementation Plan — Inbound callback lifecycle (grouping + when a candidate leaves the queue)

## The reported problem

The Inbound Callbacks list showed Anchal Shukla three times and Prateek Sharma twice.

**These are not duplicate rows.** Each has a distinct `plivo_call_uuid` and its own
`received_at` — they are genuinely separate calls:

| Candidate | Calls | Times | `dial_status` |
|---|---|---|---|
| Anchal Shukla (11092) | 3 | 10:37:33, 10:39:48, 10:46:57 | cancel, no-answer, no-answer |
| Prateek Sharma (1126) | 2 | 05:08:33 (9s), 05:51:15 | —, no-answer |

Anchal rang three times in ten minutes *because nobody picked up*. Storing one row per call is
correct and worth keeping — it is the audit trail. The bug is that the **UI renders one card per
call**, when a recruiter thinks in terms of one person to ring back. The queue currently punishes
persistence: the more desperate the candidate, the more work the list appears to contain.

## What the queue is actually for

> Candidates who tried to reach us and have not been reached back.

A row should exist exactly while that sentence is true. Everything below follows from that.

Contact gets made in three ways:

1. **Recruiter answers the inbound call live.** Contact made when the call ends.
2. **Recruiter calls back and connects.** Contact made.
3. **Recruiter calls back and does not connect.** Contact *not* made — the contentious case.

## The one genuine product decision: case 3

The founder's standing rule is explicit: *"the moment you complete the callback, irrespective of
whether it is connected or not, it should be marked as callback completed and the number here will
reduce."* Case 3 resolves the row.

That is a defensible stance, and the alternative is worse: gating on connection means an
unreachable candidate sits in the queue forever, and the count stops meaning "work outstanding".

**But as it stands the rule loses people.** A no-answer callback clears the row, and nothing
guarantees the candidate is followed up — they simply disappear from the queue. The fix is not to
change the rule; it is to make resolution always leave a trail, so the *cadence* owns "keep
trying" and the callback queue stays a true inbox of un-actioned items. Clean separation:

- **Inbound Callbacks** = "you have not dealt with this yet."
- **Call cadence** = "keep pursuing this person."

Recommendation: **keep the founder's rule, add the trail.**

---

## Phase 1 — Group by candidate (fixes the visible problem)

- `GET /calls/inbound` returns one item per candidate instead of per call. Group by
  `candidate_id`; fall back to the normalised phone number for unknown callers so they still
  collapse.
- Each item gains `call_count`, `first_received_at`, `last_received_at`, and the id of every
  underlying row (`inbound_ids`) so resolution can clear them together.
- Card reads `3 calls · last 1 hour ago` rather than three identical cards.
- **Group only unresolved rows.** A candidate who calls again after being resolved must reappear
  as a fresh item, not be swallowed into the old group.
- The stat card and sidebar badge count **candidates, not calls** — today's `5` becomes `2`.

## Phase 2 — Resolve the whole group, not one row

`POST /calls/inbound/{id}/resolve` currently clears a single row, so calling Anchal back once
would leave two of her three cards behind. Resolution takes the candidate: clear every unresolved
row for them in one transaction, recording the same `resolved_by` / `resolved_call_id`.

## Phase 3 — Auto-resolve calls answered live

`record_inbound_dial_result` sets `status = 'answered'` when a recruiter picks up the ring-all,
but the Pending filter is `status IN ('pending', 'answered')` — so a candidate you *just spoke to*
still sits in the queue asking to be called back.

- Drop `'answered'` from the Pending filter.
- Resolve on hangup for answered calls (`/api/plivo/incoming-hangup`), with
  `resolution = 'answered_live'`. Nothing to call back — the conversation happened.
- The in-call modal already resolves on close; this covers the case where the recruiter answers
  and the modal never opens (unknown caller, or the tab is closed mid-call).

## Phase 4 — Resolution always leaves a trail

Add `resolution VARCHAR(32)` to `inbound_calls`: `answered_live | called_back | manual | auto`.

- Any resolution records **how**, so "the count went down" is auditable.
- When a callback resolves *without connecting*, ensure the candidate has an open follow-up call
  task. This is what stops the founder's rule from silently dropping people: the row leaves the
  inbox, and the cadence picks the candidate up.
- Do **not** gate resolution on the outcome. The follow-up is a consequence of resolution, not a
  condition for it.

## Phase 5 — Manual control

- **Mark resolved** without calling — wrong number, junk, or handled over email/LinkedIn.
- **Dismiss** for unknown numbers that are not candidates at all.
- Both go through the same grouped resolve path with `resolution = 'manual'`.

---

## Resulting lifecycle

```
candidate calls
      |
      +-- recruiter answers live ......... conversation ends -> resolved (answered_live)
      |
      +-- nobody answers -> voicemail ---> row PENDING, grouped with any earlier calls
                                              |
                                              +-- 1-Click Call Back, connected ---> resolved (called_back)
                                              +-- 1-Click Call Back, no answer --> resolved (called_back)
                                              |                                     + follow-up task created
                                              +-- Mark resolved ------------------> resolved (manual)
                                              |
                                              +-- candidate calls again ----------> joins the same group
                                                                                     (count increments,
                                                                                      no new card)
after resolution, a new call creates a NEW group and the candidate reappears
```

## Verification

1. Three calls from one candidate render **one** card reading `3 calls`; badge counts 1.
2. Calling that candidate back clears **all three** rows, not one.
3. A candidate answered live never appears in Pending.
4. A no-answer callback resolves the group **and** leaves an open follow-up task.
5. A resolved candidate who calls again reappears as a new pending group.
6. Unknown numbers still group by number and are never silently dropped.
7. Re-run the suites against a freshly measured baseline (currently 5 pre-existing failures in
   `calls_initiate_test.py` — the baseline drifts, always re-measure).
8. Live-call verification uses the **Nethranand P S** profile only, per the standing rule.

## Risks

- **Grouping hides the audit trail if done in the wrong place.** Group in the read model only;
  never collapse or delete `inbound_calls` rows. Each call is a real event.
- **Resolving a group is a multi-row write** — do it in one transaction, or a partial failure
  leaves some cards behind and the count wrong.
- **Phase 3 changes what Pending means.** Anyone reading the count as "inbound calls received"
  will see it drop; it means "candidates awaiting a callback", which is the intent.
- **Unknown callers grouped by number** will merge two different people sharing a number
  (rare, but reception desks exist). Acceptable — they are shown flagged, not auto-actioned.
