# Implementation Plan — Mass update (add to shortlist, bulk status, bulk call list)

## The ask

From the selection bar that already exists on Roles (and Talent Pool): with candidates ticked (or
"Use All Filtered"), be able to **add to a call list** (exists), **add to shortlist** (new), and
**change the status dropdown in bulk** (new).

## Key finding: shortlist and status are the same thing

`Shortlisted` is a value in `RECRUITMENT_STAGES` (`browse.py:219`), not a separate table or flag.
There is no shortlist membership model to build.

```python
RECRUITMENT_STAGES = [
    'To be started', 'Shortlisted', 'Rejected', 'For Future',
    'Reached out - Linkedin', 'Reached out - Phone', 'Not Interested',
    'Followup / In conversation', 'Shortlist - Rejected', 'High CTC',
    'Duplicate', 'Not responding', 'Internal Review', 'Shared with customer'
]
```

So **"Add to Shortlist" is a preset for "set status = Shortlisted"**. The two requested features
collapse into one backend capability: *bulk status update*. Build that once; "Add to Shortlist"
becomes a one-click shortcut beside the generic picker, not a second code path.

## Current state

| Piece | Today |
|---|---|
| Selection bar | Exists on `Roles.jsx:2498` and `TalentPool.jsx:3948`. Count, "Use All Filtered", "Add to Call List", Cancel. |
| Add to Call List | `POST /api/calls/add-candidates` — already bulk, already set-based. |
| Status change | `POST /api/browse/candidates/{id}/status` — **one candidate per request**. |
| Bulk endpoints | **None** beyond `add-candidates`. `grep -E "bulk|batch"` over all routes returns nothing. |
| Shortlist | No endpoint. It is a status value. |

## Constraints that shape the design

### 1. The database is remote and slow

`add-candidates` carries the note *"each statement costs ~0.6s against the remote DB"*, and does
ownership check + duplicate count + insert in **one** round trip using CTEs.

Looping the existing single-status endpoint 77 times would mean 77 HTTP requests and several
hundred SQL statements — roughly a minute of wall clock, with cache invalidation thrashing
throughout. **The bulk path must be set-based**, mirroring `add-candidates`.

### 2. The single endpoint does far more than an UPDATE

`update_status` (`browse.py:1655`) also:

- authorizes per candidate (`_authorize_candidate_update`, which may hit the DB per id),
- invalidates candidate-count caches with `refresh_profile_ids`,
- invalidates the role-detail cache,
- invalidates the calls cache,
- **and, for terminal statuses, completes that candidate's pending call rows** so they stop
  resurfacing in the calling loop.

That last one is not incidental — skipping it in bulk would leave candidates marked
`Not Interested` still being dialled. The bulk endpoint must reproduce every one of these, but
**once for the batch** rather than per candidate.

`TERMINAL_CANDIDATE_STATUSES` is the existing source of truth; reuse it, do not re-list it.

### 3. Selection semantics are already hard-won — do not re-derive them

`Roles.jsx:1248-1292` documents real bugs already fixed:

- `selectedIds` is **pruned when a filter narrows the view**, because stale hidden ids were
  previously carried into "Add to Call List".
- The header checkbox selects **only the loaded/visible rows**, never "everything matching the
  filter" — that escalation is exclusively the "Use All Filtered" button.

New actions must consume `selectedIds` / `allFilteredSelected` exactly as `Add to Call List`
does. Any new selection logic risks reintroducing those bugs.

### 4. "Use All Filtered" is capped at the loaded page

`Roles.jsx:780` fetches with `page_size=5000`, and "Use All Filtered" materialises ids from
`filteredCandidates` client-side. So it means *"all filtered rows I have loaded"*. Under 5000 per
role this is invisible; above it, the button would silently act on a subset.

Options: (a) accept and document, (b) have the bulk endpoint accept **filter criteria** instead of
ids and resolve them server-side. (b) is correct long-term but materially larger — it means
sharing the browse filter parser between endpoints. **Recommend (a) now**, with an explicit
count in the confirm step so the number acted on is never a surprise, and (b) only if a role
realistically exceeds 5000.

### 5. Authorization has to be decided, not inherited

`_authorize_candidate_update` raises **403 for a single candidate**. In bulk, one unauthorized id
among 77 must not fail the whole batch — nor should it silently update rows the user cannot edit.

**Recommend:** filter to authorized ids **in SQL** (owner match, or admin, or role access), update
those, and return `{updated, skipped}` so the UI can say "72 updated, 5 skipped (not yours)".
All-or-nothing would make a mixed selection unusable.

---

## Phase 1 — Bulk status endpoint

`POST /api/browse/candidates/bulk-status` — `{ candidate_ids: [int], status: str }`

- Validate `status` against `RECRUITMENT_STAGES`; reject anything else (the dropdown is not a
  free-text field, and a typo would create an unfilterable stage).
- Authorize set-based, returning the ids actually permitted.
- One `UPDATE ... WHERE id = ANY(%s::int[])` for the permitted ids.
- If the status is terminal, one set-based `UPDATE calls ... WHERE candidate_id = ANY(...)` to
  complete pending rows in this recruiter's lists — the same rule as the single endpoint, applied
  once.
- Invalidate the three caches **once**, passing all touched ids to `refresh_profile_ids`.
- Return `{updated, skipped, status}`.
- Cap batch size (suggest 5000, matching the page cap) so one request cannot lock the table for
  an unbounded set.

## Phase 2 — Selection-bar actions

In the existing bar on `Roles.jsx`, beside `Add to Call List`:

- **Add to Shortlist** — one click, calls Phase 1 with `Shortlisted`.
- **Set Status ▾** — the same `RECRUITMENT_STAGES` list the per-row dropdown uses, driving the
  same endpoint.

Both take `Array.from(selectedIds)`, exactly as `Add to Call List` does. No new selection logic.

Confirm before applying, showing the exact count and target status — this is the first action in
the app that can change hundreds of records at once, and it has no undo.

## Phase 3 — Optimistic update and cache coherence

A candidate's status appears in the roles grid, Talent Pool, the calls list and call rows
(`candidate_status`). The single-update store action already patches calls caches
(`useAppStore.js:2828`); the bulk action must patch **every affected id** the same way, or rows
will show stale statuses until a refetch.

Reuse `updateCallsByCandidateId` / `patchCallsByCandidateIdAcrossCaches` in a loop over the
returned ids — client-side loop is fine, it is the *server* round trips that are expensive.

## Phase 4 — Talent Pool parity

`TalentPool.jsx:3948` has the same bar. Once Phase 2 works on Roles, lift the buttons into a
shared component rather than copy-pasting, so the two bars cannot drift.

## Phase 5 (optional) — Filter-based selection

Only if roles exceed 5000 candidates: accept filter criteria server-side so "Use All Filtered"
means all matching rows, not all loaded rows. Deferred deliberately — see constraint 4.

---

## Verification

1. Select 3, set a status → exactly those 3 change; the other rows are untouched.
2. "Use All Filtered (77)" → all 77 change in **one** request; measure it (should be ~1s, not ~60s).
3. Set a **terminal** status in bulk → those candidates' pending call rows are completed, so they
   leave the calling loop. This is the regression most likely to be missed.
4. Mixed ownership selection → `{updated, skipped}` reported honestly; nothing the user cannot
   edit is modified.
5. Apply a filter that narrows the list while rows are selected → stale hidden ids are still
   pruned (guard against reintroducing the `Roles.jsx:1248` bug).
6. An invalid status is rejected with 400, not written.
7. Statuses update live in the roles grid **and** any open calls list, with no refetch.
8. Re-run suites against a freshly measured baseline (currently 5 pre-existing failures in
   `calls_initiate_test.py` — the baseline drifts, always re-measure).

## Risks

- **No undo.** A mis-click on "Use All Filtered" then a status change rewrites hundreds of rows.
  The confirm step is the only safeguard; consider recording the previous status per candidate so
  a follow-up "undo last bulk change" is possible.
- **Terminal statuses have side effects on calling.** Bulk-setting `Not Interested` silently
  closes call tasks. Correct, but it should be stated in the confirm dialog, not discovered.
- **Cache invalidation at scale** — `refresh_profile_ids` with 5000 ids may be slower than
  clearing wholesale. Measure both; prefer whichever is faster for large batches.
- **`Duplicate` and `Rejected` are terminal too.** Bulk-marking duplicates is a plausible
  first use of this feature, and it will close call tasks as a side effect. Intended, worth
  saying out loud.
