"""One-time (idempotent) rewrite of ``candidates.normalized_linkedin`` into the
canonical ``/in/<slug>`` form.

Production held three formats side by side (bare slug 2,504 rows, raw URL
1,667, canonical 1,493 on 2026-09-25), so the same person's rows could not
find each other. Canonicalising creates collisions inside a uniqueness scope
— typically a legacy-pipeline master row and a newer catalog master row for
the same profile. Those rows are both real and both keep their history; the
loser's key gets the ``_legacy_<id>`` suffix the existing master dedupe already
uses, and person-history linking reads through that suffix.

The planner is pure so it can be tested on fixtures; ``apply`` runs the UPDATEs.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from backend.services.linkedin_normalize import (
    LEGACY_SUFFIX_RE,
    is_canonical_linkedin_key,
    normalize_linkedin,
    person_key,
)

logger = logging.getLogger(__name__)

# (id, owner_user_id, is_archived, normalized_linkedin, linkedin, updated_at)
Row = Tuple[int, Optional[int], bool, Optional[str], Optional[str], object]


@dataclass
class CanonicalPlan:
    updates: List[Tuple[int, str]] = field(default_factory=list)   # (id, new key)
    collisions: Dict[Tuple[Optional[int], str], List[int]] = field(default_factory=dict)
    unresolvable: List[int] = field(default_factory=list)          # no key derivable
    unchanged: int = 0


def _derive_key(stored: Optional[str], linkedin: Optional[str]) -> Optional[str]:
    return person_key(stored) or normalize_linkedin(linkedin)


def plan_canonical_keys(rows: Iterable[Row]) -> CanonicalPlan:
    plan = CanonicalPlan()
    # Rows that lost an earlier dedupe keep their suffix: re-canonicalising
    # them would only collide with the row that won.
    groups: Dict[Tuple[Optional[int], str], List[dict]] = defaultdict(list)
    for cid, owner, archived, stored, linkedin, updated_at in rows:
        if stored and LEGACY_SUFFIX_RE.search(stored):
            plan.unchanged += 1
            continue
        key = _derive_key(stored, linkedin)
        if not key:
            if stored or linkedin:
                plan.unresolvable.append(cid)
            continue
        if archived:
            # Outside every unique index; just canonicalise.
            if key != stored:
                plan.updates.append((cid, key))
            else:
                plan.unchanged += 1
            continue
        groups[(owner, key)].append({
            "id": cid, "stored": stored, "updated_at": updated_at,
        })

    for (owner, key), members in groups.items():
        if len(members) == 1:
            m = members[0]
            if m["stored"] == key:
                plan.unchanged += 1
            else:
                plan.updates.append((m["id"], key))
            continue
        # Keeper: the row already holding the canonical key, else the most
        # recently updated, else the lowest id — deterministic and idempotent.
        canonical_holders = [m for m in members if m["stored"] == key]
        if canonical_holders:
            keeper = canonical_holders[0]
        else:
            keeper = max(members, key=lambda m: ((m["updated_at"] or 0) if not isinstance(m["updated_at"], str) else m["updated_at"], -m["id"]))
        plan.collisions[(owner, key)] = [m["id"] for m in members]
        if keeper["stored"] != key:
            plan.updates.append((keeper["id"], key))
        for m in members:
            if m is keeper:
                continue
            plan.updates.append((m["id"], f"{key}_legacy_{m['id']}"))
    return plan


def load_rows(cur) -> List[Row]:
    cur.execute(
        """
        SELECT id, owner_user_id, COALESCE(is_archived, FALSE), normalized_linkedin, linkedin, updated_at
        FROM candidates
        WHERE normalized_linkedin IS NOT NULL OR linkedin IS NOT NULL
        """
    )
    return [tuple(r) for r in cur.fetchall()]


_BATCH = 500


def _update_batch(cur, pairs: List[Tuple[int, str]]) -> None:
    """One statement per batch: the remote DB costs ~0.6s per round trip, so
    4,000 single-row UPDATEs at startup would hold the container's warm-up
    probe for 40 minutes. ~10 statements instead."""
    values_sql = ",".join(cur.mogrify("(%s, %s)", (cid, key)).decode() for cid, key in pairs)
    cur.execute(
        "UPDATE candidates AS c SET normalized_linkedin = v.key "
        f"FROM (VALUES {values_sql}) AS v(id, key) "
        "WHERE c.id = v.id AND c.normalized_linkedin IS DISTINCT FROM v.key"
    )


def apply_canonical_keys(cur, plan: CanonicalPlan) -> int:
    """Write the plan. Suffixed keys first so a keeper never trips the unique
    index on a value its loser is still holding."""
    suffixed = [u for u in plan.updates if LEGACY_SUFFIX_RE.search(u[1])]
    keepers = [u for u in plan.updates if not LEGACY_SUFFIX_RE.search(u[1])]
    for group in (suffixed, keepers):
        for start in range(0, len(group), _BATCH):
            _update_batch(cur, group[start:start + _BATCH])
    return len(suffixed) + len(keepers)


def canonicalise_linkedin_keys(cur) -> CanonicalPlan:
    rows = load_rows(cur)
    plan = plan_canonical_keys(rows)
    if plan.updates:
        n = apply_canonical_keys(cur, plan)
        logger.info(
            "Canonicalised %s candidates.normalized_linkedin value(s); %s already canonical; "
            "%s collision group(s) resolved with _legacy_ suffixes; %s row(s) with no derivable key.",
            n, plan.unchanged, len(plan.collisions), len(plan.unresolvable),
        )
        if plan.collisions:
            sample = list(plan.collisions.items())[:10]
            logger.warning("LinkedIn key collisions (owner, key) -> ids, first %s: %s", len(sample), sample)
    return plan
