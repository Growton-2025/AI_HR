"""Write the output of ``profile_mapper.map_scraped_profile`` to the database.

``persist_mapped_profile(tables, dry_run=True)`` does every insert inside one
transaction, reads back per-table row counts, then rolls back, so the full write
path can be tested against the real schema without leaving anything behind.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from psycopg2.extras import Json

from backend.db.connection import get_db_connection, return_db_connection

logger = logging.getLogger(__name__)

CHILD_SPECS = {
    # table, parent fk column, child table, child fk column, child columns
    "functional_experiences": ("functional_experience_roles", "functional_experience_id",
                               ["company", "activity_type", "reason", "duration_years"], "roles"),
    "industry_experiences": ("industry_experience_roles", "industry_experience_id",
                             ["company", "industry", "reason", "duration_years"], "roles"),
    "segment_experiences": ("segment_experience_roles", "segment_experience_id",
                            ["company", "segment", "reason", "duration_years"], "roles"),
}


_LIMITS: Dict[str, Dict[str, int]] = {}


def _limits(cur, table: str) -> Dict[str, int]:
    """varchar(n) limits per column, so model-written text is clipped, not rejected."""
    if table not in _LIMITS:
        cur.execute("SELECT column_name, character_maximum_length FROM information_schema.columns "
                    "WHERE table_name = %s AND character_maximum_length IS NOT NULL", (table,))
        _LIMITS[table] = dict(cur.fetchall())
    return _LIMITS[table]


def _insert(cur, table: str, row: Dict[str, Any], returning: bool = False):
    limits = _limits(cur, table)
    row = {c: (v[:limits[c]] if isinstance(v, str) and c in limits else v) for c, v in row.items()}
    cols = list(row)
    marks = ["%s::vector" if c == "embedding" else "%s" for c in cols]
    vals = []
    for c in cols:
        v = row[c]
        if c == "embedding" and v is not None:
            v = "[" + ",".join(str(x) for x in v) + "]"
        elif c == "raw_fields":
            v = Json(v, dumps=lambda o: json.dumps(o, default=str))
        vals.append(v)
    cur.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(marks)})"
                + (" RETURNING id" if returning else ""), vals)
    return cur.fetchone()[0] if returning else None


def _upsert_company(cur, c: Dict[str, Any]) -> int:
    """Create the company, or fill only its empty columns. Never overwrites."""
    cur.execute("SELECT id FROM companies WHERE LOWER(name) = LOWER(%s) LIMIT 1", (c["name"],))
    hit = cur.fetchone()
    fillable = ["funding_stage", "revenue", "business_model", "product_service",
                "customer_segment", "customer_presence", "culture_type", "headquarters"]
    if not hit:
        return _insert(cur, "companies", {k: c.get(k) for k in ["name", *fillable] if c.get(k) not in (None, "", [])}
                       , returning=True)
    sets, vals = [], []
    for k in fillable:
        if c.get(k) in (None, "", []):
            continue
        sets.append(f"{k} = CASE WHEN {k} IS NULL OR {k}::text IN ('', '{{}}') THEN %s ELSE {k} END")
        vals.append(c[k])
    if sets:
        cur.execute(f"UPDATE companies SET {', '.join(sets)}, updated_at = NOW() WHERE id = %s", [*vals, hit[0]])
    return hit[0]


def persist_mapped_profile(tables: Dict[str, Any], *, owner_user_id: Optional[int] = None,
                           pool_source: str = "catalog_from_upload", dry_run: bool = True,
                           created_by: str = "profile_mapper",
                           extra_candidate_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``extra_candidate_fields`` are extra ``candidates`` columns to write as-is
    (e.g. ``source_upload_ids``); they override nothing the mapper produced."""
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise RuntimeError("Database connection failed")
    counts: Dict[str, int] = {}
    try:
        with conn.cursor() as cur:
            company_ids = {c["name"]: _upsert_company(cur, c) for c in tables["companies"]}

            cand = {**(extra_candidate_fields or {}), **tables["candidates"], "pool_source": pool_source,
                    "owner_user_id": owner_user_id, "created_by": created_by}
            cid = _insert(cur, "candidates", cand, returning=True)

            for r in tables["roles"]:
                _insert(cur, "roles", {"candidate_id": cid, "company_id": company_ids.get(r["company"]),
                                       "title": r["title"], "details": r["details"],
                                       "duration_years": r["duration_years"]})
            for table in ("titles_held", "education", "company_years", "experience_gaps",
                          "education_gaps", "industry_gaps"):
                for row in tables[table]:
                    _insert(cur, table, {"candidate_id": cid, **row})

            for table, (child, fk, cols, key) in CHILD_SPECS.items():
                block = tables.get(table)
                if not block:
                    continue
                pid = _insert(cur, table, {"candidate_id": cid, "score": block["score"],
                                           "rationale": block["rationale"]}, returning=True)
                for item in block.get(key) or []:
                    _insert(cur, child, {fk: pid, **{c: item.get(c) for c in cols}})
            geo = tables.get("geography_experiences")
            if geo:
                gid = _insert(cur, "geography_experiences", {"candidate_id": cid, "score": geo["score"],
                                                              "rationale": geo["rationale"]}, returning=True)
                for region in geo.get("regions") or []:
                    _insert(cur, "geography_experience_regions", {"geography_experience_id": gid, "region": str(region)})

            # read back inside the same transaction
            for t in ("roles", "titles_held", "education", "company_years", "experience_gaps",
                      "education_gaps", "industry_gaps", "functional_experiences",
                      "industry_experiences", "segment_experiences", "geography_experiences"):
                cur.execute(f"SELECT count(*) FROM {t} WHERE candidate_id = %s", (cid,))
                counts[t] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM functional_experience_roles WHERE functional_experience_id IN "
                        "(SELECT id FROM functional_experiences WHERE candidate_id = %s)", (cid,))
            counts["functional_experience_roles"] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM geography_experience_regions WHERE geography_experience_id IN "
                        "(SELECT id FROM geography_experiences WHERE candidate_id = %s)", (cid,))
            counts["geography_experience_regions"] = cur.fetchone()[0]
            cur.execute("SELECT (embedding IS NOT NULL), total_experience_years, team_management_score "
                        "FROM candidates WHERE id = %s", (cid,))
            emb, tot, tm = cur.fetchone()
            cur.execute("SELECT count(*) FROM companies WHERE id = ANY(%s)", (list(company_ids.values()),))
            counts["companies_touched"] = cur.fetchone()[0]
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
        return {"candidate_id": cid if not dry_run else None, "dry_run": dry_run, "counts": counts,
                "embedding_stored": emb, "total_experience_years": float(tot) if tot is not None else None,
                "team_management_score": tm}
    except Exception:
        conn.rollback()
        raise
    finally:
        return_db_connection(conn)
