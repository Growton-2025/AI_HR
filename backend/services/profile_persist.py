"""Write the output of ``profile_mapper.map_scraped_profile`` to the database.

``persist_mapped_profile(tables, dry_run=True)`` does every insert inside one
transaction, reads back per-table row counts, then rolls back, so the full write
path can be tested against the real schema without leaving anything behind.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from psycopg2.extras import Json, execute_values

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
_limits_loaded = False


def _limits(cur, table: str) -> Dict[str, int]:
    """varchar(n) limits per column, so model-written text is clipped, not rejected.
    Loaded for every table in one query, once per process."""
    global _limits_loaded
    if not _limits_loaded:
        cur.execute("SELECT table_name, column_name, character_maximum_length FROM information_schema.columns "
                    "WHERE character_maximum_length IS NOT NULL")
        for table_name, column, length in cur.fetchall():
            _LIMITS.setdefault(table_name, {})[column] = length
        _limits_loaded = True
    return _LIMITS.get(table, {})


def _insert_many(cur, table: str, rows: List[Dict[str, Any]]) -> None:
    """Insert plain rows (no embedding / raw_fields) in one statement."""
    if not rows:
        return
    cols = list(rows[0])
    if any(list(r) != cols for r in rows):          # uneven rows: fall back to one by one
        for r in rows:
            _insert(cur, table, r)
        return
    limits = _limits(cur, table)
    values = [tuple(v[:limits[c]] if isinstance(v, str) and c in limits else v for c, v in r.items())
              for r in rows]
    execute_values(cur, f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s", values)


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


_COMPANY_FILLABLE = ["funding_stage", "revenue", "business_model", "product_service",
                     "customer_segment", "customer_presence", "culture_type", "headquarters"]


def _existing_companies(cur, names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Stored companies keyed by lower-case name (lowest id wins), in one query."""
    if not names:
        return {}
    cur.execute(f"SELECT DISTINCT ON (LOWER(name)) LOWER(name), id, {', '.join(_COMPANY_FILLABLE)} "
                "FROM companies WHERE LOWER(name) = ANY(%s) ORDER BY LOWER(name), id",
                ([n.lower() for n in names],))
    return {row[0]: dict(zip(["id", *_COMPANY_FILLABLE], row[1:])) for row in cur.fetchall()}


def _upsert_company(cur, c: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> int:
    """Create the company, or fill only its empty columns. Never overwrites.
    ``existing`` is the stored row for this name, or ``None`` when there is none."""
    if not existing:
        # Another profile being saved at the same moment may be creating this employer
        # too. The name is unique, so yield to whichever insert lands first and use its row.
        row = {k: c.get(k) for k in ["name", *_COMPANY_FILLABLE] if c.get(k) not in (None, "", [])}
        limits = _limits(cur, "companies")
        row = {k: (v[:limits[k]] if isinstance(v, str) and k in limits else v) for k, v in row.items()}
        cur.execute(f"INSERT INTO companies ({', '.join(row)}) VALUES ({', '.join(['%s'] * len(row))}) "
                    "ON CONFLICT (name) DO NOTHING RETURNING id", list(row.values()))
        created = cur.fetchone()
        if created:
            return created[0]
        cur.execute("SELECT id FROM companies WHERE name = %s", (row["name"],))
        return cur.fetchone()[0]
    # Only columns that are empty in the table and filled in the new data need a write.
    fill = {k: c[k] for k in _COMPANY_FILLABLE
            if c.get(k) not in (None, "", []) and existing.get(k) in (None, "", [], "{}")}
    if fill:
        sets = [f"{k} = CASE WHEN {k} IS NULL OR {k}::text IN ('', '{{}}') THEN %s ELSE {k} END" for k in fill]
        cur.execute(f"UPDATE companies SET {', '.join(sets)}, updated_at = NOW() WHERE id = %s",
                    [*fill.values(), existing["id"]])
    return existing["id"]


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
            stored = _existing_companies(cur, [c["name"] for c in tables["companies"]])
            company_ids: Dict[str, int] = {}
            # Sorted, so two saves that share employers always take them in the same
            # order and can never wait on each other in a circle.
            for c in sorted(tables["companies"], key=lambda item: item["name"]):
                key = c["name"].lower()
                company_ids[c["name"]] = _upsert_company(cur, c, stored.get(key))
                stored.setdefault(key, {"id": company_ids[c["name"]]})   # same employer, other spelling

            cand = {**(extra_candidate_fields or {}), **tables["candidates"], "pool_source": pool_source,
                    "owner_user_id": owner_user_id, "created_by": created_by}
            cid = _insert(cur, "candidates", cand, returning=True)

            # child rows go in one statement per table
            _insert_many(cur, "roles", [
                {"candidate_id": cid, "company_id": company_ids.get(r["company"]), "title": r["title"],
                 "details": r["details"], "duration_years": r["duration_years"]} for r in tables["roles"]])
            for table in ("titles_held", "education", "company_years", "experience_gaps",
                          "education_gaps", "industry_gaps"):
                _insert_many(cur, table, [{"candidate_id": cid, **row} for row in tables[table]])

            for table, (child, fk, cols, key) in CHILD_SPECS.items():
                block = tables.get(table)
                if not block:
                    continue
                pid = _insert(cur, table, {"candidate_id": cid, "score": block["score"],
                                           "rationale": block["rationale"]}, returning=True)
                _insert_many(cur, child, [{fk: pid, **{c: item.get(c) for c in cols}}
                                          for item in block.get(key) or []])
            geo = tables.get("geography_experiences")
            if geo:
                gid = _insert(cur, "geography_experiences", {"candidate_id": cid, "score": geo["score"],
                                                              "rationale": geo["rationale"]}, returning=True)
                _insert_many(cur, "geography_experience_regions",
                             [{"geography_experience_id": gid, "region": str(region)}
                              for region in geo.get("regions") or []])

            # read back inside the same transaction, in one round trip
            count_tables = ("roles", "titles_held", "education", "company_years", "experience_gaps",
                            "education_gaps", "industry_gaps", "functional_experiences",
                            "industry_experiences", "segment_experiences", "geography_experiences")
            cur.execute(
                "SELECT " + ", ".join(f"(SELECT count(*) FROM {t} WHERE candidate_id = %(cid)s)" for t in count_tables)
                + ", (SELECT count(*) FROM functional_experience_roles WHERE functional_experience_id IN "
                  "(SELECT id FROM functional_experiences WHERE candidate_id = %(cid)s))"
                + ", (SELECT count(*) FROM geography_experience_regions WHERE geography_experience_id IN "
                  "(SELECT id FROM geography_experiences WHERE candidate_id = %(cid)s))"
                + ", (SELECT count(*) FROM companies WHERE id = ANY(%(companies)s))"
                + ", c.embedding IS NOT NULL, c.total_experience_years, c.team_management_score "
                  "FROM candidates c WHERE c.id = %(cid)s",
                {"cid": cid, "companies": list(company_ids.values())},
            )
            row = cur.fetchone()
            counts.update(zip(count_tables, row))
            counts["functional_experience_roles"], counts["geography_experience_regions"] = row[11], row[12]
            counts["companies_touched"] = row[13]
            emb, tot, tm = row[14], row[15], row[16]
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
