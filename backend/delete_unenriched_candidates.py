"""
Delete all unenriched uploaded candidates from the Talent Pool.

Strategy:
  - Targets candidates with pool_source IN ('recruiter_upload', 'catalog_from_upload')
    whose total_experience_years IS NULL (enrichment never wrote a value).
  - Cascades deletes to: recruitment_role_candidates, candidate_uploads linkage.
  - Masters (owner_user_id IS NULL / pool_source = 'legacy_master') are NEVER touched.

Usage:
  python -m backend.delete_unenriched_candidates            # dry run (safe, prints counts)
  python -m backend.delete_unenriched_candidates --execute  # actually deletes
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_PARAMS = dict(
    dbname=os.getenv("DB_NAME", "growton"),
    user=os.getenv("DB_USER", "growton"),
    password=os.getenv("DB_PASSWORD", "Postgres-2026"),
    host=os.getenv("DB_HOST", "growton-restore-may26.postgres.database.azure.com"),
    port=int(os.getenv("DB_PORT", "5432")),
    sslmode="require",
    connect_timeout=30,
)

EXECUTE = "--execute" in sys.argv


def run():
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = False
    cur = conn.cursor()

    print("=" * 60)
    print("  DELETE UNENRICHED UPLOADED CANDIDATES")
    print(f"  Mode: {'⚠️  EXECUTE (REAL DELETE)' if EXECUTE else '🔍 DRY RUN (no changes)'}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Count what will be affected                                       #
    # ------------------------------------------------------------------ #
    cur.execute("""
        SELECT COUNT(*) FROM candidates
        WHERE pool_source IN ('recruiter_upload', 'catalog_from_upload')
          AND (total_experience_years IS NULL OR total_experience_years = 0)
          AND COALESCE(is_archived, FALSE) = FALSE
    """)
    total_count = cur.fetchone()[0]
    print(f"\n📊 Unenriched uploaded candidates found: {total_count}")

    if total_count == 0:
        print("✅ Nothing to delete. Exiting.")
        cur.close()
        conn.close()
        return

    # ------------------------------------------------------------------ #
    # 2. Show breakdown by pool_source                                     #
    # ------------------------------------------------------------------ #
    cur.execute("""
        SELECT pool_source, COUNT(*)
        FROM candidates
        WHERE pool_source IN ('recruiter_upload', 'catalog_from_upload')
          AND (total_experience_years IS NULL OR total_experience_years = 0)
          AND COALESCE(is_archived, FALSE) = FALSE
        GROUP BY pool_source
    """)
    print("\n  Breakdown by pool_source:")
    for row in cur.fetchall():
        print(f"    {row[0]}: {row[1]} candidates")

    # ------------------------------------------------------------------ #
    # 3. Check for related rows in other tables                            #
    # ------------------------------------------------------------------ #
    cur.execute("""
        SELECT COUNT(*) FROM recruitment_role_candidates rrc
        JOIN candidates c ON c.id = rrc.candidate_id
        WHERE c.pool_source IN ('recruiter_upload', 'catalog_from_upload')
          AND (c.total_experience_years IS NULL OR c.total_experience_years = 0)
          AND COALESCE(c.is_archived, FALSE) = FALSE
    """)
    rrc_count = cur.fetchone()[0]
    print(f"\n  Related recruitment_role_candidates rows: {rrc_count}")

    # Check upload records
    cur.execute("""
        SELECT COUNT(DISTINCT source_upload_id)
        FROM candidates
        WHERE pool_source IN ('recruiter_upload', 'catalog_from_upload')
          AND (total_experience_years IS NULL OR total_experience_years = 0)
          AND COALESCE(is_archived, FALSE) = FALSE
          AND source_upload_id IS NOT NULL
    """)
    upload_count = cur.fetchone()[0]
    print(f"  Originating candidate_uploads records: {upload_count}")

    print()

    if not EXECUTE:
        print("🔍 DRY RUN complete. No changes made.")
        print("   Run with --execute to actually delete:\n")
        print("   python -m backend.delete_unenriched_candidates --execute\n")
        cur.close()
        conn.close()
        return

    # ------------------------------------------------------------------ #
    # 4. EXECUTE: Delete in the right order                               #
    # ------------------------------------------------------------------ #
    print("🗑️  Starting deletion...")

    try:
        # Step 1: Delete recruitment_role_candidates rows first (FK child)
        cur.execute("""
            DELETE FROM recruitment_role_candidates
            WHERE candidate_id IN (
                SELECT id FROM candidates
                WHERE pool_source IN ('recruiter_upload', 'catalog_from_upload')
                  AND (total_experience_years IS NULL OR total_experience_years = 0)
                  AND COALESCE(is_archived, FALSE) = FALSE
            )
        """)
        deleted_rrc = cur.rowcount
        print(f"   ✓ Deleted {deleted_rrc} recruitment_role_candidates rows")

        # Step 2: Delete the candidates themselves
        cur.execute("""
            DELETE FROM candidates
            WHERE pool_source IN ('recruiter_upload', 'catalog_from_upload')
              AND (total_experience_years IS NULL OR total_experience_years = 0)
              AND COALESCE(is_archived, FALSE) = FALSE
        """)
        deleted_cands = cur.rowcount
        print(f"   ✓ Deleted {deleted_cands} candidate rows")

        # Step 3: Clean up upload records that now have zero candidates
        cur.execute("""
            DELETE FROM candidate_uploads
            WHERE id NOT IN (
                SELECT DISTINCT UNNEST(source_upload_ids)
                FROM candidates
                WHERE source_upload_ids IS NOT NULL
                  AND array_length(source_upload_ids, 1) > 0
            )
              AND id NOT IN (
                SELECT DISTINCT source_upload_id
                FROM candidates
                WHERE source_upload_id IS NOT NULL
            )
        """)
        deleted_uploads = cur.rowcount
        print(f"   ✓ Cleaned {deleted_uploads} orphaned candidate_uploads records")

        conn.commit()
        print(f"\n✅ Done! {deleted_cands} unenriched candidates removed successfully.")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error during deletion, rolled back: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run()
