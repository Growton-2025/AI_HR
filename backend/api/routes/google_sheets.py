"""Google Sheet import: connect Google, preview a tab's mapping, import, and re-sync.

One set of routes serves both Talent Pool and Role imports; a role import passes
``role_id`` in the body. Progress is polled on the existing
``GET /api/candidates/uploads/{upload_id}``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from backend.api import deps, schemas
from backend.api.routes import candidate_imports as ci
from backend.api.routes.roles import _is_admin, _resolve_role
from backend.db.connection import get_db_connection, return_db_connection
from backend.services import google_oauth, google_sheet_source as sheets, sheet_import
from backend.services.google_oauth import GoogleAuthError
from backend.services.google_sheet_source import SheetSourceError

router = APIRouter()
logger = logging.getLogger(__name__)


class GoogleConnectBody(BaseModel):
    code: str


class SheetRefBody(BaseModel):
    sheet_url: str
    tab: Optional[str] = None
    role_id: Optional[int] = None


class SheetPreviewBody(SheetRefBody):
    use_llm: bool = False


class SheetCommitBody(SheetRefBody):
    mapping: Dict[str, Any] = Field(default_factory=dict)
    enrich_linkedin: bool = True
    enrichment_mode: str = "none"


# ---------------------------------------------------------------------- helpers
def _auth_http_error(exc: GoogleAuthError) -> HTTPException:
    status = {"not_configured": 503, "not_connected": 409, "reconnect": 409}.get(exc.code, 400)
    return HTTPException(status_code=status, detail={"message": str(exc), "code": exc.code})


def _sheet_http_error(exc: SheetSourceError) -> HTTPException:
    status = {"reconnect": 409, "no_access": 403, "not_found": 404, "api_disabled": 503}.get(exc.code, 400)
    return HTTPException(status_code=status, detail={"message": str(exc), "code": exc.code})


def _resolve_target(current_user: schemas.User, role_id: Optional[int]) -> Dict[str, Any]:
    """Who owns the imported candidates, and which role (if any) they join.

    Pool imports are recruiter-only, like spreadsheet uploads. For a role, admins may
    target any role and the candidates belong to that role's recruiter."""
    if role_id is None:
        if _is_admin(current_user):
            raise HTTPException(status_code=403, detail="Admins import through a role")
        return {"owner_user_id": int(current_user.id), "role_id": None, "role_name": None}

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_id=role_id)
    finally:
        return_db_connection(conn)
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    owner = int(role[2]) if _is_admin(current_user) and role[2] else int(current_user.id)
    return {"owner_user_id": owner, "role_id": int(role[0]), "role_name": role[1]}


def _load_tab(user_id: int, sheet_url: str, tab: Optional[str]) -> Dict[str, Any]:
    """Blocking: resolve the link and tab and read it with the user's Google token."""
    spreadsheet_id, gid = sheets.parse_sheet_url(sheet_url)
    token = google_oauth.access_token_for_user(user_id)
    meta = sheets.list_tabs(token, spreadsheet_id)
    chosen = sheets.resolve_tab(meta["tabs"], tab_title=tab, gid=gid)
    frame = sheets.read_tab(token, spreadsheet_id, chosen["title"])
    return {"spreadsheet_id": spreadsheet_id, "sheet_title": meta["title"], "tab": chosen,
            "tabs": meta["tabs"], "frame": frame}


async def _load_tab_async(user_id: int, sheet_url: str, tab: Optional[str]) -> Dict[str, Any]:
    try:
        return await run_in_threadpool(_load_tab, user_id, sheet_url, tab)
    except GoogleAuthError as exc:
        raise _auth_http_error(exc)
    except SheetSourceError as exc:
        raise _sheet_http_error(exc)


# ------------------------------------------------------------- Google connection
@router.get("/integrations/google/status")
def google_status(current_user: schemas.User = Depends(deps.get_current_user)):
    return {**google_oauth.status(int(current_user.id)), "scope": google_oauth.REQUESTED_SCOPES}


@router.post("/integrations/google/connect")
def google_connect(body: GoogleConnectBody, current_user: schemas.User = Depends(deps.get_current_user)):
    try:
        return google_oauth.connect(int(current_user.id), body.code)
    except GoogleAuthError as exc:
        raise _auth_http_error(exc)


@router.delete("/integrations/google")
def google_disconnect(current_user: schemas.User = Depends(deps.get_current_user)):
    try:
        google_oauth.disconnect(int(current_user.id))
    except GoogleAuthError as exc:
        raise _auth_http_error(exc)
    return {"connected": False}


# ------------------------------------------------------------------ sheet import
@router.post("/import/gsheet/tabs")
def gsheet_tabs(body: SheetRefBody, current_user: schemas.User = Depends(deps.get_current_user)):
    """Tabs of the linked sheet, with the one the link points at marked as selected."""
    try:
        spreadsheet_id, gid = sheets.parse_sheet_url(body.sheet_url)
        token = google_oauth.access_token_for_user(int(current_user.id))
        meta = sheets.list_tabs(token, spreadsheet_id)
        chosen = sheets.resolve_tab(meta["tabs"], gid=gid)
    except GoogleAuthError as exc:
        raise _auth_http_error(exc)
    except SheetSourceError as exc:
        raise _sheet_http_error(exc)
    return {"spreadsheet_id": spreadsheet_id, "sheet_title": meta["title"], "tabs": meta["tabs"],
            "selected_tab": chosen["title"], "max_rows": sheets.max_rows()}


@router.post("/import/gsheet/preview")
async def gsheet_preview(body: SheetPreviewBody, current_user: schemas.User = Depends(deps.get_current_user)):
    """Same shape as the spreadsheet upload preview, plus where the LinkedIn URLs are."""
    _resolve_target(current_user, body.role_id)
    loaded = await _load_tab_async(int(current_user.id), body.sheet_url, body.tab)
    frame: pd.DataFrame = loaded["frame"]
    preview = await ci.build_preview_from_frame(
        frame, filename=f"{loaded['sheet_title']} / {loaded['tab']['title']}", use_llm=body.use_llm)
    counts = sheet_import.linkedin_url_counts(frame)
    # The column that holds the profile URLs is the LinkedIn column, whatever the headers
    # say; a header that only mentions LinkedIn gives the field up.
    best = max(counts, key=counts.get) if counts else None
    if best and preview["suggested_mapping"].get(best) != "linkedin":
        for header, target in list(preview["suggested_mapping"].items()):
            if target == "linkedin":
                has_values = any(preview["mapping_details"].get(header, {}).get("sample_values") or [])
                preview["suggested_mapping"][header] = "custom" if has_values else "ignore"
                preview["mapping_details"].setdefault(header, {})["target"] = preview["suggested_mapping"][header]
        preview["suggested_mapping"][best] = "linkedin"
        detail = preview["mapping_details"].get(best) or {}
        detail.update({"target": "linkedin", "source": "alias", "confidence": 0.95,
                       "reason": "Column contains LinkedIn profile URLs", "category": "Required Fields"})
        preview["mapping_details"][best] = detail
        preview["missing_required"] = [t for t in preview["missing_required"] if t != "linkedin"]
    return {**preview, "spreadsheet_id": loaded["spreadsheet_id"], "sheet_title": loaded["sheet_title"],
            "tab": loaded["tab"]["title"], "linkedin_url_counts": counts,
            "linkedin_required_targets": ["linkedin"]}


def _start(*, current_user: schemas.User, target: Dict[str, Any], loaded: Dict[str, Any],
           mapping: Dict[str, str], enrich_linkedin: bool, enrichment_mode: str) -> Dict[str, Any]:
    connection_id = sheet_import.save_connection(
        owner_user_id=target["owner_user_id"], connected_by_user_id=int(current_user.id),
        role_id=target["role_id"], spreadsheet_id=loaded["spreadsheet_id"], tab_title=loaded["tab"]["title"],
        sheet_title=loaded["sheet_title"], mapping=mapping, enrich_linkedin=enrich_linkedin,
        enrichment_mode=enrichment_mode,
    )
    result = sheet_import.start_sheet_import(
        df=loaded["frame"], mapping=mapping, owner_user_id=target["owner_user_id"],
        user_role=(current_user.role or "").strip().lower(), role_id=target["role_id"],
        enrich_linkedin=enrich_linkedin, enrichment_mode=enrichment_mode,
        spreadsheet_id=loaded["spreadsheet_id"], tab_title=loaded["tab"]["title"],
        sheet_title=loaded["sheet_title"], connection_id=connection_id,
    )
    logger.info("sheet import accepted upload_id=%s rows=%s user=%s role=%s enrich_linkedin=%s",
                result["upload_id"], result["row_count"], current_user.id, target["role_id"], enrich_linkedin)
    return {**result, "role_id": target["role_id"], "role_name": target["role_name"]}


@router.post("/import/gsheet/commit")
async def gsheet_commit(body: SheetCommitBody, current_user: schemas.User = Depends(deps.get_current_user)):
    """Start the import with the confirmed mapping and remember the connection for Sync now."""
    target = _resolve_target(current_user, body.role_id)
    try:
        mapping = sheet_import.validate_sheet_mapping(body.mapping, enrich_linkedin=body.enrich_linkedin)
    except sheet_import.SheetImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    loaded = await _load_tab_async(int(current_user.id), body.sheet_url, body.tab)
    missing_columns = [h for h in mapping if h not in set(map(str, loaded["frame"].columns))]
    if missing_columns and len(missing_columns) == len(mapping):
        raise HTTPException(status_code=400, detail="None of the mapped columns exist in this tab any more")
    try:
        return await run_in_threadpool(
            lambda: _start(current_user=current_user, target=target, loaded=loaded, mapping=mapping,
                           enrich_linkedin=body.enrich_linkedin, enrichment_mode=body.enrichment_mode))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("sheet import could not start")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/import/gsheet/connections")
def gsheet_connections(role_id: Optional[int] = Query(None),
                       current_user: schemas.User = Depends(deps.get_current_user)):
    """Saved sheets for this role, or the user's pool-level ones when no role is given."""
    return {"connections": sheet_import.list_connections(
        user_id=int(current_user.id), role_id=role_id, pool_only=role_id is None)}


@router.get("/import/gsheet/sync-status")
def gsheet_sync_status(current_user: schemas.User = Depends(deps.get_current_user)):
    """Live state of the automatic sync for the user's linked sheets (polled by the app)."""
    return {"auto_sync": sheet_import.auto_sync_seconds() > 0,
            "connections": sheet_import.sync_status(user_id=int(current_user.id))}


@router.post("/import/gsheet/connections/{connection_id}/sync")
async def gsheet_sync(connection_id: int, current_user: schemas.User = Depends(deps.get_current_user)):
    """Re-read the saved tab and import again; people already in the pool are skipped unscraped."""
    connection = sheet_import.get_connection(connection_id, user_id=int(current_user.id))
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    target = _resolve_target(current_user, connection["role_id"])
    try:
        mapping = sheet_import.validate_sheet_mapping(
            {k: v for k, v in connection["mapping"].items() if isinstance(v, str)},
            enrich_linkedin=connection["enrich_linkedin"])
    except sheet_import.SheetImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    loaded = await _load_tab_async(int(current_user.id), connection["sheet_url"], connection["tab_title"])
    linkedin_headers = [h for h, t in mapping.items() if t == "linkedin"]
    if connection["enrich_linkedin"] and not any(h in loaded["frame"].columns for h in linkedin_headers):
        raise HTTPException(
            status_code=400,
            detail="The LinkedIn column was renamed or removed in the sheet. Import the sheet again to remap it.")
    try:
        return await run_in_threadpool(
            lambda: _start(current_user=current_user, target=target, loaded=loaded, mapping=mapping,
                           enrich_linkedin=connection["enrich_linkedin"],
                           enrichment_mode=connection["enrichment_mode"]))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("sheet sync could not start")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/import/gsheet/connections/{connection_id}")
def gsheet_delete_connection(connection_id: int, current_user: schemas.User = Depends(deps.get_current_user)):
    if not sheet_import.delete_connection(connection_id, user_id=int(current_user.id)):
        raise HTTPException(status_code=404, detail="Connection not found")
    return {"deleted": True}
