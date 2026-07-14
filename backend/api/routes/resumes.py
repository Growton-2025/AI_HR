"""Candidate resume endpoints: upload, metadata, file streaming, reparse, delete."""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import Response

from backend.api import deps, schemas
from backend.api.routes.browse import _authorize_candidate_update
from backend.services import resume_service
from backend.services.resume_parse import ALLOWED_EXTENSIONS

router = APIRouter()
logger = logging.getLogger(__name__)

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "application/octet-stream",  # browsers sometimes send this for txt/md; extension gate still applies
}


def _validate_upload(filename: str, content_type: str, size: int) -> None:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if content_type and content_type.split(";")[0].strip().lower() not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type '{content_type}'")
    if size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    max_bytes = resume_service.RESUME_MAX_MB * 1024 * 1024
    if size > max_bytes:
        raise HTTPException(status_code=413, detail=f"Resume exceeds {resume_service.RESUME_MAX_MB} MB limit")


@router.post("/candidates/{candidate_id}/resume", status_code=202)
async def upload_resume(
    candidate_id: int,
    file: UploadFile = File(...),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    _authorize_candidate_update(candidate_id, current_user, allow_role_access=True)
    data = await file.read()
    content_type = (file.content_type or "application/octet-stream").split(";")[0].strip()
    _validate_upload(file.filename or "", content_type, len(data))

    meta = resume_service.create_resume(
        candidate_id,
        filename=file.filename or "resume",
        content_type=content_type,
        data=data,
        uploaded_by_user_id=current_user.id,
    )
    resume_service.spawn_parse_thread(meta["id"], candidate_id)
    return {"resume": _public_meta(meta)}


@router.get("/candidates/{candidate_id}/resume")
def get_resume(
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    _authorize_candidate_update(candidate_id, current_user, allow_role_access=True)
    meta = resume_service.fetch_resume_meta(candidate_id)
    if not meta:
        raise HTTPException(status_code=404, detail="No resume on file for this candidate")
    payload = _public_meta(meta)
    if meta["parse_status"] in {"complete", "low_text", "failed"}:
        payload["extracted_text"] = resume_service.fetch_resume_text(candidate_id)
    return {"resume": payload}


@router.get("/candidates/{candidate_id}/resume/file")
def download_resume(
    candidate_id: int,
    disposition: str = Query("inline", pattern="^(inline|attachment)$"),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    _authorize_candidate_update(candidate_id, current_user, allow_role_access=True)
    result = resume_service.fetch_resume_file(candidate_id)
    if not result:
        raise HTTPException(status_code=404, detail="No resume on file for this candidate")
    meta, data = result
    safe_name = (meta["filename"] or "resume").replace('"', "")
    return Response(
        content=data,
        media_type=meta["content_type"] or "application/octet-stream",
        headers={
            "Content-Disposition": f'{disposition}; filename="{safe_name}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/candidates/{candidate_id}/resume/reparse", status_code=202)
def reparse_resume(
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    _authorize_candidate_update(candidate_id, current_user, allow_role_access=True)
    meta = resume_service.fetch_resume_meta(candidate_id)
    if not meta:
        raise HTTPException(status_code=404, detail="No resume on file for this candidate")
    resume_service.spawn_parse_thread(meta["id"], candidate_id)
    payload = _public_meta(meta)
    payload["parse_status"] = "pending"
    return {"resume": payload}


@router.delete("/candidates/{candidate_id}/resume")
def delete_resume(
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    _authorize_candidate_update(candidate_id, current_user, allow_role_access=True)
    meta = resume_service.fetch_resume_meta(candidate_id)
    if not meta:
        raise HTTPException(status_code=404, detail="No resume on file for this candidate")
    resume_service._update_resume(meta["id"], is_current=False)
    try:
        from backend.pipeline import query

        query.refresh_profiles_in_cache([candidate_id])
    except Exception:
        logger.exception("Resume delete: cache refresh failed for candidate %s", candidate_id)
    return {"deleted": True}


def _public_meta(meta: dict) -> dict:
    """Strip storage internals from API responses."""
    return {
        key: value
        for key, value in meta.items()
        if key not in {"storage_backend", "storage_key", "uploaded_by_user_id"}
    }
