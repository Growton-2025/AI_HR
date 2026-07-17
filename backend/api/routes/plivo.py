import logging
import os
import asyncio
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import Response
from backend.integrations import plivo_service

router = APIRouter()
logger = logging.getLogger(__name__)


def _parse_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _recording_callback_is_final(form_data) -> bool:
    duration = _parse_int(form_data.get("RecordingDuration"))
    duration_ms = _parse_int(form_data.get("RecordingDurationMs"))
    end_ms = _parse_int(form_data.get("RecordingEndMs"))

    if duration is not None and duration < 0:
        return False
    if duration_ms is not None and duration_ms < 0:
        return False
    if end_ms is not None and end_ms < 0:
        return False

    return duration is not None or duration_ms is not None or end_ms is not None


def _extract_duration_seconds(form_data):
    """Return the authoritative recording duration in whole seconds.

    Plivo sends ``RecordingDuration`` in seconds; when absent we fall back to
    ``RecordingDurationMs`` (milliseconds). Returns ``None`` when neither is a
    usable positive value.
    """
    duration = _parse_int(form_data.get("RecordingDuration"))
    if duration is not None and duration > 0:
        return duration

    duration_ms = _parse_int(form_data.get("RecordingDurationMs"))
    if duration_ms is not None and duration_ms > 0:
        return max(1, round(duration_ms / 1000))

    return None


@router.post("/dial")
async def plivo_dial(request: Request):
    form_data = await request.form()
    to_number = form_data.get("To")
    from_uri = form_data.get("From")
    call_uuid = form_data.get("CallUUID")
    
    logger.info(f"Dialing from endpoint to: {to_number}, From: {from_uri}, CallUUID: {call_uuid}")
    
    username = from_uri.split(":")[1].split("@")[0] if from_uri and ":" in from_uri and "@" in from_uri else None
    if username and call_uuid:
        plivo_service.record_browser_dial(username, call_uuid, to_number)
        logger.info(f"Mapped {username} and latest call to {call_uuid}")
        
    to_number = plivo_service.normalize_number(to_number)
    logger.info(f"Normalized number: {to_number}")
    
    ngrok_url = plivo_service.get_ngrok_url()
    action_url = f"{ngrok_url}/api/plivo/recording" if ngrok_url else ""
    
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Record action="{action_url}" startOnDialAnswer="true" redirect="false" fileFormat="mp3" maxLength="14400" />
        <Dial callerId="{plivo_service.PLIVO_NUMBER}">
            <Number>{to_number}</Number>
        </Dial>
    </Response>
    """
    return Response(content=xml_response, media_type="application/xml")

@router.post("/recording")
async def plivo_recording(request: Request, background_tasks: BackgroundTasks):
    form_data = await request.form()
    logger.info(f"Raw form data from Plivo: {form_data}")
    
    call_uuid = form_data.get("CallUUID")
    recording_url = form_data.get("RecordingUrl") or form_data.get("RecordUrl")
    
    logger.info(f"Received recording callback. CallUUID: {call_uuid}, RecordingUrl: {recording_url}")
    
    if call_uuid and recording_url:
        plivo_service.recordings[call_uuid] = recording_url
        logger.info(f"Stored recording for {call_uuid}")
        duration_seconds = _extract_duration_seconds(form_data)
        logger.info(f"Provider recording duration for {call_uuid}: {duration_seconds}s")
        if _recording_callback_is_final(form_data):
            background_tasks.add_task(
                plivo_service.process_call_insights,
                call_uuid,
                recording_url,
                0,
                duration_seconds,
            )
        else:
            logger.info(
                "Recording callback for %s is not final yet; delaying processing until Plivo media is likely ready.",
                call_uuid,
            )
            background_tasks.add_task(
                plivo_service.process_call_insights,
                call_uuid,
                recording_url,
                30,
                duration_seconds,
            )
        
    return Response(status_code=200)

@router.post("/test-answer")
@router.get("/test-answer")
async def plivo_test_answer():
    """Diagnostic answer_url for server-initiated test calls: speaks a short
    message and hangs up. Lets us verify Plivo can reach this backend through
    the tunnel and measure dial->answer latency via CDRs, without the browser."""
    logger.info("[TestCall] Plivo fetched /test-answer through the tunnel")
    xml_response = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Speak>This is a Hayasa test call verifying the calling pipeline. Goodbye.</Speak>
    </Response>
    """
    return Response(content=xml_response, media_type="application/xml")


@router.post("/client-timing")
async def client_timing(request: Request):
    """Diagnostic beacon: the frontend reports per-leg call-setup timings here
    so they land in the server log with timestamps (browser console is often
    unavailable when debugging recruiter machines)."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    logger.info(
        "[ClientTiming] leg=%s ms=%s detail=%s",
        payload.get("leg"), payload.get("ms"), payload.get("detail", ""),
    )
    return Response(status_code=204)


@router.get("/credentials")
async def get_credentials():
    result = await plivo_service.setup_plivo()
    if not result.get("success"):
        raise HTTPException(
            status_code=503,
            detail={
                "code": result.get("code") or "plivo_unavailable",
                "message": result.get("error") or "Unable to prepare Plivo softphone credentials",
                "metadata": {
                    "public_url": plivo_service.endpoint_public_url or None,
                },
            },
        )
    return {
        "username": plivo_service.endpoint_username,
        "password": plivo_service.endpoint_password,
        "public_url": plivo_service.endpoint_public_url or None,
    }

@router.get("/credentials/refresh")
async def refresh_credentials():
    """Force-regenerate Plivo app + endpoint credentials. Call this after changing NGROK_URL."""
    result = await plivo_service.setup_plivo(force=True)
    if not result.get("success"):
        raise HTTPException(
            status_code=503,
            detail={
                "code": result.get("code") or "plivo_unavailable",
                "message": result.get("error") or "Unable to prepare Plivo softphone credentials",
                "metadata": {
                    "public_url": plivo_service.endpoint_public_url or None,
                },
            },
        )
    return {
        "username": plivo_service.endpoint_username,
        "password": plivo_service.endpoint_password,
        "public_url": plivo_service.endpoint_public_url or None,
        "refreshed": True,
    }

@router.get("/recording/{call_uuid}")
async def get_recording(call_uuid: str):
    url = plivo_service.recordings.get(call_uuid)
    return {"recording_url": url}

@router.get("/insights/{call_uuid}")
async def get_insights(call_uuid: str):
    if call_uuid in plivo_service.call_insights:
        return {"insights": plivo_service.call_insights[call_uuid]}
        
    if call_uuid in plivo_service.recordings:
        await plivo_service.process_call_insights(call_uuid, plivo_service.recordings[call_uuid])
        return {"insights": plivo_service.call_insights.get(call_uuid)}
        
    return {"insights": None}

@router.get("/last-call-uuid/{username}")
async def get_last_call_uuid(username: str):
    uuid = plivo_service.last_calls.get(username)
    return {"call_uuid": uuid}

@router.get("/call-state/{username}")
async def get_call_state(username: str):
    state = plivo_service.last_call_states.get(username) or {}
    return {
        "call_uuid": state.get("call_uuid"),
        "username": state.get("username") or username,
        "to_number": state.get("to_number"),
        "seen_at": state.get("seen_at"),
    }

@router.get("/test-dummy")
async def test_dummy():
    dummy_uuid = "test-call-uuid-12345"
    dummy_url = "https://aps1.media.plivo.com/v1/Account/MAZTQ2ZTEWMGMXZDU0ZG/Recording/a49894f4-3f72-4f8d-867d-6f90e8b806d0.mp3"
    
    plivo_service.recordings[dummy_uuid] = dummy_url
    for user_key in plivo_service.last_calls.keys():
        plivo_service.last_calls[user_key] = dummy_uuid
    plivo_service.last_calls["default"] = dummy_uuid
    
    asyncio.create_task(plivo_service.process_call_insights(dummy_uuid, dummy_url))
    return {"message": "Dummy processing triggered successfully", "uuid": dummy_uuid}
