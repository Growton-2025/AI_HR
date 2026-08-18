import logging
import os
import asyncio
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Depends
from fastapi.responses import Response
from backend.integrations import plivo_service
from backend.api import schemas, deps
# Inbound persistence lives with the rest of the call storage; imported lazily
# inside the handlers would re-import per webhook, so bind it once here.
from backend.api.routes import calls as calls_module

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


def _extract_dial_token(form_data) -> str:
    """Pull the per-attempt dial token out of the answer-URL payload.

    Plivo documents that browser-SDK ``X-PH-*`` headers reach the answer URL,
    but not the exact parameter name it gives them — it may arrive as
    ``X-PH-DialToken``, lowercased, or with the prefix stripped. Rather than
    guess one spelling and silently fall back to the buggy username matching
    forever, match any key that ends in ``dialtoken`` once punctuation is
    removed. The ``[DialTokenProbe]`` log records the real name so this can be
    tightened to an exact lookup once observed in production.
    """
    for key, value in form_data.items():
        normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
        if normalized.endswith("dialtoken") and value:
            return str(value).strip()
    return ""


@router.post("/dial")
async def plivo_dial(request: Request):
    form_data = await request.form()
    to_number = form_data.get("To")
    from_uri = form_data.get("From")
    call_uuid = form_data.get("CallUUID")
    
    logger.info(f"Dialing from endpoint to: {to_number}, From: {from_uri}, CallUUID: {call_uuid}")

    # Probe for docs/call-attribution-plan.md Phase 3: does Plivo actually
    # forward the browser SDK's X-PH-* headers to this answer URL? Phase 3
    # replaces "most recently updated calls row for this SIP username" (which
    # misattributes recordings across concurrent recruiters) with an exact
    # per-attempt token, and that only works if these arrive. Log both the
    # header we care about and every X-PH-ish key present, since Plivo's exact
    # parameter naming for forwarded headers is what we need to confirm.
    forwarded = {k: v for k, v in form_data.items() if "dialtoken" in k.lower() or k.lower().startswith("x-ph")}
    if forwarded:
        logger.info("[DialTokenProbe] Plivo forwarded custom headers: %s", forwarded)
    else:
        logger.info(
            "[DialTokenProbe] No X-PH header in the dial webhook. Payload keys: %s",
            sorted(form_data.keys()),
        )

    username = from_uri.split(":")[1].split("@")[0] if from_uri and ":" in from_uri and "@" in from_uri else None
    dial_token = _extract_dial_token(form_data)
    if username and call_uuid:
        plivo_service.record_browser_dial(username, call_uuid, to_number, dial_token)
        logger.info(f"Mapped {username} and latest call to {call_uuid}")
        
    to_number = plivo_service.normalize_number(to_number)
    logger.info(f"Normalized number: {to_number}")
    
    ngrok_url = plivo_service.get_ngrok_url()
    action_url = f"{ngrok_url}/api/plivo/recording" if ngrok_url else ""
    
    # `action` only ever gets Plivo's interim callback for a startOnDialAnswer
    # recording (RecordingDuration=-1, fired the instant recording *starts*).
    # The real, final callback — with the actual duration and a fully-written
    # file — only ever arrives at `callbackUrl`, which was previously unset.
    # Without it, process_call_insights had no authoritative "recording is
    # done" signal and fell back to a flat 30s guess-delay before downloading,
    # which truncated the audio/transcript for any call longer than that.
    # Both point at the same endpoint on purpose: plivo_recording() already
    # branches correctly on payload content (_recording_callback_is_final),
    # so no new route is needed — Plivo will now actually invoke it twice
    # (interim, then final) instead of once.
    #
    # `timeout` (silence-detection cutoff) also previously defaulted to
    # Plivo's built-in 15s — any natural pause in conversation ≥15s
    # permanently stopped the recording for the rest of the call (redirect is
    # false, so it never re-arms). Set generously high so normal pauses in a
    # screening call never trigger it.
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Record action="{action_url}" callbackUrl="{action_url}" callbackMethod="POST" startOnDialAnswer="true" redirect="false" fileFormat="mp3" maxLength="14400" timeout="300" />
        <Dial callerId="{plivo_service.PLIVO_NUMBER}">
            <Number>{to_number}</Number>
        </Dial>
    </Response>
    """
    return Response(content=xml_response, media_type="application/xml")

# ── Inbound calls ────────────────────────────────────────────────────────────
# Plivo carries no bearer token, so these are unauthenticated by necessity.
# They are also retried and may be delivered more than once, so every write is
# keyed on CallUUID.

def _voicemail_xml() -> str:
    """Short greeting + record, used when nobody picks up."""
    ngrok_url = plivo_service.get_ngrok_url() or ""
    action = f"{ngrok_url}/api/plivo/incoming-voicemail" if ngrok_url else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Speak>Thanks for calling. Our team is unavailable right now. Please leave a short message after the beep and we will call you back.</Speak>
        <Record action="{action}" method="POST" maxLength="120" finishOnKey="#" playBeep="true" redirect="false" />
    </Response>
    """


@router.post("/incoming")
async def plivo_incoming(request: Request):
    """Answer URL for the inbound number: ring every registered recruiter at once."""
    form = await request.form()
    call_uuid = form.get("CallUUID")
    from_number = form.get("From") or ""
    to_number = form.get("To") or ""
    logger.info("Inbound call %s from %s to %s", call_uuid, from_number, to_number)

    await asyncio.to_thread(
        calls_module.record_inbound_call,
        call_uuid, from_number, to_number, form.get("CallStatus"),
    )

    usernames = await asyncio.to_thread(plivo_service.get_registered_endpoint_usernames)
    if not usernames:
        logger.info("No registered softphones for inbound %s — going to voicemail.", call_uuid)
        return Response(content=_voicemail_xml(), media_type="application/xml")

    ngrok_url = plivo_service.get_ngrok_url() or ""
    action = f"{ngrok_url}/api/plivo/incoming-unanswered" if ngrok_url else ""
    # Multiple <User> entries ring simultaneously; the first to answer wins and
    # Plivo cancels the rest.
    users = "".join(f"<User>sip:{u}@phone.plivo.com</User>" for u in usernames)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Dial timeout="25" action="{action}" method="POST" redirect="false" callerId="{from_number}">
            {users}
        </Dial>
    </Response>
    """
    return Response(content=xml, media_type="application/xml")


@router.post("/incoming-unanswered")
async def plivo_incoming_unanswered(request: Request):
    """`action` URL of the ring-all <Dial>. Answered → record who took it.
    Timed out / no answer → fall through to a short voicemail."""
    form = await request.form()
    call_uuid = form.get("CallUUID")
    dial_status = (form.get("DialStatus") or "").lower()
    b_leg = form.get("DialBLegUUID") or ""

    await asyncio.to_thread(
        calls_module.record_inbound_dial_result, call_uuid, dial_status, b_leg,
    )

    if dial_status == "completed" and b_leg:
        return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response/>',
                        media_type="application/xml")
    return Response(content=_voicemail_xml(), media_type="application/xml")


@router.post("/incoming-voicemail")
async def plivo_incoming_voicemail(request: Request):
    form = await request.form()
    await asyncio.to_thread(
        calls_module.attach_inbound_recording,
        form.get("CallUUID"), form.get("RecordUrl") or form.get("RecordingUrl"),
    )
    return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response/>',
                    media_type="application/xml")


@router.post("/incoming-hangup")
async def plivo_incoming_hangup(request: Request):
    form = await request.form()
    await asyncio.to_thread(
        calls_module.finalize_inbound_call,
        form.get("CallUUID"), form.get("CallStatus"),
        form.get("HangupCause"), form.get("Duration"),
    )
    return Response(status_code=200)


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
            # Put the URL on the row NOW, before any of the slow work. The
            # player is gated on calls.recording_url, and that used to be
            # written only once transcription and summarising had finished — so
            # the recruiter watched "Polling for recording stream from
            # Plivo..." for up to three minutes while we had been holding the
            # link since the call ended. It has to be the row rather than the
            # in-memory map, too: that map is per gunicorn worker, and the
            # recruiter's next poll may be served by a different one.
            #
            # Only on the FINAL callback: the interim one points at a partial
            # file, and offering that as "the recording" would hand the
            # recruiter a truncated call.
            await asyncio.to_thread(
                plivo_service.persist_recording_url, call_uuid, recording_url
            )
            calls_module.invalidate_calls_cache()
            background_tasks.add_task(
                plivo_service.process_call_insights,
                call_uuid,
                recording_url,
                0,
                duration_seconds,
            )
        else:
            # This is the interim callback (fired the instant a
            # startOnDialAnswer recording *starts*, with -1 durations) — the
            # file at recording_url is only a partial recording of a call
            # still in progress. We now have `callbackUrl` configured to
            # deliver the genuine final callback once Plivo confirms the file
            # is complete, so there's no need to guess a delay and download
            # early: doing so used to race the final callback and win (since
            # process_call_insights skips reprocessing once call_insights has
            # a result), permanently locking in the truncated recording and
            # transcript for any call that outlasted the guess. Just record
            # that we've seen this call_uuid and wait for the real signal —
            # the manual "Sync Recording" button covers the rare case where
            # the final callback never arrives (e.g. webhook delivery failure).
            logger.info(
                "Recording callback for %s is not final yet; waiting for the final callbackUrl notification instead of guessing.",
                call_uuid,
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
async def get_credentials(current_user: schemas.User = Depends(deps.get_current_user)):
    """SIP credentials for the caller's own softphone.

    Now authenticated and per-user: inbound "ring everyone" dials one <User> per
    recruiter, which needs distinct endpoints, and this route previously handed
    working SIP credentials to any unauthenticated caller.
    """
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

    # Fall back to the shared endpoint if per-user provisioning fails, so a
    # Plivo hiccup degrades to the previous behaviour instead of killing dialling.
    own = await plivo_service.ensure_endpoint_for_user(current_user.id)
    if not own:
        # Transient Plivo 5xx is the common failure here and a single retry
        # usually clears it, which is far preferable to the shared endpoint.
        await asyncio.sleep(0.75)
        own = await plivo_service.ensure_endpoint_for_user(current_user.id)

    if not own and not plivo_service.claim_shared_endpoint(current_user.id):
        # Someone else already holds the shared endpoint. Sharing it would give
        # two recruiters one SIP username, and attribution would start writing
        # one person's recording onto the other's candidate. A blocked dial is
        # recoverable; two candidates' mixed transcripts are not.
        logger.error(
            "Refusing to share the Plivo endpoint: user %s cannot be provisioned "
            "and user %s already holds the shared endpoint.",
            current_user.id, plivo_service.get_shared_endpoint_holder(),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "plivo_endpoint_unavailable",
                "message": (
                    "Your calling line could not be set up, and the backup line is "
                    "already in use by another recruiter. Please retry in a moment — "
                    "calling now would attach recordings to the wrong candidate."
                ),
                "metadata": {"public_url": plivo_service.endpoint_public_url or None},
            },
        )

    if not own:
        # ERROR, not warning: if a second recruiter also lands here they share
        # one SIP username, and call attribution (which matches the most
        # recently updated `calls` row for that username) starts writing one
        # recruiter's recording and transcript onto the other's candidate. That
        # is silent data corruption, so it must not look routine in the logs.
        # See docs/call-attribution-plan.md.
        logger.error(
            "DEGRADED: user %s fell back to the shared Plivo endpoint. Inbound "
            "ring-all will skip them, and if another recruiter is also on the "
            "shared endpoint, call recordings may attach to the wrong candidate.",
            current_user.id,
        )
        return {
            "username": plivo_service.endpoint_username,
            "password": plivo_service.endpoint_password,
            "public_url": plivo_service.endpoint_public_url or None,
            "degraded": True,
            "degraded_reason": (
                "Your softphone could not be provisioned and is running on a shared line. "
                "Call recordings may be attached to the wrong candidate."
            ),
        }

    # Provisioning recovered (or never failed) — hand the fallback back so the
    # next recruiter who needs it is not locked out until the claim expires.
    plivo_service.release_shared_endpoint(current_user.id)
    return {
        "username": own["username"],
        "password": own["password"],
        "public_url": plivo_service.endpoint_public_url or None,
        "degraded": False,
    }


@router.post("/registered")
async def mark_registered(current_user: schemas.User = Depends(deps.get_current_user)):
    """Called when the browser softphone finishes SIP registration, so inbound
    calls only ring endpoints that are plausibly online."""
    await asyncio.to_thread(plivo_service.mark_endpoint_registered, current_user.id)
    return {"success": True}

@router.post("/busy")
async def mark_busy(
    payload: dict,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Browser reports whether this recruiter is currently on a call, so the
    inbound ring-all fork can skip them. Best-effort: `get_registered_endpoint_usernames`
    ages the flag out, so a lost 'idle' beacon cannot strand an endpoint."""
    busy = bool(payload.get("busy"))
    if busy:
        await asyncio.to_thread(plivo_service.mark_endpoint_busy, current_user.id)
    else:
        await asyncio.to_thread(plivo_service.clear_endpoint_busy, current_user.id)
    return {"success": True, "busy": busy}


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

@router.get("/call-state-by-token/{dial_token}")
async def get_call_state_by_token(dial_token: str):
    """Dial handshake keyed on the attempt rather than the SIP username.

    The username-keyed route below returns whatever call that endpoint placed
    most recently and is never cleared, so a redial passes the handshake
    instantly on the *previous* call's UUID — masking genuine webhook failures.
    """
    state = plivo_service.dial_token_states.get(dial_token) or {}
    return {
        "call_uuid": state.get("call_uuid"),
        "username": state.get("username"),
        "to_number": state.get("to_number"),
        "seen_at": state.get("seen_at"),
        "dial_token": dial_token,
    }


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
