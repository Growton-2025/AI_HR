import logging
import os
import asyncio
from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import Response
from backend.integrations import plivo_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/dial")
async def plivo_dial(request: Request):
    form_data = await request.form()
    to_number = form_data.get("To")
    from_uri = form_data.get("From")
    call_uuid = form_data.get("CallUUID")
    
    logger.info(f"Dialing from endpoint to: {to_number}, From: {from_uri}, CallUUID: {call_uuid}")
    
    username = from_uri.split(":")[1].split("@")[0] if from_uri and ":" in from_uri and "@" in from_uri else None
    if username and call_uuid:
        plivo_service.last_calls[username] = call_uuid
        plivo_service.latest_call_uuid = call_uuid
        logger.info(f"Mapped {username} and latest call to {call_uuid}")
        
    to_number = plivo_service.normalize_number(to_number)
    logger.info(f"Normalized number: {to_number}")
    
    ngrok_url = plivo_service.get_ngrok_url()
    action_url = f"{ngrok_url}/api/plivo/recording" if ngrok_url else ""
    
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Record action="{action_url}" startOnDialAnswer="true" redirect="false" fileFormat="mp3" />
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
        background_tasks.add_task(plivo_service.process_call_insights, call_uuid, recording_url)
        
    return Response(status_code=200)

@router.get("/credentials")
async def get_credentials():
    return {
        "username": plivo_service.endpoint_username,
        "password": plivo_service.endpoint_password
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
