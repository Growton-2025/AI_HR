
import os
import requests
import logging

logger = logging.getLogger(__name__)

# --- CONFIG (loaded from environment variables) ---
NGROK_URL = os.getenv("NGROK_URL", "https://14af-115-245-215-165.ngrok-free.app")
CLAY_URL = os.getenv("CLAY_URL", "https://api.clay.com/v3/sources/webhook/pull-in-data-from-a-webhook-5232297e-b5f6-4eed-a2f1-79fd3dbbc652")
CLAY_AUTH = os.getenv("CLAY_AUTH", "ed3167e26e52ac14c377")

def trigger_clay(first_name: str, last_name: str, linkedin_url: str) -> bool:
    """
    Sends data to Clay - exact same logic as user's script.
    """
    payload = {
        "first_name": first_name,
        "last_name": last_name,
        "linkedin_url": linkedin_url,
        "callback_url": f"{NGROK_URL}/results"
    }
    headers = {
        "Content-Type": "application/json",
        "x-clay-webhook-auth": CLAY_AUTH
    }
    
    logger.info(f"🚀 Sending {first_name} to Clay Waterfall...")
    try:
        resp = requests.post(CLAY_URL, json=payload, headers=headers, timeout=10)
        logger.info(f"Clay Webhook Status: {resp.text}")
        return resp.ok
    except Exception as e:
        logger.error(f"Clay trigger failed: {e}")
        return False
