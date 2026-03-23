import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8001"

def trigger_sync():
    print("Triggering sync for role 44...")
    # Need token
    token = os.getenv("TEST_USER_TOKEN")
    if not token:
        # Try to get it from local login if possible, or just use a mock request if we can skip auth
        # But our backend needs auth. I'll look for a token in the logs or try to get one.
        print("⚠️ TEST_USER_TOKEN not found.")
        return

    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.post(f"{BASE_URL}/api/outreach/sync-responses/44", headers=headers)
        print(f"Sync result: {res.status_code}")
        print(res.json())
    except Exception as e:
        print(f"Error triggering sync: {e}")

if __name__ == "__main__":
    trigger_sync()
