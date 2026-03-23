
import sys
import os
import json
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.getcwd())

from backend.integrations.heyreach import HeyReachBot

load_dotenv()

def test_sync_fix():
    bot = HeyReachBot()
    profile_url = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"
    
    print(f"Testing sync for profile: {profile_url}")
    activity = bot.get_lead_activity(profile_url)
    
    if activity:
        print("✅ Activity fetched successfully.")
        print(json.dumps(activity, indent=2))
        
        if activity.get('is_replied'):
            print("✅ SUCCESS: Candidate reply correctly detected!")
        else:
            print("❌ FAILURE: Candidate reply still not detected.")
            
        if activity.get('reply_text') == "How are you?":
            print("✅ SUCCESS: Correct reply text captured!")
    else:
        print("❌ FAILURE: Could not fetch activity.")

if __name__ == "__main__":
    test_sync_fix()
