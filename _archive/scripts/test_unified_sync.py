
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import backend
sys.path.append(os.getcwd())

from backend.integrations.heyreach import HeyReachBot
from backend.integrations.smartlead import SmartleadBot

load_dotenv()

def test_heyreach_status():
    print("--- Testing HeyReach Status Fetch ---")
    bot = HeyReachBot()
    # Test with a known campaign and profile if possible, otherwise just test connectivity
    campaign_id = 332428 
    profile_url = "https://www.linkedin.com/in/some-profile" # Placeholder
    
    print(f"Fetching status for campaign {campaign_id}...")
    try:
        res = bot.get_lead_status(campaign_id, profile_url)
        print(f"Result: {res}")
    except Exception as e:
        print(f"Failed: {e}")

def test_smartlead_status():
    print("\n--- Testing Smartlead Status Fetch ---")
    bot = SmartleadBot()
    email = "test@example.com"
    bot.campaign_id = 12345
    
    print(f"Fetching activity for {email}...")
    try:
        # This will likely fail with 401/404 if IDs are invalid but tests the code path
        res = bot.get_lead_activity(email)
        print(f"Result: {res}")
    except Exception as e:
        print(f"Code path test: {e}")

if __name__ == "__main__":
    test_heyreach_status()
    test_smartlead_status()
