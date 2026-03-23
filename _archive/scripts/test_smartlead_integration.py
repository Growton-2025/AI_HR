import os
import sys
from datetime import datetime, timedelta, timezone as tz_module

# Add project root to path
sys.path.append(os.getcwd())

from backend.integrations.smartlead import SmartleadBot

def load_env():
    """Manually load .env file"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except Exception as e:
        print(f"⚠️ Could not load .env: {e}")

# Load env vars
load_env()

def test_integration():
    print("🚀 Starting Smartlead Integration Test...")
    
    api_key = os.getenv("SMARTLEAD_API_KEY")
    if not api_key:
        print("❌ SMARTLEAD_API_KEY not found in environment")
        return

    bot = SmartleadBot(api_key)
    
    # 1. Create Campaign
    campaign_name = f"Test Campaign - {datetime.now().strftime('%H:%M:%S')}"
    print(f"1. Creating campaign '{campaign_name}'...")
    campaign_id = bot.create_campaign(campaign_name)
    
    if not campaign_id:
        print("❌ Failed to create campaign")
        return

    # 2. Add Sender (Optional for this test, but good for completeness)
    sender = os.getenv("SMARTLEAD_SENDER_EMAIL")
    if sender:
        print(f"2. Adding sender {sender}...")
        bot.add_email_account(sender)
    
    # 3. Set Schedule with Delay
    print("3. Setting schedule with 3-minute delay...")
    start_time = datetime.now(tz_module.utc) + timedelta(minutes=3)
    print(f"   Calculated Start Time (UTC): {start_time.isoformat()}")
    
    bot.set_schedule(
        tz="Asia/Kolkata", 
        start_hour="00:00", 
        end_hour="23:59", 
        start_time=start_time
    )
    
    print("\n✅ Test Complete! Check Smartlead dashboard for the new campaign and verify the schedule.")

if __name__ == "__main__":
    test_integration()
