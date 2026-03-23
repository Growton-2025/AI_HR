
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from backend.integrations.heyreach import HeyReachBot

load_dotenv()

def test_find_campaign():
    bot = HeyReachBot()
    # Test with a name that is likely to exist or a common keyword
    test_name = "AI HR Developer" # Example name
    
    print(f"Searching for campaign: '{test_name}'...")
    campaign_id = bot.find_campaign_by_name(test_name)
    
    if campaign_id:
        print(f"✅ Found Campaign ID: {campaign_id}")
    else:
        print(f"❌ Campaign not found by name. Checking all campaigns...")
        # Let's see what's available
        import requests
        url = "https://api.heyreach.io/api/public/campaign/GetAll"
        headers = {"X-API-KEY": bot.api_key, "Content-Type": "application/json"}
        res = requests.post(url, headers=headers, json={"offset":0, "limit": 5})
        if res.status_code == 200:
            items = res.json().get('items', [])
            print("Available campaigns:")
            for item in items:
                print(f"- {item.get('name')} (ID: {item.get('id')})")

if __name__ == "__main__":
    test_find_campaign()
