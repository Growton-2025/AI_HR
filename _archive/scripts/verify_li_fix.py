
import sys
import os
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.getcwd())

from backend.integrations.heyreach import HeyReachBot

load_dotenv()

def test_fix():
    bot = HeyReachBot()
    # Using the profile discovered in the previous diagnostic
    profile_url = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"
    message = "Testing the LinkedIn Reply Fix from the AI HR Platform."
    
    print(f"Testing fix for profile: {profile_url}")
    success = bot.send_li_message(profile_url, message)
    
    if success:
        print("✅ SUCCESS: LinkedIn message sent using the new logic!")
    else:
        print("❌ FAILURE: LinkedIn message could not be sent.")

if __name__ == "__main__":
    test_fix()
