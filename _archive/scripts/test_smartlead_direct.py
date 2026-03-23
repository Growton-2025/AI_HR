import os
from backend.integrations.smartlead import SmartleadBot
from dotenv import load_dotenv

load_dotenv()

# Direct test of Smartlead Bot without auth
bot = SmartleadBot()

print("🧪 Testing Smartlead Integration Directly")
print("=" * 50)

# 1. Create Campaign
campaign_name = "Test Campaign - AIML Engineer"
campaign_id = bot.create_campaign(campaign_name)

if not campaign_id:
    print("❌ Failed to create campaign. Check API key.")
    exit(1)

# 2. Configure Campaign
sender_email = os.getenv("SMARTLEAD_SENDER_EMAIL")
bot.add_email_account(sender_email)

subject = "Exciting Opportunity at AIML Engineer"
body = """Hi Nethranand,

I came across your profile and was impressed by your experience. We're currently hiring for a AIML Engineer position that I believe would be a great fit for your background.

Would you be open to a quick conversation to learn more?

Best regards,
Sydney
Recruitment Team"""

# Calculate current time + 3 minutes
from datetime import datetime, timedelta
import pytz

ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist)
send_time = now + timedelta(minutes=3)
start_hour = send_time.strftime("%H:%M")
end_hour = "23:59"

# Get today's day of week (1=Monday, 7=Sunday in Smartlead)
# Python's weekday(): 0=Monday, 6=Sunday, so we add 1
today_day = now.weekday() + 1

print(f"⏰ Scheduling for: {send_time.strftime('%Y-%m-%d %H:%M:%S IST')} (in 3 minutes)")
print(f"📅 Day of week: {today_day} ({now.strftime('%A')})")

bot.set_schedule(
    tz="Asia/Kolkata", 
    start_hour=start_hour, 
    end_hour=end_hour,
    days_of_the_week=[today_day]  # Only today's day
)
bot.update_campaign_settings(follow_up_percentage=50)

# 3. Add Lead
leads = [{
    "first_name": "Nethranand",
    "last_name": "P S",
    "email": "nethranandps2001@gmail.com"
}]

bot.add_leads(leads)

# 4. Start Campaign
bot.start_campaign()

print("\n" + "=" * 50)
print("✅ Test Complete!")
print(f"   Campaign ID: {campaign_id}")
print(f"   Recipient: nethranandps2001@gmail.com")
print("\n📧 Check your inbox for the email!")
