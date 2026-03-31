import sys
import os
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from backend.integrations.heyreach import HeyReachBot

bot = HeyReachBot()
conv = bot._find_conversation(
    profile_url="https://www.linkedin.com/in/nethranand-p-s-b3b41b218/",
    campaign_id=379659
)
print("Found conversation ID:", conv.get('id') if conv else None)
