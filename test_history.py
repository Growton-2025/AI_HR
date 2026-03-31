import sys
import os
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from backend.integrations.heyreach import HeyReachBot

bot = HeyReachBot()
try:
    history = bot.get_li_chat_history(
        profile_url="https://www.linkedin.com/in/nethranand-p-s-b3b41b218/",
        campaign_id=379659
    )
    print("History length:", len(history))
    for m in history:
        print(m)
except Exception as e:
    print("Exception:", e)
