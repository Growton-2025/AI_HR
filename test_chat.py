import sys
import os
sys.path.append(os.getcwd())
import asyncio
from backend.api.routes.outreach import get_linkedin_chat_history
from backend.api import schemas

class DummyUser:
    email = "admin@example.com"
    role = "admin"

async def test():
    try:
        user = DummyUser()
        res = await get_linkedin_chat_history(role_id=0, candidate_id=2519, current_user=user)
        print("Chat history length:", len(res.get('messages', [])))
        for m in res.get('messages', []):
            print(m)
    except Exception as e:
        print("Exception:", e)

if __name__ == "__main__":
    asyncio.run(test())
