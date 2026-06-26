import re
with open("backend/api/routes/outreach.py", "r") as f:
    code = f.read()
match = re.search(r'async def get_linkedin_chat_history\(.*?\):(.*?)try:\n        t0 = time.time()', code, re.DOTALL)
if match:
    print(match.group(1))
