import re

with open("backend/api/routes/outreach.py", "r") as f:
    code = f.read()

# Just print the first 20 lines of get_linkedin_chat_history to see how it starts
match = re.search(r'async def get_linkedin_chat_history\(.*?\):', code, re.DOTALL)
if match:
    idx = match.end()
    print(code[idx:idx+500])
