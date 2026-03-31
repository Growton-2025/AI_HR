import requests
import json
try:
    res = requests.get('http://localhost:3002/api/outreach/chat/linkedin/0/2519', timeout=30)
    print("STATUS:", res.status_code)
    data = res.json()
    msgs = data.get("messages", [])
    print("MESSAGES COUNT:", len(msgs))
    if msgs:
        print("FIRST MESSAGE:", json.dumps(msgs[0]))
except Exception as e:
    print("ERROR:", e)
