import requests
import json
import time

t0 = time.time()
try:
    res = requests.get('http://localhost:3002/api/outreach/chat/linkedin/0/2519', timeout=10)
    print("STATUS:", res.status_code)
    data = res.json()
    msgs = data.get("messages", [])
    print("MESSAGES:", len(msgs))
    if msgs:
        print("FIRST MSGS TIME:", msgs[0].get('time'))
except Exception as e:
    print("ERROR:", e)
print(f"Elapsed: {time.time()-t0:.2f}s")
