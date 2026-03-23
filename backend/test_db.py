import json

with open('db/candidates.json') as f:
    data = json.load(f)
    print("KEYS for first candidate:", data[0].keys())
    print("raw_fields keys:", json.loads(data[0].get("raw_fields", "{}")).keys())
