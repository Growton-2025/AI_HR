import json

with open('db/candidates.json') as f:
    data = json.load(f)
    if data:
        # print keys of first candidate
        print("Candidate keys:", data[0].keys())
        # print keys of first role of first candidate
        if data[0].get("roles"):
            print("Role keys:", data[0]["roles"][0].keys())
            print("Role company_details keys:", data[0]["roles"][0].get("company_details", {}).keys())
