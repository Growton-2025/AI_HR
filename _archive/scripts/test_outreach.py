import requests
import json

# Test the outreach trigger endpoint
API_URL = "http://localhost:8001/api/outreach/trigger"

# You'll need a valid token - get it from browser localStorage after logging in
# Or use a test user token if available
TOKEN = input("Enter your auth token (from browser localStorage): ").strip()

# Prepare the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

payload = {
    "candidate_ids": [2519],  # Nethranand P S
    "role_id": 1,  # You may need to change this to an actual role ID
    "role_name": "AIML Engineer"
}

print("🚀 Triggering Smartlead Outreach Campaign...")
print(f"   API: {API_URL}")
print(f"   Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = requests.post(API_URL, json=payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    
    if response.status_code == 200:
        print("\n✅ Campaign triggered successfully!")
        print("   Check your Smartlead dashboard and nethranandps2001@gmail.com inbox")
    else:
        print("\n❌ Campaign trigger failed!")
        
except Exception as e:
    print(f"❌ Error: {e}")
