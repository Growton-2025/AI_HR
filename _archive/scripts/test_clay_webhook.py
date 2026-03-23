
import requests
import json

# Try port 8000 and 8001
PORTS = [8001]

def test_webhook():
    payload = {
        "first_name": "Test",
        "last_name": "Bot",
        "result_email": "test.bot@example.com",
        "mobile_phone": "+1-555-0199",
        "linkedin_url": "https://www.linkedin.com/in/non-existent-test-user/"
    }
    
    paths = ["/results", "/api/results"]
    
    # Targets: Local Backend (8001) AND Public Ngrok
    targets = [
        "http://localhost:8001/results",
        "https://f19f-2409-40f4-4102-5b4-e9cf-cd35-2dd1-aee8.ngrok-free.app/results"
    ]
    
    for url in targets:
        print(f"Testing URL: {url}")
        try:
            response = requests.post(url, json=payload, timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            if response.status_code == 200:
                print("✅ Webhook endpoint is active and accepted the data!")
            else:
                print("❌ Endpoint returned error.")
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    test_webhook()
