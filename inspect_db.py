import requests

url = "http://localhost:8001/api/outreach/shortlist/2519"
response = requests.post(url)
print(response.status_code)
print(response.json())
