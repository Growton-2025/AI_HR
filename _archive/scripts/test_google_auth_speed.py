import time
import httpx

# Simulate the Google token verification call
async def test_google_api():
    # Use a dummy token to see how long the API call takes
    start = time.time()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://oauth2.googleapis.com/tokeninfo?id_token=dummy_token_for_timing_test",
                timeout=10.0
            )
            elapsed = time.time() - start
            print(f"Google API Response Time: {elapsed:.2f}s")
            print(f"Status Code: {response.status_code}")
        except Exception as e:
            elapsed = time.time() - start
            print(f"Google API Error after {elapsed:.2f}s: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_google_api())
