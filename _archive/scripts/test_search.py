
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from backend.pipeline.query import process_query_main, TokenCostTracker

async def test_search():
    query = "list candidates worked in moengage"
    print(f"Testing query: '{query}'")
    
    tracker = TokenCostTracker()
    session_id = "test_session"
    
    async for item in process_query_main(query, session_id, tracker):
        if isinstance(item, str):
            print(f"STATUS: {item}")
        elif isinstance(item, dict):
            if item.get("type") == "complete":
                candidates = item.get("data", [])
                print(f"FOUND {len(candidates)} CANDIDATES")
                for c in candidates:
                     print(f" - {c.get('name')} (Company: {c.get('roles', [{}])[0].get('company')})")
            elif item.get("type") == "error":
                print(f"ERROR: {item.get('message')}")

if __name__ == "__main__":
    asyncio.run(test_search())
