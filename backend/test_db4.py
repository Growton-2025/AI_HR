import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline.query import get_analytics_summary
import asyncio

async def run():
    res = await get_analytics_summary("admin@example.com", "admin")
    print("Functional Distribution:", res["distributions"]["functional"])

asyncio.run(run())
