
#!/usr/bin/env python3
import sys
import os

# Add the project root to sys.path to allow imports from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.pipeline.ingest import ingest_data

if __name__ == "__main__":
    print("Running data ingestion pipeline...")
    ingest_data()
    print("Pipeline finished.")
