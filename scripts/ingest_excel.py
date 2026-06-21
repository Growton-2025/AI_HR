import pandas as pd
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Ensure we load the environment for DB and OpenAI
load_dotenv()

from backend.db.connection import get_db_connection_context
from backend.services.import_enrichment import (
    parse_roles_from_raw, parse_education_from_raw,
    calculate_tenure_metrics, extract_profile_claims,
    _llm_classify_function, classify_company
)
from backend.api.routes.candidate_imports import _normalized_header_key, _prepare_import_rows
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _extract_raw_fields_from_row(row_series):
    raw = {}
    for col, val in row_series.items():
        if pd.isna(val):
            continue
        keys = str(col).split('/')
        current = raw
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                # peek ahead to see if next is digit
                if i + 1 < len(keys) and keys[i+1].isdigit():
                    current[k] = []
                else:
                    current[k] = {}
            current = current[k]
            if isinstance(current, list):
                idx = int(keys[i+1])
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
                break
        
        last_key = keys[-1]
        if isinstance(current, dict):
            current[last_key] = str(val)
    return raw

def test_extract():
    df = pd.read_excel("For Hayasa Product - BDR - Master List.xlsx", nrows=2)
    for idx, row in df.iterrows():
        raw = _extract_raw_fields_from_row(row)
        print(json.dumps(raw, indent=2))
        
if __name__ == "__main__":
    test_extract()
