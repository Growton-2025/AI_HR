
import sys
import os

# Add the individual folder to path to import existing modules
# This allows us to import 'query' from the individual directory
# Path: web/backend/app/services/query_wrapper.py
# Target: individual/query.py
current_dir = os.path.dirname(os.path.abspath(__file__))
individual_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', 'individual'))

if individual_path not in sys.path:
    sys.path.insert(0, individual_path)

try:
    from query import (
        process_query_main,
        load_all_profiles_from_db,
        load_all_company_names_from_db,
        TokenCostTracker,
        PROFILES_BY_ID,
        ALL_COMPANY_NAMES,
        get_db_connection,
        profiles_to_excel,
        SALES_TAXONOMY,
        SEGMENT_SYNONYMS,
        COMPANY_DETAILS_TAXONOMY,
        CULTURE_TAXONOMY,
        GEOGRAPHY_COUNTRY_TO_REGION_MAP
    )
except ImportError as e:
    print(f"Error importing query module from {individual_path}: {e}")
    # Define dummy objects to prevent crash during development if path is wrong
    PROFILES_BY_ID = {}
    ALL_COMPANY_NAMES = []
    SALES_TAXONOMY = {}
    SEGMENT_SYNONYMS = {}
    COMPANY_DETAILS_TAXONOMY = {}
    CULTURE_TAXONOMY = {}
    GEOGRAPHY_COUNTRY_TO_REGION_MAP = {}
    
    def process_query_main(*args, **kwargs): pass
    def load_all_profiles_from_db(*args, **kwargs): pass
    def load_all_company_names_from_db(*args, **kwargs): pass
    def get_db_connection(*args, **kwargs): pass
    def profiles_to_excel(*args, **kwargs): pass
    class TokenCostTracker: 
        total_tokens = 0
        total_cost = 0.0
