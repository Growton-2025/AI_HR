
from typing import Dict, Any

# --- In-memory Role Storage (in production, use database) ---
roles_store: Dict[str, Dict[str, Any]] = {
    "Account Executive Role - Middle East - Clear": {"candidates": []},
    "Account Executive Role - Deque": {"candidates": []},
    "Senior Account Manager - APAC": {"candidates": []},
}
