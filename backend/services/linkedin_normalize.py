"""Single shared LinkedIn URL normalization for imports, enrichment, and webhooks."""
import re
from urllib.parse import urlparse
from typing import Optional

_LI_HOST_RE = re.compile(r"(linkedin\.com|www\.linkedin\.com)$", re.I)


def normalize_linkedin(url: Optional[str]) -> Optional[str]:
    if not url or not str(url).strip():
        return None
    raw = str(url).strip()
    if not raw.startswith("http"):
        raw = f"https://{raw.lstrip('/')}"
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not _LI_HOST_RE.search(host or ""):
        path = re.sub(r"/+", "/", (parsed.path or "").strip().lower())
        path = path.rstrip("/") or path
        return path or None
    path = re.sub(r"/+", "/", (parsed.path or "").strip().lower())
    path = path.rstrip("/") or path
    return path or None
