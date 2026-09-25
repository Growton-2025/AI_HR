"""Single shared LinkedIn identity normalisation for imports, enrichment,
webhooks, pollers and the person-history linking that sits on top of them.

The canonical key for a LinkedIn profile is ``/in/<slug>``: lower-cased,
percent-decoded, no host, no query, no trailing segments. Everything that
matches candidates by LinkedIn must compare canonical keys — the column used
to hold three formats at once (``/in/slug`` from the app, bare ``slug`` from
the legacy pipeline, raw URLs from scripts), so 73% of rows could never match
each other and the HeyReach poller had to try all three by hand.
"""
import re
from typing import Optional
from urllib.parse import unquote, urlparse

# Old-style public profile URLs: linkedin.com/pub/first-last/1a/2b3/4c5. The
# vanity slug is the first segment; the rest are opaque routing codes.
_PROFILE_PATH_RE = re.compile(r"^/(?:in|pub)/([^/?#]+)")
# A bare vanity slug typed or stored on its own (no host, no path).
_BARE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._%-]{1,99}$")
# Suffix the master-library dedupe appends to a duplicate row's key so the
# unique index holds; the person behind the row is unchanged.
LEGACY_SUFFIX_RE = re.compile(r"_legacy_\d+$")


def _clean_slug(slug: str) -> Optional[str]:
    slug = unquote(slug or "").strip().strip("/").lower()
    return slug or None


def normalize_linkedin(url: Optional[str]) -> Optional[str]:
    """Canonical ``/in/<slug>`` for any way a LinkedIn profile can be written,
    or None when the value is not a LinkedIn *profile* (company pages, other
    hosts, empty)."""
    if url is None:
        return None
    raw = str(url).strip()
    if not raw:
        return None

    lowered = raw.lower()
    # Stored/typed as just the vanity slug ("jane-doe-123").
    if "/" not in lowered and "." not in lowered and ":" not in lowered:
        slug = _clean_slug(lowered)
        return f"/in/{slug}" if slug and _BARE_SLUG_RE.match(slug) else None

    if raw.startswith("/"):
        # Already a path ('/in/jane-doe'): what the app itself stores.
        path = re.sub(r"/+", "/", raw)
    else:
        if not re.match(r"^[a-z][a-z0-9+.-]*://", lowered):
            raw = f"https://{raw}"
        try:
            parsed = urlparse(raw)
        except Exception:
            return None

        host = (parsed.netloc or "").lower().split("@")[-1].split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        # Locale mirrors (uk.linkedin.com, in.linkedin.com, …) serve the same profile.
        if host != "linkedin.com" and not host.endswith(".linkedin.com"):
            return None
        path = re.sub(r"/+", "/", (parsed.path or "").strip())
    match = _PROFILE_PATH_RE.match(path.lower())
    if not match:
        return None
    slug = _clean_slug(match.group(1))
    return f"/in/{slug}" if slug else None


def person_key(stored: Optional[str]) -> Optional[str]:
    """The canonical key for a value already in ``candidates.normalized_linkedin``
    — any of the historical formats, with a dedupe suffix stripped."""
    if not stored:
        return None
    text = LEGACY_SUFFIX_RE.sub("", str(stored).strip())
    if not text:
        return None
    if text.startswith("/") and "/" not in text[1:] and "." not in text:
        # '/aroon' — a slug with a stray leading slash.
        text = text[1:]
    return normalize_linkedin(text)


def is_canonical_linkedin_key(stored: Optional[str]) -> bool:
    return bool(stored) and normalize_linkedin(stored) == stored


def canonical_email(value: Optional[str]) -> Optional[str]:
    """Lower-cased, trimmed; None for blank. Email matching used to be
    case-sensitive and untrimmed, so ``Jane@X.com`` never matched ``jane@x.com``."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None
