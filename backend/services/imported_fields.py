"""Helpers for preserving and exposing spreadsheet-only candidate fields."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, Optional, Tuple


IMPORTED_EXTRA_FIELDS_KEY = "imported_extra_fields"

_HEADER_SYNONYMS: Dict[str, str] = {
    "current ctc": "current_ctc",
    "current compensation": "current_ctc",
    "current salary": "current_ctc",
    "curr ctc": "current_ctc",
    "ctc": "current_ctc",
    "expected ctc": "expected_ctc",
    "expected compensation": "expected_ctc",
    "expected salary": "expected_ctc",
    "exp ctc": "expected_ctc",
    "notice period": "notice_period",
    "np": "notice_period",
    "preferred loc": "preferred_location",
    "preferred location": "preferred_location",
    "preferred city": "preferred_location",
    "focused geog": "focused_geography",
    "focused geo": "focused_geography",
    "focused geography": "focused_geography",
    "geography focus": "focused_geography",
    "shift timings": "shift_timings",
    "shift timing": "shift_timings",
    "shift time": "shift_timings",
    "outbound exp": "outbound_experience",
    "outbound experience": "outbound_experience",
    "outbound": "outbound_experience",
    "targets": "targets",
    "target": "targets",
    "cv": "cv",
    "resume": "cv",
}


def normalize_imported_field_key(header: str) -> str:
    """Return a stable AI-token key for a variable spreadsheet header."""
    text = re.sub(r"[^a-z0-9]+", " ", str(header or "").strip().lower()).strip()
    if not text:
        return "field"
    if text in _HEADER_SYNONYMS:
        return _HEADER_SYNONYMS[text]
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "field"


def labelize_imported_field_key(key: str, fallback: Optional[str] = None) -> str:
    label = str(fallback or "").strip()
    if label:
        return label
    return " ".join(part.capitalize() for part in re.split(r"[_\.\-]+", str(key or "")) if part)


def build_imported_extra_fields(raw_values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build normalized metadata while keeping source headers in the raw row."""
    out: Dict[str, Dict[str, Any]] = {}
    for source_header, value in (raw_values or {}).items():
        if source_header == IMPORTED_EXTRA_FIELDS_KEY or value in (None, ""):
            continue
        base_key = normalize_imported_field_key(source_header)
        key = base_key
        suffix = 2
        while key in out and out[key].get("source_header") != source_header:
            key = f"{base_key}_{suffix}"
            suffix += 1
        out[key] = {
            "label": labelize_imported_field_key(base_key, source_header),
            "source_header": str(source_header),
            "value": value,
        }
    return out


def merge_imported_extra_fields(raw_values: Dict[str, Any]) -> Dict[str, Any]:
    """Return raw fields plus the normalized imported-extra metadata block."""
    raw = dict(raw_values or {})
    extras = build_imported_extra_fields(raw)
    if extras:
        existing = raw.get(IMPORTED_EXTRA_FIELDS_KEY)
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(extras)
            extras = merged
        raw[IMPORTED_EXTRA_FIELDS_KEY] = extras
    return raw


def iter_imported_extra_fields(raw_fields: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if not isinstance(raw_fields, dict):
        return
    extras = raw_fields.get(IMPORTED_EXTRA_FIELDS_KEY)
    if not isinstance(extras, dict):
        return
    for key, meta in extras.items():
        if not isinstance(meta, dict):
            continue
        value = meta.get("value")
        if value in (None, ""):
            continue
        yield str(key), meta
