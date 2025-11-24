"""Common helpers for model-specific transcription modules."""

from __future__ import annotations

from .hassil_fst import decode_meta
from .models import Model
from .vi_normalize import normalize_vietnamese_transcript


def finalize_transcript(model: Model, text: str) -> str:
    """Apply language-specific normalization and decode Hass meta markers."""

    normalized = text.strip()
    if not normalized:
        return ""

    if model.language_family == "vi":
        normalized = normalize_vietnamese_transcript(normalized)

    return decode_meta(normalized)


__all__ = ["finalize_transcript"]