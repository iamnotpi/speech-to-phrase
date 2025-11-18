"""Utilities for normalizing Vietnamese ASR transcripts."""

from __future__ import annotations

import re
from typing import List

_NUMBER_UNITS = {
    "không": 0,
    "một": 1,
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
}
_NUMBER_TOKENS = set(_NUMBER_UNITS) | {"mười", "mươi", "linh", "lẻ", "trăm"}
_NUMBER_FOLLOWERS = {
    "giờ",
    "phút",
    "giây",
    "ngày",
    "tháng",
    "năm",
    "độ",
    "phần",
    "rưỡi",
}


def _words_to_number(words: List[str]) -> int | None:
    """Convert a list of Vietnamese number words to an integer."""

    if not words:
        return None

    total = 0
    value_set = False

    for word in words:
        if word == "mười":
            if total == 0:
                total = 10
            else:
                total += 10
            value_set = True
            continue

        if word == "mươi":
            total = (total if total else 1) * 10
            value_set = True
            continue

        if word in ("linh", "lẻ"):
            if total == 0:
                total = 0
            value_set = True
            continue

        if word == "trăm":
            total = (total if total else 1) * 100
            value_set = True
            continue

        unit = _NUMBER_UNITS.get(word)
        if unit is None:
            return None

        if total >= 10 or value_set:
            total += unit
        else:
            total = unit

        value_set = True

    return total if value_set else None


def normalize_vietnamese_numbers(text: str) -> str:
    """Replace Vietnamese number words with digits for consistency."""

    tokens = text.split()
    if not tokens:
        return text

    normalized_tokens: List[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _NUMBER_TOKENS:
            j = i
            while j < len(tokens) and tokens[j] in _NUMBER_TOKENS:
                j += 1

            number_value = _words_to_number(tokens[i:j])
            next_token = tokens[j] if j < len(tokens) else ""

            should_convert = False
            if next_token in _NUMBER_FOLLOWERS:
                should_convert = True
            elif next_token == "phần" and j + 1 < len(tokens) and tokens[j + 1] == "trăm":
                should_convert = True
            elif next_token == "rưỡi":
                should_convert = True
            elif j == len(tokens) and number_value and number_value >= 10:
                should_convert = True

            prev_token = tokens[i - 1] if i > 0 else ""

            if (
                number_value is not None
                and should_convert
                and not (number_value == 0 and tokens[i:j] == ["không"])
                and not (tokens[i:j] == ["trăm"] and prev_token == "phần")
            ):
                normalized_tokens.append(str(number_value))
                i = j
                continue

        normalized_tokens.append(tokens[i])
        i += 1

    return " ".join(normalized_tokens)


def normalize_cancellation_terms(text: str) -> str:
    """Align common cancellation spellings with Home Assistant expectations."""

    normalized = text.replace("hủy", "huỷ")
    normalized = normalized.replace("Hủy", "Huỷ")
    return normalized


def normalize_vietnamese_transcript(text: str) -> str:
    """Apply common normalizations to Vietnamese transcripts from ASR models."""

    normalized = text.replace("tivi", "tv")
    normalized = normalized.replace("ti vi", "tv")
    normalized = normalized.replace("ti-vi", "tv")
    normalized = normalized.replace("ti. vi", "tv")
    normalized = re.sub("ga ra", "gara", normalized)
    normalized = normalize_vietnamese_numbers(normalized)
    normalized = normalize_cancellation_terms(normalized)
    return normalized


__all__ = [
    "normalize_cancellation_terms",
    "normalize_vietnamese_numbers",
    "normalize_vietnamese_transcript",
]

