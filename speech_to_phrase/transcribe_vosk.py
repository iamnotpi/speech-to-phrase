"""Transcribe audio using a Vosk model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import AsyncIterable, Dict, List

from .const import RATE, TranscribingError, WordCasing
from .hassil_fst import decode_meta
from .models import Model

_LOGGER = logging.getLogger(__name__)

_VOSK_MODELS: Dict[Path, "vosk.Model"] = {}


async def transcribe_vosk(
    model: Model, settings: "Settings", audio_stream: AsyncIterable[bytes]
) -> str:
    """Transcribe audio using Vosk."""

    try:
        from vosk import KaldiRecognizer, Model as VoskModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise TranscribingError(
            "Vosk support requires the 'vosk' extra. Install with 'pip install speech-to-phrase[vosk]'"
        ) from exc

    model_dir = (settings.models_dir / model.id).absolute()
    if not model_dir.is_dir():
        raise TranscribingError(f"Vosk model directory does not exist: {model_dir}")

    vosk_model = _VOSK_MODELS.get(model_dir)
    if vosk_model is None:
        _LOGGER.debug("Loading Vosk model from %s", model_dir)
        vosk_model = VoskModel(str(model_dir))
        _VOSK_MODELS[model_dir] = vosk_model

    recognizer = KaldiRecognizer(vosk_model, RATE)
    casing_func = WordCasing.get_function(model.casing)

    recognized_fragments: List[str] = []
    async for chunk in audio_stream:
        if not chunk:
            continue

        if recognizer.AcceptWaveform(chunk):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                recognized_fragments.append(text)

    final_result = json.loads(recognizer.FinalResult())
    final_text = final_result.get("text", "").strip()
    if final_text:
        recognized_fragments.append(final_text)

    if not recognized_fragments:
        return ""

    text = " ".join(recognized_fragments)
    text = casing_func(text)
    return decode_meta(text)


# Import at runtime to avoid circular dependency in type checking
from .const import Settings  # noqa: E402  # pylint: disable=wrong-import-position