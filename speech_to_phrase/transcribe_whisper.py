"""Transcribe audio using a Whisper model (e.g., PhoWhisper)."""

from __future__ import annotations

import array
import logging
from pathlib import Path
from typing import Any, AsyncIterable, Dict

import numpy as np

from .const import RATE, TranscribingError, WordCasing
from .models import Model
from .post_process import finalize_transcript

_LOGGER = logging.getLogger(__name__)

_WHISPER_MODELS: Dict[str, "Any"] = {}


async def transcribe_whisper(
    model: Model, settings: "Settings", audio_stream: AsyncIterable[bytes]
) -> str:
    """Transcribe audio using Whisper (PhoWhisper)."""

    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise TranscribingError(
            "Whisper support requires the 'transformers' library. Install with 'pip install transformers torch'"
        ) from exc

    # Get model identifier from model.url or model.id
    # If model.url contains HuggingFace path, extract it
    # Otherwise, use model.id to map to HuggingFace model
    whisper_model_id = model.id
    if model.url and "huggingface.co" in model.url:
        # Extract model ID from URL like "https://huggingface.co/vinai/PhoWhisper-base"
        parts = model.url.rstrip("/").split("/")
        if len(parts) >= 2:
            whisper_model_id = f"{parts[-2]}/{parts[-1]}"
    elif model.id == "phowhisper-tiny":
        whisper_model_id = "vinai/PhoWhisper-tiny"
    elif model.id == "phowhisper-base":
        whisper_model_id = "vinai/PhoWhisper-base"
    elif model.id == "phowhisper-small":
        whisper_model_id = "vinai/PhoWhisper-small"
    elif model.id == "phowhisper-medium":
        whisper_model_id = "vinai/PhoWhisper-medium"
    elif model.id == "phowhisper-large-v2":
        whisper_model_id = "vinai/PhoWhisper-large-v2"

    # Cache loaded models
    if whisper_model_id not in _WHISPER_MODELS:
        _LOGGER.debug("Loading Whisper model: %s", whisper_model_id)
        try:
            transcriber = pipeline(
                "automatic-speech-recognition",
                model=whisper_model_id,
                device=-1,  # Use CPU by default, can be set to 0 for GPU
            )
            _WHISPER_MODELS[whisper_model_id] = transcriber
        except Exception as exc:
            raise TranscribingError(
                f"Failed to load Whisper model {whisper_model_id}: {exc}"
            ) from exc

    transcriber = _WHISPER_MODELS[whisper_model_id]
    casing_func = WordCasing.get_function(model.casing)

    # Collect all audio chunks
    audio_bytes = bytearray()
    async for chunk in audio_stream:
        if chunk:
            audio_bytes.extend(chunk)

    if not audio_bytes:
        return ""

    # Convert bytes to numpy array (16-bit PCM, mono)
    # Audio is 16-bit signed integers (2 bytes per sample)
    audio_samples = array.array("h", audio_bytes)
    audio_np = np.array(audio_samples, dtype=np.float32)
    # Normalize to [-1.0, 1.0]
    audio_np = audio_np / 32768.0

    # Transcribe
    try:
        result = transcriber({"raw": audio_np, "sampling_rate": RATE})
        text = result.get("text", "").strip()
    except Exception as exc:
        _LOGGER.exception("Error during Whisper transcription")
        raise TranscribingError(f"Whisper transcription failed: {exc}") from exc

    if not text:
        return ""

    text = casing_func(text)
    normalized = text.rstrip(".")
    return finalize_transcript(model, normalized)


# Import at runtime to avoid circular dependency in type checking
from .const import Settings  # noqa: E402  # pylint: disable=wrong-import-position