"""Transcribe audio using the ChunkFormer Vietnamese model."""

from __future__ import annotations

import logging
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Dict, Iterable
from urllib.parse import urlparse

from .const import CHANNELS, RATE, WIDTH, TranscribingError, WordCasing
from .models import Model
from .post_process import finalize_transcript

_LOGGER = logging.getLogger(__name__)


@dataclass
class _Resources:
    model: "ChunkFormerModel"
    device: "torch.device"


_RESOURCES: Dict[Path, _Resources] = {}


async def transcribe_chunkformer(
    model: Model, settings: "Settings", audio_stream: AsyncIterable[bytes]
) -> str:
    """Transcribe audio using a ChunkFormer model."""

    ChunkFormerModel, snapshot_download, torch = _require_dependencies()
    resources = _load_resources(model, settings, ChunkFormerModel, snapshot_download, torch)

    audio_bytes = bytearray()
    async for chunk in audio_stream:
        if chunk:
            audio_bytes.extend(chunk)

    if not audio_bytes:
        return ""

    temp_path = _write_temp_wav(audio_bytes)

    try:
        try:
            decoded = resources.model.endless_decode(
                str(temp_path),
                chunk_size=64,
                left_context_size=128,
                right_context_size=128,
                total_batch_duration=600,
                return_timestamps=False,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            _LOGGER.exception("ChunkFormer transcription failed")
            raise TranscribingError(f"ChunkFormer transcription failed: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    text = _decode_to_text(decoded)
    if not text:
        return ""

    casing_func = WordCasing.get_function(model.casing)
    text = casing_func(text).strip()
    if not text:
        return ""

    return finalize_transcript(model, text)


def _require_dependencies():
    try:
        from chunkformer import ChunkFormerModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise TranscribingError(
            "ChunkFormer support requires the 'chunkformer' package. "
            "Install with 'pip install chunkformer'."
        ) from exc

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise TranscribingError(
            "ChunkFormer support requires the 'huggingface-hub' package. "
            "Install with 'pip install huggingface_hub'."
        ) from exc

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise TranscribingError(
            "ChunkFormer support requires the 'torch' library. Install with 'pip install torch'."
        ) from exc

    return ChunkFormerModel, snapshot_download, torch


def _load_resources(
    model: Model,
    settings: "Settings",
    ChunkFormerModel,
    snapshot_download,
    torch,
) -> _Resources:
    model_dir = settings.model_data_dir(model.id)
    model_dir.mkdir(parents=True, exist_ok=True)

    cached = _RESOURCES.get(model_dir)
    if cached is not None:
        return cached

    _ensure_model_files(model, model_dir, snapshot_download)

    _LOGGER.debug("Loading ChunkFormer model from %s", model_dir)
    try:
        chunkformer_model = ChunkFormerModel.from_pretrained(str(model_dir))
    except Exception as exc:  # pragma: no cover - runtime guard
        raise TranscribingError(f"Failed to load ChunkFormer model: {exc}") from exc

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    chunkformer_model.to(device)
    chunkformer_model.eval()

    resources = _Resources(model=chunkformer_model, device=device)
    _RESOURCES[model_dir] = resources
    return resources


def _ensure_model_files(model: Model, model_dir: Path, snapshot_download) -> None:
    """Ensure the ChunkFormer model files are available locally."""

    expected_files = ("config.yaml", "pytorch_model.pt", "vocab.txt")
    if all((model_dir / name).exists() for name in expected_files):
        return

    repo_id = _extract_repo_id(model)
    if repo_id is None:
        raise TranscribingError(
            f"Cannot determine HuggingFace repo id for ChunkFormer model '{model.id}'. "
            "Set the model URL to the HuggingFace repository."
        )

    _LOGGER.info("Downloading ChunkFormer model %s to %s", repo_id, model_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        raise TranscribingError(
            f"Failed to download ChunkFormer model '{repo_id}': {exc}"
        ) from exc


def _extract_repo_id(model: Model) -> str | None:
    """Extract the HuggingFace repository id from the model definition."""

    if model.url and "huggingface.co" in model.url:
        parsed = urlparse(model.url)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"

    return None


def _write_temp_wav(audio_bytes: bytes) -> Path:
    handle, path_str = tempfile.mkstemp(suffix=".wav")
    os.close(handle)

    path = Path(path_str)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(WIDTH)
        wav_file.setframerate(RATE)
        wav_file.writeframes(audio_bytes)

    return path


def _decode_to_text(decoded: object) -> str:
    if decoded is None:
        return ""

    if isinstance(decoded, str):
        return decoded.strip()

    if isinstance(decoded, dict):
        pieces = [
            str(item).strip()
            for item in decoded.values()
            if isinstance(item, str) and item.strip()
        ]
        return " ".join(pieces).strip()

    if isinstance(decoded, Iterable):
        segments: list[str] = []
        for item in decoded:
            if isinstance(item, str):
                piece = item.strip()
            elif isinstance(item, dict):
                piece = str(item.get("decode", "")).strip()
            else:
                piece = str(item).strip()

            if piece:
                segments.append(piece)

        return " ".join(segments).strip()

    return str(decoded).strip()


# Import at runtime to avoid circular dependency in type checking
from .const import Settings  # noqa: E402  # pylint: disable=wrong-import-position


