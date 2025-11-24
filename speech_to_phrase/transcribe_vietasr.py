"""Transcribe audio using the VietASR Zipformer checkpoint."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Dict, Sequence

import numpy as np

from .const import RATE, TranscribingError, WordCasing
from .models import Model
from .post_process import finalize_transcript

_LOGGER = logging.getLogger(__name__)

_CHECKPOINT_NAME = "cpu_jit.pt"
_TOKENS_NAME = "tokens.txt"

@dataclass
class _Resources:
    model: "torch.jit.ScriptModule"
    fbank: "kaldifeat.Fbank"
    device: "torch.device"
    tokens: List[str]


_RESOURCES: Dict[Path, _Resources] = {}


async def transcribe_vietasr(
    model: Model, settings: "Settings", audio_stream: AsyncIterable[bytes]
) -> str:
    """Transcribe audio using VietASR."""

    torch, pad_sequence, kaldifeat = _require_dependencies()
    resources = _load_resources(model, settings, torch, kaldifeat)

    casing_func = WordCasing.get_function(model.casing)

    audio_bytes = bytearray()
    async for chunk in audio_stream:
        if chunk:
            audio_bytes.extend(chunk)

    if not audio_bytes:
        return ""

    waveform = _bytes_to_tensor(audio_bytes, torch, resources.device)
    if waveform.numel() == 0:
        return ""

    features_seq = resources.fbank([waveform])
    feature_lengths = [f.size(0) for f in features_seq]

    features = pad_sequence(
        features_seq,
        batch_first=True,
        padding_value=math.log(1e-10),
    )

    feature_lengths_tensor = torch.tensor(
        feature_lengths, device=resources.device, dtype=torch.int64
    )

    encoder_out, encoder_out_lens = resources.model.encoder(
        features=features, feature_lengths=feature_lengths_tensor
    )

    hypotheses = _greedy_search(resources.model, encoder_out, encoder_out_lens, torch)
    if not hypotheses:
        return ""

    # VietASR greedy search returns a single hypothesis for non-batch usage.
    token_ids = hypotheses[0]
    text = _tokens_to_text(token_ids, resources.tokens)
    if not text:
        return ""

    text = casing_func(text)
    text = text.strip()

    if not text:
        return ""

    return finalize_transcript(model, text)


def _require_dependencies():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise TranscribingError(
            "VietASR support requires the 'torch' library. Install with 'pip install torch'"
        ) from exc

    try:
        from torch.nn.utils.rnn import pad_sequence  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise TranscribingError(
            "VietASR support requires torch.nn utilities. Verify your torch installation."
        ) from exc

    try:
        import kaldifeat  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise TranscribingError(
            "VietASR support requires the 'kaldifeat' library. Install with 'pip install kaldifeat'."
        ) from exc

    return torch, pad_sequence, kaldifeat


def _load_resources(model: Model, settings: "Settings", torch, kaldifeat) -> _Resources:
    model_dir = settings.model_data_dir(model.id)
    model_dir.mkdir(parents=True, exist_ok=True)

    cached = _RESOURCES.get(model_dir)
    if cached is not None:
        return cached

    checkpoint_path = model_dir / _CHECKPOINT_NAME
    tokens_path = model_dir / _TOKENS_NAME

    missing_files: List[str] = []
    if not checkpoint_path.exists():
        missing_files.append(str(checkpoint_path))
    if not tokens_path.exists():
        missing_files.append(str(tokens_path))

    if missing_files:
        raise TranscribingError(
            "Missing VietASR assets. Place 'cpu_jit.pt' and 'tokens.txt' under "
            f"{model_dir}. Missing: {', '.join(missing_files)}"
        )

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    _LOGGER.debug("Loading VietASR torchscript model from %s", checkpoint_path)
    scripted_model = torch.jit.load(str(checkpoint_path), map_location=device)
    scripted_model.eval()
    scripted_model.to(device)

    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = RATE
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)
    tokens = _read_tokens(tokens_path)
    if not tokens:
        raise TranscribingError(
            f"No tokens loaded from {tokens_path}. Verify the VietASR tokens.txt file."
        )

    resources = _Resources(
        model=scripted_model,
        fbank=fbank,
        device=device,
        tokens=tokens,
    )
    _RESOURCES[model_dir] = resources
    return resources


def _bytes_to_tensor(audio_bytes: bytes, torch, device):
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return torch.zeros(0, device=device)

    waveform = torch.from_numpy(samples).to(device=device)
    waveform = waveform / 32768.0
    return waveform


def _greedy_search(model, encoder_out, encoder_out_lens, torch) -> List[List[int]]:
    if encoder_out.dim() != 3:
        raise TranscribingError("Unexpected encoder output shape for VietASR decoding.")

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = encoder_out.device
    blank_id = int(model.decoder.blank_id)
    context_size = int(model.decoder.context_size)

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    if not batch_size_list:
        return []

    batch = encoder_out.size(0)
    assert batch == batch_size_list[0], (batch, batch_size_list)

    hyps = [[blank_id] * context_size for _ in range(batch)]
    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )
    need_pad = torch.tensor([False], device=device)
    decoder_out = model.decoder(decoder_input, need_pad=need_pad).squeeze(1)

    offset = 0
    for current_batch in batch_size_list:
        start = offset
        end = offset + current_batch
        encoder_slice = packed_encoder_out.data[start:end]
        offset = end

        decoder_slice = decoder_out[:current_batch]
        logits = model.joiner(encoder_slice, decoder_slice)

        top_ids = logits.argmax(dim=1).tolist()
        emitted = False
        for idx, token_id in enumerate(top_ids):
            if token_id != blank_id:
                hyps[idx].append(int(token_id))
                emitted = True

        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:current_batch]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=need_pad).squeeze(1)

    sorted_ans = [h[context_size:] for h in hyps]
    results: List[List[int]] = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(batch):
        results.append(sorted_ans[unsorted_indices[i]])
    return results


def _tokens_to_text(token_ids: Sequence[int], tokens: Sequence[str]) -> str:
    pieces: List[str] = []
    for idx in token_ids:
        if 0 <= idx < len(tokens):
            token = tokens[idx]
            if token == "<blk>":
                continue
            pieces.append(token)

    text = "".join(pieces)
    text = text.replace("▁", " ")
    return text.strip()


def _read_tokens(tokens_path: Path) -> List[str]:
    tokens: List[str] = []
    with tokens_path.open("r", encoding="utf-8") as token_file:
        for line in token_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            try:
                index = int(parts[-1])
            except ValueError:
                continue

            if index < 0:
                continue

            if index >= len(tokens):
                tokens.extend([""] * (index + 1 - len(tokens)))
            tokens[index] = token

    return tokens


# Import at runtime to avoid circular dependency in type checking
from .const import Settings  # noqa: E402  # pylint: disable=wrong-import-position

