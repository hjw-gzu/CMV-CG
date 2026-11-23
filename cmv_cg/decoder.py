"""Frozen GPT-2 prefix decoder and CLIP caption scoring."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .search import beam_search, select_best

COMPONENTS = (
    "original_clip", "multiview_clip", "language_probability", "caption_consensus"
)
DOMAIN_WEIGHTS = (0.35, 0.25, 0.20, 0.20)
MAX_TOKENS = 39
BEAM_SIZE = 4
MINMAX_EPSILON = 1e-12


class _MLP(nn.Module):
    """Exactly the original MLP layout, including state_dict key names."""

    def __init__(self, prefix_size: int, embed_size: int, prefix_length: int):
        super().__init__()
        hidden = embed_size * prefix_length // 2
        self.model = nn.Sequential(
            nn.Linear(prefix_size, hidden), nn.Tanh(),
            nn.Linear(hidden, embed_size * prefix_length),
        )

    def forward(self, x):
        return self.model(x)


class _CaptionDecoder(nn.Module):
    def __init__(self, gpt2):
        super().__init__()
        self.model = gpt2
        self.embed_size = int(gpt2.transformer.wte.weight.shape[1])
        self.prefix_length = 10
        self.clip_project = _MLP(512, self.embed_size, self.prefix_length)


def load_caption(config_path, gpt2_dir, checkpoint, device):
    """Strictly load the archived 512->MLP->10x768->GPT-2 architecture.

    Unsupported configurations and unmatched checkpoint keys are errors, never
    silently ignored. Only a Lightning state_dict whose every key starts with
    ``model.`` is accepted; stripping it once recovers the original Decoder keys.
    """
    import yaml
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    with Path(config_path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    expected = {
        "decoder.modality": "language", "model.mapping_type": "mlp",
        "model.prefix_length": 10, "encoder.embed_dim": 512,
    }
    for name, wanted in expected.items():
        value = cfg[name.split(".")[0]][name.split(".")[1]]
        if isinstance(value, str):
            value = value.lower()
        if value != wanted:
            raise ValueError(f"Unsupported {name}={value!r}; expected {wanted!r}")
    if cfg["model"].get("checkpoint") is not None:
        raise ValueError("An auxiliary model.checkpoint is not allowed in this decoder")
    model_dir = Path(gpt2_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    config = GPT2Config.from_pretrained(str(model_dir), local_files_only=True)
    if config.n_embd != 768 or config.vocab_size != 50257:
        raise ValueError("Expected GPT-2 small (embedding 768; vocabulary 50257)")
    gpt2 = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir), local_files_only=True)
    stop_ids = tokenizer.encode(".", add_special_tokens=False)
    if stop_ids != [13]:
        raise ValueError(f"Unexpected GPT-2 period token: {stop_ids!r}")
    decoder = _CaptionDecoder(gpt2)
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
        raise ValueError("Expected a Lightning checkpoint with a state_dict mapping")
    state = saved["state_dict"]
    other_keys = [name for name in state if not name.startswith("model.")]
    if not state or other_keys:
        raise ValueError(f"Unsupported non-decoder checkpoint keys: {other_keys[:20]}")
    decoder.load_state_dict({key[6:]: value for key, value in state.items()}, strict=True)
    decoder.to(device).eval()
    decoder.requires_grad_(False)
    metadata = {
        "loader": "strict_minimal_original_mlp_decoder",
        "checkpoint_keys": len(state), "missing_keys": [], "unexpected_keys": [],
        "prefix_length": 10, "encoder_embed_dim": 512, "gpt2_embed_dim": 768,
        "vocab_size": 50257, "stop_token": 13, "local_files_only": True,
        "checkpoint_epoch": saved.get("epoch"),
        "checkpoint_global_step": saved.get("global_step"),
    }
    return decoder, tokenizer, metadata


def _device(decoder):
    return next(decoder.parameters()).device


def _project(decoder, features):
    features = features.to(device=_device(decoder), dtype=torch.float32)
    projected = decoder.clip_project(features)
    return projected.reshape(features.shape[0], decoder.prefix_length, decoder.embed_size)


def _caption(candidate, tokenizer, source):
    result = dict(candidate)
    result["tokens"] = list(candidate["tokens"])
    result["caption"] = tokenizer.decode(result["tokens"]).strip()
    result["source"] = source
    if not result["caption"]:
        raise ValueError("An empty decoded caption cannot enter evaluation or scoring")
    return result


def _check_logits(logits):
    if logits.ndim != 2 or not torch.isfinite(logits).all().item():
        raise ValueError("Decoder produced non-finite logits or an unexpected shape")




@torch.inference_mode()
def beam_candidates(decoder, tokenizer, feature, source, beam_size=BEAM_SIZE,
                    max_tokens=MAX_TOKENS):
    """Raw-cumulative beam search; finished sequences never re-enter the model."""
    prefix = _project(decoder, feature.reshape(1, -1))
    stop = tokenizer.encode(".", add_special_tokens=False)
    if len(stop) != 1:
        raise ValueError("Period must be a single tokenizer token")

    def next_logprobs(prefixes):
        lengths = {len(tokens) for tokens in prefixes}
        if len(lengths) != 1:
            raise ValueError("Active hypotheses must have the same search depth")
        embeddings = prefix.expand(len(prefixes), -1, -1)
        if prefixes and prefixes[0]:
            tokens = torch.tensor(prefixes, device=prefix.device, dtype=torch.long)
            embeddings = torch.cat((embeddings, decoder.model.transformer.wte(tokens)), dim=1)
        logits = decoder.model(inputs_embeds=embeddings, use_cache=False).logits[:, -1, :].float()
        _check_logits(logits)
        logprobs = F.log_softmax(logits, dim=-1)
        # Stable order of equal logits is increasing token id, not torch.topk's
        # unspecified tie order. Scores remain FULL-vocabulary log probabilities.
        indices = torch.argsort(logprobs, dim=-1, descending=True, stable=True)[:, :beam_size]
        values = torch.gather(logprobs, 1, indices)
        return [list(zip(ids, scores)) for ids, scores in zip(indices.tolist(), values.tolist())]

    return [_caption(candidate, tokenizer, source) for candidate in beam_search(
        next_logprobs, beam_size=beam_size, max_tokens=max_tokens, stop_token=stop[0]
    )]


def _validate_candidate(candidate, stop_token):
    tokens = candidate["tokens"]
    if not tokens or len(tokens) != candidate["length"] or len(tokens) > MAX_TOKENS:
        raise ValueError("Candidate effective token length is inconsistent")
    if stop_token in tokens and tokens.index(stop_token) != len(tokens) - 1:
        raise ValueError("Post-stop padding detected in beam-search candidate")
    if bool(candidate["ended"]) != (tokens[-1] == stop_token):
        raise ValueError("Candidate stop flag is inconsistent")
    if not math.isclose(candidate["mean_logprob"] * len(tokens), candidate["sum_logprob"],
                        rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("Candidate mean probability uses an incorrect denominator")


def _deduplicate(candidates):
    """Stable first text occurrence; higher-mean duplicate replaces in place."""
    unique = {}
    for candidate in candidates:
        key = candidate["caption"].lower().strip()
        if not key:
            raise ValueError("Empty caption cannot enter the pool")
        if key not in unique:
            unique[key] = dict(candidate, sources=[candidate["source"]])
        else:
            sources = unique[key]["sources"]
            if candidate["source"] not in sources:
                sources = sources + [candidate["source"]]
            if candidate["mean_logprob"] > unique[key]["mean_logprob"]:
                unique[key] = dict(candidate, sources=sources)
            else:
                unique[key]["sources"] = sources
    return list(unique.values())


@torch.inference_mode()
def score_pool(pool, clip_model, raw_views, weights=DOMAIN_WEIGHTS, *, visual_scores=None):
    """Reference-free score computation; min-max and weighted sum use float64."""
    import clip

    if not pool:
        raise ValueError("Cannot rank an empty pool")
    if len(weights) != 4 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Expected four finite nonnegative weights")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
        raise ValueError("Weights must sum to one")
    device = next(clip_model.parameters()).device
    texts = clip.tokenize([item["caption"] for item in pool], truncate=True).to(device)
    text = F.normalize(clip_model.encode_text(texts).float(), dim=-1)
    if visual_scores is None:
        image = F.normalize(raw_views.to(device=device, dtype=torch.float32), dim=-1)
        original = text @ image[0]
        multiview = (text @ image.T).mean(1)
    else:
        original, multiview = torch.tensor(visual_scores, dtype=torch.float64).T
    consensus = ((text @ text.T).sum(1).sub(1).div(len(pool) - 1)
                 if len(pool) > 1 else torch.ones(1, device=device))
    components = [original.tolist(), multiview.tolist(),
                  [math.exp(item["mean_logprob"]) for item in pool], consensus.tolist()]
    normalized = []
    component_ranges = {}
    for name, values in zip(COMPONENTS, components):
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite raw {name} score")
        minimum, maximum = min(values), max(values)
        difference = maximum - minimum
        constant = difference <= MINMAX_EPSILON
        normalized.append([0.0 if constant else (value - minimum) / difference for value in values])
        component_ranges[name] = {"min": minimum, "max": maximum, "constant": constant}
    scored = []
    for index, candidate in enumerate(pool):
        item = dict(candidate)
        item["raw_scores"] = {name: components[j][index] for j, name in enumerate(COMPONENTS)}
        item["normalized_scores"] = {name: normalized[j][index] for j, name in enumerate(COMPONENTS)}
        item["final_score"] = math.fsum(weights[j] * normalized[j][index] for j in range(4))
        scored.append(item)
    best = max(range(len(scored)), key=lambda index: scored[index]["final_score"])
    return scored, best, component_ranges
