"""Deterministic beam search with immutable completed hypotheses.

This module is deliberately framework independent. The model adapter supplies
the *true* conditional log probabilities (log-softmax over the full vocabulary),
not values re-normalized after top-k truncation. For each unfinished prefix it
must supply at least its best ``beam_size`` next tokens, unless the vocabulary
or an explicit vocabulary constraint contains fewer tokens.

Search uses cumulative log probability only. The first stop token contributes
both its log probability and one token to the effective length. There are no
post-stop padding tokens. The returned mean is solely a reporting/reranking
quantity; it does not change the beam-search objective.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral, Real
from typing import Any


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _token_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("token ids must be nonnegative integers")
    return int(value)


def _rank_key(candidate: dict[str, Any]) -> tuple[float, tuple[int, ...]]:
    return (-candidate["sum_logprob"], tuple(candidate["tokens"]))


def beam_search(
    next_logprobs: Callable[
        [list[tuple[int, ...]]], Sequence[Iterable[tuple[int, float]]]
    ],
    beam_size: int = 4,
    max_tokens: int = 39,
    stop_token: int = 13,
) -> list[dict[str, Any]]:
    """Return up to ``beam_size`` candidates in deterministic cumulative order.

    ``next_logprobs`` receives unfinished prefixes only, represented by tuples
    of generated token ids. It returns one row per prefix; a row is an iterable
    of ``(token_id, conditional_log_probability)`` pairs. Row order must match
    prefix order. Invalid positive/NaN/infinite-positive scores are rejected;
    negative infinity means an impossible token and is excluded.

    Finished hypotheses participate in later pruning with their original
    tokens and score, and are never evaluated by the model again. A candidate
    reaching ``max_tokens`` without the stop token has ``ended=False``: this
    flag describes stop-token termination, not simply search completion.
    """
    beam_size = _positive_int(beam_size, "beam_size")
    max_tokens = _positive_int(max_tokens, "max_tokens")
    stop_token = _token_id(stop_token)
    if not callable(next_logprobs):
        raise TypeError("next_logprobs must be callable")

    beam: list[dict[str, Any]] = [
        {"tokens": [], "sum_logprob": 0.0, "ended": False}
    ]
    for _ in range(max_tokens):
        active = [candidate for candidate in beam if not candidate["ended"]]
        if not active:
            break
        prefixes = [tuple(candidate["tokens"]) for candidate in active]
        rows = list(next_logprobs(prefixes))
        if len(rows) != len(active):
            raise ValueError(
                f"model returned {len(rows)} rows for {len(active)} prefixes"
            )
        proposed = [candidate for candidate in beam if candidate["ended"]]
        for candidate, row in zip(active, rows):
            seen_tokens: set[int] = set()
            for pair in row:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("each model option must be (token_id, logprob)")
                token = _token_id(pair[0])
                if token in seen_tokens:
                    raise ValueError("a model row contains a duplicate token id")
                seen_tokens.add(token)
                logprob = pair[1]
                if isinstance(logprob, bool) or not isinstance(logprob, Real):
                    raise ValueError("log probabilities must be real numbers")
                logprob = float(logprob)
                if math.isnan(logprob) or logprob > 0:
                    raise ValueError("log probabilities must be <= 0 and not NaN")
                if logprob == -math.inf:
                    continue
                score = candidate["sum_logprob"] + logprob
                if not math.isfinite(score):
                    raise ValueError("cumulative log probability overflowed")
                proposed.append(
                    {
                        "tokens": candidate["tokens"] + [token],
                        "sum_logprob": score,
                        "ended": token == stop_token,
                    }
                )
        if not proposed:
            raise ValueError("no valid candidate remains after model expansion")
        beam = sorted(proposed, key=_rank_key)[:beam_size]

    result: list[dict[str, Any]] = []
    for candidate in beam:
        length = len(candidate["tokens"])
        result.append(
            {
                "tokens": list(candidate["tokens"]),
                "sum_logprob": candidate["sum_logprob"],
                "length": length,
                "mean_logprob": candidate["sum_logprob"] / length,
                "ended": candidate["ended"],
            }
        )
    return result


def select_best(
    candidates: Sequence[dict[str, Any]], mode: str = "raw"
) -> dict[str, Any]:
    """Select among already searched candidates without altering search.

    ``lengthnorm`` ranks the final beam by its real effective token length.
    This is a final-beam rerank, NOT length-normalized beam search. It cannot
    recover hypotheses pruned by cumulative-probability beam search.
    """
    if mode not in {"raw", "lengthnorm"}:
        raise ValueError("mode must be 'raw' or 'lengthnorm'")
    if not candidates:
        raise ValueError("at least one candidate is required")

    ranked: list[tuple[float, tuple[int, ...], dict[str, Any]]] = []
    for candidate in candidates:
        tokens = tuple(_token_id(token) for token in candidate["tokens"])
        if not tokens:
            raise ValueError("candidate tokens cannot be empty")
        if candidate.get("length", len(tokens)) != len(tokens):
            raise ValueError("candidate length does not match effective tokens")
        score = candidate["sum_logprob"]
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError("candidate score must be a real number")
        score = float(score)
        if not math.isfinite(score) or score > 0:
            raise ValueError("candidate cumulative score must be finite and <= 0")
        objective = score if mode == "raw" else score / len(tokens)
        ranked.append((-objective, tokens, candidate))
    # The key excludes the dict, so duplicate identical candidates remain stable.
    return min(ranked, key=lambda value: (value[0], value[1]))[2]
