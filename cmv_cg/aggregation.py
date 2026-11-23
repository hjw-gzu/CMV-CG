"""Probability-mean token aggregation on a shared autoregressive context.

TTAug equation (2), and official `average` branch, commit
8433322cf538efe1c251f9945bf43181a389581f. This is an adaptation to the
frozen Diffusion Bridge captioner; no text augmentation or parameter updates.
Beam > 1 is explicitly a budget-calibration extension, not the paper's greedy rule.
"""
import math
import torch
from torch.nn import functional as F
from .search import beam_search
from .decoder import _project, _caption, _check_logits


def mixture_logprobs(logits):
    """[hypothesis, view, vocabulary] -> normalized log arithmetic mean."""
    if logits.ndim != 3 or logits.shape[1] < 1:
        raise ValueError('Expected hypotheses x views x vocabulary')
    _check_logits(logits.reshape(-1, logits.shape[-1]))
    return torch.logsumexp(F.log_softmax(logits.float(), dim=-1), dim=1) - math.log(logits.shape[1])


@torch.inference_mode()
def token_candidates(decoder, tokenizer, features, beam=1, max_tokens=39):
    if features.ndim != 2 or len(features) < 1:
        raise ValueError('Expected one feature row per view')
    visual_prefix = _project(decoder, features)
    views, prefix_length, embed = visual_prefix.shape
    def next_rows(prefixes):
        count = len(prefixes)
        prefix = visual_prefix.unsqueeze(0).expand(count, -1, -1, -1)
        prefix = prefix.reshape(count * views, prefix_length, embed)
        if prefixes[0]:
            tokens = torch.tensor(prefixes, device=prefix.device, dtype=torch.long)
            tokens = tokens[:, None, :].expand(-1, views, -1).reshape(count * views, -1)
            prefix = torch.cat([prefix, decoder.model.transformer.wte(tokens)], dim=1)
        logits = decoder.model(inputs_embeds=prefix, use_cache=False).logits[:, -1, :]
        logp = mixture_logprobs(logits.reshape(count, views, -1))
        ids = torch.argsort(logp, dim=-1, descending=True, stable=True)[:, :beam]
        values = logp.gather(1, ids)
        return [list(zip(a,b)) for a,b in zip(ids.tolist(),values.tolist())]
    found = beam_search(next_rows, beam_size=beam, max_tokens=max_tokens, stop_token=13)
    return [_caption(c,tokenizer,f'token_probability_mean_beam{beam}') for c in found]
