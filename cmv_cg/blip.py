"""Supervised BLIP-base search with the recorded BOS-only prompt and SEP stop."""
import torch
import torch.nn.functional as F
from .search import beam_search
@torch.inference_mode()
def blip_search(model, image_embed, beam_size, max_tokens=39, chunk_size=4):
    """Only generated tokens count; stop at BLIP SEP (102), not config EOS (2)."""
    bos = int(model.config.text_config.bos_token_id)
    stop = int(model.config.text_config.sep_token_id)
    device = image_embed.device

    def next_logprobs(prefixes):
        answers = []
        for start in range(0, len(prefixes), chunk_size):
            batch = prefixes[start:start+chunk_size]
            tokens = torch.tensor([[bos] + list(p) for p in batch], dtype=torch.long, device=device)
            visual = image_embed.expand(len(batch), -1, -1)
            logits = model.text_decoder(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                encoder_hidden_states=visual,
                encoder_attention_mask=torch.ones(visual.shape[:2], dtype=torch.long, device=device),
                use_cache=False, return_dict=True).logits[:, -1, :].float()
            if not torch.isfinite(logits).all().item():
                raise ValueError('Nonfinite BLIP logits')
            scores = F.log_softmax(logits, dim=-1)
            ix = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :beam_size]
            vals = scores.gather(1, ix)
            answers.extend([list(zip(a, b)) for a, b in zip(ix.tolist(), vals.tolist())])
        return answers

    return beam_search(next_logprobs, beam_size=beam_size, max_tokens=max_tokens, stop_token=stop)
