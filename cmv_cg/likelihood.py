"""Teacher-forced probabilities on the original visual prefix (Table 3e)."""
import math
import torch

def likelihoods(decoder, tokenizer, feature, cs):
    """p(token_t | original-image prefix, all preceding actual output tokens)."""
    t = torch
    F = torch.nn.functional
    device = feature.device
    with t.inference_mode():
        prefix=decoder.clip_project(feature).reshape(1,decoder.prefix_length,decoder.embed_size)
        result=[]
        for offset in range(0,len(cs),4):
            batch=cs[offset:offset+4];lengths=[len(c['tokens']) for c in batch];mx=max(lengths)
            for c,n in zip(batch,lengths):
                if n!=c['length'] or not 1<=n<=39 or tokenizer.decode(c['tokens']).strip()!=c['caption']:
                    raise ValueError('Archived GPT-2 tokens/length/text mismatch')
                if 13 in c['tokens'][:-1]:raise ValueError('Post-period tokens in archived candidate')
            ids=t.zeros((len(batch),mx),dtype=t.long,device=device)
            for j,c in enumerate(batch):ids[j,:lengths[j]]=t.tensor(c['tokens'],device=device)
            inp=t.cat([prefix.expand(len(batch),-1,-1),decoder.model.transformer.wte(ids[:,:-1])],dim=1)
            # Right padding is after every valid next-token position and cannot
            # affect earlier causal predictions. It is also masked explicitly.
            mask=t.ones(inp.shape[:2],dtype=t.long,device=device)
            for j,n in enumerate(lengths):mask[j,decoder.prefix_length+n-1:]=0
            logits=decoder.model(inputs_embeds=inp,attention_mask=mask,use_cache=False).logits[:,decoder.prefix_length-1:decoder.prefix_length+mx-1,:].float()
            logp=F.log_softmax(logits,dim=-1).gather(2,ids.unsqueeze(-1)).squeeze(-1)
            for j,n in enumerate(lengths):
                terms=logp[j,:n].tolist();s=math.fsum(terms)
                if not math.isfinite(s) or s>0:raise ValueError('Nonfinite or positive log likelihood')
                result.append(dict(sum_logprob=s,mean_logprob=s/n,length=n,probability=math.exp(s/n)))
    return result
