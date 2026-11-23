"""Re-score matched unique pools using the original hash-priority sampling rule."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cmv_cg.io import rows
from cmv_cg.models import load_clip
from cmv_cg.replay import select


def indices(dataset,backbone,iid,method,n,k,seed):
    def priority(j):
        key=json.dumps([dataset,backbone,iid,method,seed,j],sort_keys=True,ensure_ascii=False,separators=(",",":"))
        return hashlib.sha256(key.encode()).hexdigest(),j
    return sorted(sorted(range(n),key=priority)[:k])


@torch.inference_mode()
def main():
    import clip
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pools",type=Path,required=True)
    p.add_argument("--dataset",choices=("coco","flickr"),required=True)
    p.add_argument("--backbone",choices=("db","blip"),required=True)
    p.add_argument("--assets",type=Path,default=Path("assets"))
    p.add_argument("--device",default="cuda",choices=("cpu","cuda"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();model,_=load_clip(a.assets,torch.device(a.device))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as f:
        for row in rows(a.pools):
            arms={m:row["choices"][m]["pool"] for m in ("single20","ours")}
            limit=min(map(len,arms.values()))
            out={}
            captions=list(dict.fromkeys(c['caption'] for pool in arms.values() for c in pool))
            vectors=[]
            for i in range(0,len(captions),16):
                tokens=clip.tokenize(captions[i:i+16],truncate=True).to(a.device)
                vectors.append(F.normalize(model.encode_text(tokens).float(),dim=-1))
            lookup=dict(zip(captions,torch.cat(vectors)))
            for method,pool in arms.items():
                text=torch.stack([lookup[c["caption"]] for c in pool])
                gram=(text @ text.T).cpu().numpy()
                found=[]
                for seed in range(1001,1011):
                    ix=indices(a.dataset,a.backbone,row["image_id"],method,len(pool),limit,seed)
                    submatrix=gram[ix][:,ix]
                    cons=(submatrix.sum(1)-1)/(limit-1) if limit>1 else [1.]
                    sub=[dict(caption=pool[j]["caption"],raw_scores={**pool[j]["raw_scores"],
                              "caption_consensus":float(cons[t])}) for t,j in enumerate(ix)]
                    found.append(sub[select(sub)]["caption"])
                out[method]=found
            f.write(json.dumps(dict(image_id=row["image_id"],K=limit,captions=out))+"\n")


if __name__=="__main__":main()

