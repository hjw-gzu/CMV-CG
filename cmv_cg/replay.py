"""Reference-free candidate selection using archived scores; no model imports."""
import math

KEYS=("original_clip","multiview_clip","language_probability","caption_consensus")
DEFAULT=(.35,.25,.20,.20)


def select(pool,weights=DEFAULT):
    if not pool or len(weights)!=4 or not all(math.isfinite(w) and w>=0 for w in weights) or not math.isclose(sum(weights),1,abs_tol=1e-12):
        raise ValueError("Expected a non-empty pool and four normalized weights.")
    components=[[float(c["raw_scores"][k]) for c in pool] for k in KEYS]
    if not all(math.isfinite(v) for col in components for v in col):
        raise ValueError("Nonfinite component score")
    normalized=[]
    for col in components:
        lo,hi=min(col),max(col)
        normalized.append([(v-lo)/(hi-lo) if hi-lo>1e-12 else 0.0 for v in col])
    scores=[math.fsum(w*col[i] for w,col in zip(weights,normalized)) for i in range(len(pool))]
    return max(range(len(pool)),key=scores.__getitem__)

