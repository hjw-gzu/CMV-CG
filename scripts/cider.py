"""Recompute corpus CIDEr-D or per-image candidate-pool oracles with fixed split IDF."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmv_cg.io import read, rows, write


def score(labels, predictions=None, pools=None):
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    refs = defaultdict(list)
    for r in read(labels)["annotations"]:
        refs[int(r["image_id"])].append({"caption": r["caption"]})
    tok = PTBTokenizer()
    ground = tok.tokenize(dict(refs))
    if predictions:
        opener = gzip.open if predictions.suffix == ".gz" else open
        with opener(predictions, "rt", encoding="utf-8") as f:
            records = json.load(f)
        hyps = {r["image_id"]: [{"caption": r["caption"]}] for r in records}
        if len(hyps) != len(records) or ground.keys() != hyps.keys():
            raise ValueError("Supply unique predictions for every image in the same split")
        value, _ = Cider().compute_score(ground, tok.tokenize(hyps))
        return {"images": len(hyps), "CIDEr_x100": float(value) * 100}

    from cmv_cg.oracle import FixedCider
    oracle = FixedCider(ground)
    ordered = dict.fromkeys(c["caption"] for r in rows(pools)
                            for arm in ("single20", "ours") for c in r["choices"][arm]["pool"])
    texts = list(ordered)
    tokens = tok.tokenize({i: [{"caption": s}] for i, s in enumerate(texts)})
    text_to_tokens = {s: tokens[i][0] for i, s in enumerate(texts)}
    outputs = defaultdict(list)
    seen = set()
    for r in rows(pools):
        image = int(r["image_id"])
        if image in seen:
            raise ValueError(f"Duplicate image ID {image}")
        seen.add(image)
        for arm in ("single20", "ours"):
            record = r["choices"][arm]
            scores = [100 * oracle.score(image, text_to_tokens[c["caption"]]) for c in record["pool"]]
            chosen = scores[record["selected"]]
            outputs[arm].append((len(scores), max(scores), chosen, max(scores) - chosen))
    if seen != set(ground):
        raise ValueError("Reference and candidate image IDs differ")
    metrics = ("unique", "oracle_CIDEr", "selected_CIDEr", "gap")
    return {arm: {name: math.fsum(col) / len(col) for name, col in zip(metrics, zip(*values))}
            for arm, values in outputs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, help="Optional Python 3 COCO evaluator checkout")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--pools", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.eval_root:
        sys.path.insert(0, str(a.eval_root.resolve()))
    res = score(a.labels, a.predictions, a.pools)
    if a.output:
        write(a.output, res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
