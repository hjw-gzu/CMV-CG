"""Isolated COCOEvalCap entry; imports no caption-training framework."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


def clean_json(value):
    """SPICE category scores may be undefined; preserve these as null, not NaN."""
    import math
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-root', type=Path, help='Optional Python-3 COCO caption evaluator checkout; defaults to the installed pycocoevalcap')
    ap.add_argument('--labels', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--portable', action='store_true', help='Score with official metric implementations using isolated Java processes and portable SPICE serialization')
    args = ap.parse_args()
    root = args.eval_root.resolve() if args.eval_root else None
    if root:
        sys.path.insert(0, str(root))
    opener = gzip.open if args.predictions.endswith('.gz') else open
    with opener(args.predictions, 'rt', encoding='utf-8') as stream:
        records = json.load(stream)
    ids = [row['image_id'] for row in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Predictions must have unique, nonempty IDs')
    labels = json.loads(Path(args.labels).read_text(encoding='utf-8'))
    if set(ids) != {r['id'] for r in labels['images']}:
        raise ValueError('Prediction IDs must exactly match the evaluation split. Use matching labels for a subset.')
    if args.portable:
        import tempfile
        if root is None:
            import pycocoevalcap
            root = Path(next(iter(pycocoevalcap.__path__))).parent
        sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
        from semantic_metrics import eval_all_metrics
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as staging:
            report = eval_all_metrics(records, labels, root, Path(staging) / "evaluation")
            if len(report["metrics"]) != 8 or report["spice"]["status"] != "available":
                raise RuntimeError("Incomplete eight-metric evaluation: " + str(report["spice"].get("reason", "")))
            payload = {"metrics": report["metrics"], "image_count": len(ids), "ids": ids,
                       "per_image_values": report["per_image_values"], "scorer": "isolated COCO evaluation"}
            with output.open("x", encoding="utf-8") as f:
                json.dump(clean_json(payload), f, allow_nan=False)
        print("All eight metrics complete:", output)
        return
    from pycocotools.coco import COCO
    import pycocoevalcap.eval as module
    from pycocoevalcap.eval import COCOEvalCap
    if root and not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Scorer import did not come from requested evaluation root')
    coco = COCO(args.labels)
    coco.dataset.setdefault('info', {})
    results = coco.loadRes(records)
    scorer = COCOEvalCap(coco, results)
    scorer.params['image_id'] = ids
    scorer.evaluate()
    required = {'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE'}
    if set(scorer.eval) != required:
        raise RuntimeError(f'Incomplete metric set: {scorer.eval}')
    import math
    if not all(math.isfinite(float(v)) for v in scorer.eval.values()):
        raise RuntimeError('Nonfinite aggregate metrics')
    output = Path(args.output)
    payload = {'metrics': scorer.eval, 'image_count': len(ids), 'ids': ids,
               'scorer_path': str(module.__file__),
               'scorer_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
               'per_image_values': scorer.imgToEval,
               'warning': 'Per-image BLEU/METEOR means are not necessarily the official corpus estimand; no CI is produced here.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(clean_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
    print('All eight official point metrics complete:', output, flush=True)


if __name__ == '__main__':
    main()
