"""Recompute reference-free selections from recorded candidate scores."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmv_cg.io import ROOT, read, rows, write
from cmv_cg.replay import DEFAULT, select


def replay(pool_file, output=None, predictions=None):
    seen = set()
    chosen = []
    run_file = pool_file.parent / "run.json"
    run = read(run_file) if run_file.is_file() else {}
    for row in rows(pool_file):
        image_id = row["image_id"]
        identity = (row.get("dataset"), row.get("backbone"), image_id)
        if identity in seen:
            raise ValueError(f"Duplicate image/backbone record {identity}")
        seen.add(identity)
        arms = row["choices"] if "choices" in row else {run.get("method", "prediction"): row}
        for method, arm in arms.items():
            if "pool" in arm:
                weights = arm.get("weights", row.get("weights", run.get("weights", DEFAULT)))
                best = select(arm["pool"], weights)
                if best != arm["selected"]:
                    raise ValueError(f"Selection mismatch for {identity}/{method}")
                caption = arm["pool"][best]["caption"]
            else:
                caption = arm["records"][0]["caption"]
            if caption != arm["caption"]:
                raise ValueError(f"Caption mismatch for {identity}/{method}")
            chosen.append(dict(image_id=image_id, method=method, caption=caption))
    if not chosen:
        raise ValueError("Empty candidate file")
    if predictions:
        pred = read(predictions)
        by_id = {r["image_id"]: r["caption"] for r in pred}
        if len(pred) != len(by_id) or len(chosen) != len(by_id):
            raise ValueError("Expected one selected caption per prediction")
        if {r["image_id"]: r["caption"] for r in chosen} != by_id:
            raise ValueError("The independent prediction file differs")
    if output:
        write(output, chosen)
    return len(chosen), len(seen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pools", type=Path, default=ROOT / "examples/pools.jsonl")
    p.add_argument("--output", type=Path)
    p.add_argument("--predictions", type=Path, help="Independent prediction file for a single-arm run")
    a = p.parse_args()
    choices, images = replay(a.pools, a.output, a.predictions)
    print(f"Verified {choices:,} selections in {images:,} image/backbone records.")


if __name__ == "__main__":
    main()
