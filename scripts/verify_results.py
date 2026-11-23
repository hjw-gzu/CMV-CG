"""Verify stored selections, prediction files and numerical evidence, without a GPU."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmv_cg.io import ROOT, read, rows
from cmv_cg.replay import select
from replay import replay
from select_weights import main as select_weights


def read_predictions(filename):
    with gzip.open(ROOT / filename, "rt", encoding="utf-8") as stream:
        pred = json.load(stream)
    values = {int(r["image_id"]): r["caption"] for r in pred}
    if len(values) != len(pred) or not all(s.strip() for s in values.values()):
        raise ValueError(filename)
    return values


def verify():
    directory = ROOT / "results"
    with (directory / "metrics-x100.csv").open(encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    predictions = {}
    for row in metrics:
        study, ds, backbone, arm = [row[k] for k in ("study", "dataset", "backbone", "method")]
        target = ds.split("_to_")[-1]
        split = "val" if target == "nocaps" else "test"
        ids = {r["image_id"] for r in read(ROOT / f"data/splits/{target}-{split}.json")}
        if len(ids) != int(row["images"]):
            raise ValueError(row)
        if row["predictions"]:
            values = read_predictions(row["predictions"])
            if set(values) != ids:
                raise ValueError(row)
            predictions[study, ds, backbone, arm] = values
    total = 0
    for f in sorted((directory / "pools").glob("*.jsonl.gz")):
        direction = f.name.split(".")[0]
        if "_to_" in direction:
            study, ds, backbone = "transfer", direction, "db"
        else:
            study = "per-view"
            ds, backbone = direction.split("-")
        count, _ = replay(f)
        used = set()
        for r in rows(f):
            image = int(r["image_id"])
            for arm, choice in r["choices"].items():
                if predictions[study, ds, backbone, arm][image] != choice["caption"]:
                    raise ValueError(f"Prediction file mismatch for {direction}/{image}/{arm}")
                used.add(image)
        exemplar = next(arm for (s, d, b, arm) in predictions if (s, d, b) == (study, ds, backbone))
        if used != set(predictions[study, ds, backbone, exemplar]):
            raise ValueError(f"Incomplete {direction} pool file")
        print(f"{direction}: {count:,} selections")
        total += count
    mot = read(ROOT / "examples/motivation.json")
    for name, arm in mot["panels"].items():
        i = select(arm["candidates"], arm["weights"])
        if arm["candidates"][i]["caption"] != arm["caption"]:
            raise ValueError(f"Motivation panel {name}")
    if select_weights() != (.35, .25, .20, .20):
        raise ValueError("Shared validation weights differ")
    print(f"{len(metrics)} metric rows, {len(predictions)} prediction files, {total:,} exact selections")


if __name__ == "__main__":
    verify()
