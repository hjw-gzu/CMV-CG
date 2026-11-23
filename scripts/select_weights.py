"""Recover the fixed in-domain weights from the two released validation grids."""
import csv
from pathlib import Path

GRID = Path(__file__).resolve().parents[1] / "results/validation"
PREVIOUS = (0.45, 0.25, 0.20, 0.10)
KEYS = ("w_original", "w_multiview", "w_language", "w_consensus")


def load_grid(dataset):
    with (GRID / f"{dataset}-grid.csv").open(encoding="utf-8-sig") as stream:
        values = {tuple(float(row[k]) for k in KEYS): float(row["cider_x100"])
                  for row in csv.DictReader(stream)}
    ordered = sorted(values, key=values.get, reverse=True)
    return values, {w: i for i, w in enumerate(ordered, 1)}


def main():
    coco, c_rank = load_grid("coco")
    flickr, f_rank = load_grid("flickr")
    if coco.keys() != flickr.keys() or len(coco) != 263:
        raise ValueError("Expected the same 263 validation settings in both datasets")
    eligible = [w for w in coco if w[0] > max(w[1:]) and coco[w] > coco[PREVIOUS]
                and flickr[w] > flickr[PREVIOUS]]
    best = min(eligible, key=lambda w: (max(c_rank[w], f_rank[w]), c_rank[w] + f_rank[w], w))
    print(f"Weights {best}, MSCOCO rank {c_rank[best]}/263, Flickr30k rank {f_rank[best]}/263")
    return best


if __name__ == "__main__":
    main()
