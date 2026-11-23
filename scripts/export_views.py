"""Export the exact paper transformations from a dataset image after verifying its hash."""
import argparse
from pathlib import Path
import sys
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmv_cg.io import ROOT, read, verified
from cmv_cg.views import ROLES, transformed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--dataset", choices=("flickr", "coco"), required=True)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--output", type=Path, required=True, help="New directory for five lossless PNGs")
    a = ap.parse_args()
    manifest = read(ROOT / f"data/splits/{a.dataset}-{a.split}.json")
    row = next((r for r in manifest if r["file_name"] == a.image.name), None)
    if row is None:
        ap.error("Supply an original image with its dataset filename from the specified split")
    verified(a.image, row["sha256"])
    config = read(ROOT / f"configs/{a.dataset}.json")
    a.output.mkdir(parents=True, exist_ok=False)
    with Image.open(a.image) as im:
        for name in ROLES:
            transformed(im, name, config["rotations"], config["zoom_ratio"]).save(a.output / (name+".png"))
    print(a.output)


if __name__ == "__main__":
    main()
