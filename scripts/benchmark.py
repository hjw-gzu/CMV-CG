"""Repeat paired token-aggregation or BLIP timings with the original image/method order."""
import argparse
from pathlib import Path
import os
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmv_cg.io import ROOT, read, verified, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("coco", "flickr"), required=True)
    p.add_argument("--backbone", choices=("bridge", "blip"), default="bridge")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=Path("assets"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from PIL import Image
    from cmv_cg.engine import Captioner
    if not torch.cuda.is_available():
        p.error("This measures GPU latency on original photos")
    if a.output.exists():
        p.error("Use a new output path")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1234)
    model = Captioner(a.assets, a.dataset, a.backbone, "cuda")
    manifest = {r["image_id"]: r for r in read(ROOT / f"data/splits/{a.dataset}-test.json")}
    record = "token32" if a.backbone == "bridge" else "blip-test_timing"
    order = read(ROOT / f"results/timing/{a.dataset}-{record}.json")["measurements"]
    comparison = "token_32" if a.backbone == "bridge" else "single20"
    methods = {"ours": ("multi", 4), "token_32": ("token", 32), "single20": ("single", 20)}
    weights = model.config["in_domain_weights"]

    def measure(iid, arm):
        row = manifest[iid]
        method, beam = methods[arm]
        torch.cuda.synchronize()
        start = time.perf_counter()
        path = a.images / row["file_name"]
        if not path.is_file() and row["file_name"].startswith("COCO_"):
            path = a.images / "val2014" / row["file_name"]
        verified(path, row["sha256"])
        with Image.open(path) as src:
            photo = src.convert("RGB")
        model.predict(iid, photo, "test", method, beam, weights)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    warmup_ids = list(dict.fromkeys(r["image_id"] for r in order))[:2]
    for iid in warmup_ids:
        for method in ("ours", comparison):
            measure(iid, method)
    samples = [dict(repeat=r["repeat"], image_id=r["image_id"], method=r["method"],
                    seconds=measure(r["image_id"], r["method"])) for r in order]
    medians = {m: statistics.median(r["seconds"] for r in samples if r["method"] == m)
               for m in ("ours", comparison)}
    means = {m: statistics.mean(r["seconds"] for r in samples if r["method"] == m)
             for m in ("ours", comparison)}
    write(a.output, dict(backbone=a.backbone, comparison=comparison, mean_seconds=means, device=torch.cuda.get_device_name(0), torch=torch.__version__,
                         median_seconds=medians, median_relative_gap=medians[comparison] / medians["ours"] - 1,
                         measurements=samples))
    print(medians)


if __name__ == "__main__":
    main()
