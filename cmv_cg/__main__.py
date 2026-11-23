"""Generate a caption file that can be passed directly to COCOEvalCap."""
import argparse
import json
from importlib.metadata import version
import os
from pathlib import Path
import tempfile
from .io import ROOT, read, verified, write

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("coco","flickr"), required=True)
    p.add_argument("--dataset", choices=("coco","flickr","nocaps"), required=True)
    p.add_argument("--assets", type=Path, default=Path("assets"))
    p.add_argument("--images", type=Path, help="Directory containing the filenames in the split manifest")
    p.add_argument("--split", choices=("test","val"), default="test")
    p.add_argument("--manifest", type=Path, help="Override the released split manifest (e.g. nocaps)")
    p.add_argument("--backbone", choices=("bridge","blip"), default="bridge")
    p.add_argument("--protocol", choices=("cached","per-view","historical-batch"), default="per-view")
    p.add_argument("--method", choices=("greedy","single","multi","token"), default="multi")
    p.add_argument("--beam", type=int)
    p.add_argument("--probability", choices=("source","original"), default="source")
    p.add_argument("--drop-view", choices=("original","rotate_left","rotate_right","zoom_in","zoom_out_black"))
    p.add_argument("--angle-scale", type=float, choices=(.5,1,2), default=1)
    p.add_argument("--input-rotation", type=int, choices=(-20,-10,-5,0,5,10,20), default=0)
    p.add_argument("--weights", type=float, nargs=4, metavar=("ORIG","VIEW","PROB","CONS"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--image-id", type=int, action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4, help="CPU threads, including CPU smoke runs")
    p.add_argument("--save-pools", action="store_true")
    p.add_argument("--output", type=Path, required=True, help="A new run directory")
    ns=p.parse_args()
    if ns.method=="greedy" and ns.beam not in (1,None):
        p.error("Greedy decoding uses one beam.")
    beam=ns.beam if ns.beam is not None else {"multi":4,"single":20,"greedy":1,"token":32}[ns.method]
    if beam<=0 or (ns.limit is not None and ns.limit<=0) or ns.threads<=0:
        p.error("Beam, limit and threads must be positive.")
    if ns.drop_view and ns.method!="multi":
        p.error("--drop-view is a multi-view generation ablation.")
    if ns.protocol=="cached" and (ns.dataset!=ns.source or ns.split!="test" or ns.input_rotation or ns.angle_scale!=1 or ns.manifest):
        p.error("Archived features bind the nominal in-domain test images and transforms.")
    if ns.protocol!="cached" and ns.images is None:
        p.error("Pixel inference requires --images.")
    if ns.dataset=="nocaps" and (ns.source!="coco" or ns.split!="val" or not ns.manifest):
        p.error("Use the COCO source, --split val and --manifest for nocaps.")
    if ns.backbone=="blip" and (ns.protocol!="per-view" or ns.method=="token" or ns.probability=="original"):
        p.error("BLIP experiments use per-view encoding with source-view probabilities.")
    if ns.probability=="original" and ns.method=="greedy":
        p.error("Original-image rescoring applies to a pool of candidates.")
    import torch
    from PIL import Image
    from .engine import Captioner
    if ns.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable. Use --device cpu for a small functional run.")
    torch.set_num_threads(ns.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(ns.seed)
    config=read(ROOT / f"configs/{ns.source}.json")
    weights=ns.weights or config["transfer_weights" if ns.dataset not in (ns.source,"nocaps") else "in_domain_weights"]
    listing=read(ns.manifest or ROOT / f"data/splits/{ns.dataset}-{ns.split}.json")
    ids={r["image_id"] for r in listing}
    if len(ids)!=len(listing) or (ns.image_id and not set(ns.image_id)<=ids):
        p.error("Duplicate or unknown image IDs in requested split.")
    rows=[r for r in listing if not ns.image_id or r["image_id"] in ns.image_id][:ns.limit]
    if not rows:
        p.error("No images selected.")
    if ns.output.exists():
        p.error("Use a new output directory.")
    runner=Captioner(ns.assets,ns.source,ns.backbone,ns.device,ns.seed,ns.protocol=="cached")
    ns.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cmvcg-",dir=ns.output.parent) as tmp:
        stage=Path(tmp)/"results"
        stage.mkdir()
        predictions=[]
        pool_file=(stage/"pools.jsonl").open("x",encoding="utf-8") if ns.save_pools else None
        try:
            for row in rows:
                iid=row["image_id"]
                image=None
                if ns.images:
                    path=ns.images / row["file_name"]
                    if not path.is_file() and row.get("subdir"):
                        path=ns.images / row["subdir"] / row["file_name"]
                    if not path.is_file() and row["file_name"].startswith("COCO_"):
                        path=ns.images / row["file_name"].split("_")[1] / row["file_name"]
                    if row.get("sha256"):
                        verified(path,row["sha256"])
                    with Image.open(path) as src:
                        image=src.convert("RGB")
                    if ns.input_rotation:
                        image=image.rotate(ns.input_rotation,Image.Resampling.BICUBIC,expand=False,fillcolor=(0,0,0))
                result=runner.predict(iid,image,ns.split,ns.method,beam,weights,ns.probability,ns.drop_view,
                                      ns.angle_scale,ns.protocol,ns.save_pools)
                predictions.append(dict(image_id=iid,caption=result["caption"]))
                if pool_file:
                    pool_file.write(json.dumps(dict(image_id=iid,**result),allow_nan=False)+"\n")
                if len(rows)<=5 or len(predictions)%100==0:
                    print(f"{len(predictions)}/{len(rows)}  {iid}  {result['caption']}",flush=True)
        finally:
            if pool_file:pool_file.close()
        write(stage/"predictions.json",predictions)
        write(stage/"run.json",dict(source=ns.source,dataset=ns.dataset,split=ns.split,method=ns.method,
              beam=beam,backbone=ns.backbone,device=str(runner.device),protocol=ns.protocol,weights=weights,
              probability=ns.probability,seed=ns.seed,drop_view=ns.drop_view,angle_scale=ns.angle_scale,
              input_rotation=ns.input_rotation,images=len(predictions),torch=torch.__version__,transformers=version("transformers")))
        stage.rename(ns.output)
    print(ns.output / "predictions.json")


if __name__=="__main__":
    main()

