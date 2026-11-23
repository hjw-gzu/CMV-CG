"""Make evaluation labels and image manifests using the dataset's own annotations."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("coco","flickr","nocaps"),required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--split",choices=("val","test"),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    data=load(a.annotations)
    if a.dataset=="nocaps":
        if a.split!="val" or len(data["images"])!=4500:
            p.error("The paper uses all 4,500 public validation images.")
        images=[{"id":int(r["id"]),"file_name":r["file_name"]} for r in data["images"]]
        annotations=data["annotations"]
        manifest=load(ROOT/"data/splits/nocaps-val.json")
        if {r["image_id"]:r["file_name"] for r in manifest}!={r["id"]:r["file_name"] for r in images}:
            raise ValueError("The nocaps IDs or filenames differ from the original validation run.")
    else:
        manifest=load(ROOT/f"data/splits/{a.dataset}-{a.split}.json")
        mapping={r["file_name"]:r["image_id"] for r in manifest}
        images=[];annotations=[]
        for row in data["images"]:
            if row["split"]!=a.split:continue
            name=row["filename"]
            if name not in mapping or int(row["imgid"])!=mapping[name]:
                raise ValueError(f"Unexpected image ID or filename: {name}")
            iid=mapping[name]
            images.append({"id":iid,"file_name":name})
            for s in row["sentences"]:
                annotations.append({"image_id":iid,"id":int(s["sentid"]),"caption":s["raw"]})
    ids=[r["id"] for r in images]
    if len(ids)!=len(set(ids)) or set(ids)!={r["image_id"] for r in manifest}:
        raise ValueError("Image identities differ from the study split.")
    if any(Path(r["file_name"]).name!=r["file_name"] for r in images):
        raise ValueError("Unexpected relative path in filenames.")
    if {r["image_id"] for r in annotations}!=set(ids):
        raise ValueError("Reference captions must cover every image.")
    labels={"info":{},"licenses":data.get("licenses",[]),"type":"captions","images":images,"annotations":annotations}
    a.output.mkdir(parents=True,exist_ok=False)
    for name,v in [("manifest",manifest),("labels",labels)]:
        (a.output/f"{name}.json").write_text(json.dumps(v,ensure_ascii=False),encoding="utf-8")
    print(f"{a.dataset}/{a.split}: {len(ids)} images, {len(annotations)} reference captions")


if __name__=="__main__":
    main()

