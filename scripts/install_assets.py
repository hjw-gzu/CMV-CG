"""Install model assets from verified release archives, local files or upstream hosts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_IMPL = Path("upstream/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py")


def read(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(path, expected, opener):
    if path.exists():
        if path.is_file() and digest(path) == expected:
            return
        raise ValueError(f"Existing file has different contents: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        tmp_path = Path(temporary.name)
    try:
        with opener() as source, tmp_path.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        if digest(tmp_path) != expected:
            raise ValueError(f"SHA-256 mismatch: {path.name}")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def remote(url):
    def open_url():
        request = urllib.request.Request(url, headers={"User-Agent": "CMV-CG"})
        return urllib.request.urlopen(request, timeout=90)
    return open_url


def release_assets(archives, target, datasets):
    for name, info in read("asset-packs.json").items():
        if name != "cmvcg-shared-runtime.zip" and not any(f"-{ds}-" in name for ds in datasets):
            continue
        path = archives / name
        if digest(path) != info["sha256"]:
            raise ValueError(f"Bad archive: {path}")
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != sorted(info["files"]):
                raise ValueError(f"Bad archive members: {name}")
            for member, expected in info["files"].items():
                install(target / member, expected, lambda: archive.open(member))
        print(f"Verified {name}", flush=True)


def upstream(target, local, pin, url):
    opener = remote(url) if local is None else lambda: local.open("rb")
    install(target, pin, opener)
    print(f"Verified {target.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=Path("assets"))
    p.add_argument("--archives", type=Path, help="Directory with release asset ZIP files")
    p.add_argument("--dataset", choices=("coco", "flickr"), nargs="+", default=["coco", "flickr"])
    p.add_argument("--download", action="store_true", help="Fetch CLIP and the pinned bridge source")
    p.add_argument("--blip", action="store_true", help="Also download the pinned BLIP model; implies --download")
    p.add_argument("--clip-file", type=Path, help="Local original OpenAI ViT-B-32.pt")
    p.add_argument("--bridge-source", type=Path, help="Local exact upstream denoising_diffusion_pytorch_1d.py")
    p.add_argument("--blip-dir", type=Path, help="Local directory containing the pinned Salesforce checkpoint and processor files")
    p.add_argument("--coco-bridge", type=Path, help="Local original COCO model-300.pt for inference from pixels")
    a = p.parse_args()
    fetch = a.download or a.blip
    if not any((a.archives, a.coco_bridge, a.clip_file, a.bridge_source, a.blip_dir, fetch)):
        p.error("Choose a local or upstream source of assets.")
    if a.archives:
        release_assets(a.archives, a.assets, a.dataset)
    if a.coco_bridge:
        pin = read("assets.json")["coco"]
        install(a.assets / "coco" / pin["bridge_name"], pin["original"]["bridge"], lambda: a.coco_bridge.open("rb"))
    public = read("public-models.json")
    if fetch or a.clip_file:
        clip = public["clip"]
        upstream(a.assets / "clip/ViT-B-32.pt", a.clip_file, clip["sha256"], clip["url"])
    if fetch or a.bridge_source:
        spec = public["upstream"]
        url = f"https://raw.githubusercontent.com/mongeoroo/diffusion-bridge/{spec.get('revision', 'master')}/{spec['file']}"
        upstream(a.assets / BRIDGE_IMPL, a.bridge_source, spec["sha256"], url)
    if a.blip or a.blip_dir:
        pin = read("blip.json")
        files = {**pin["small_files"], pin["weights"]["name"]: pin["weights"]["sha256"]}
        for name, expected in files.items():
            url = f"https://huggingface.co/{pin['model_id']}/resolve/{pin['revision']}/{name}"
            local = a.blip_dir / name if a.blip_dir else None
            upstream(a.assets / "blip" / name, local, expected, url)


if __name__ == "__main__":
    main()

