"""Frozen model construction from checksum-verified local assets."""
import importlib.util
from pathlib import Path
import sys
import types

from .io import ROOT, read, verified


def load_clip(root, device):
    import clip
    cfg = read(ROOT / "configs/public-models.json")["clip"]
    model, preprocess = clip.load(str(verified(root / "clip/ViT-B-32.pt", cfg["sha256"])),
                                  device=device, jit=False)
    return model.eval().requires_grad_(False), preprocess


def load_decoder(root, dataset, device):
    from .decoder import load_caption
    pins = read(ROOT / "configs/assets.json")[dataset]
    verified(root / dataset / "decoder.ckpt", pins["decoder"]["sha256"])
    for name, expected in read(ROOT / "configs/public-models.json")["gpt2"]["files"].items():
        verified(root / "gpt2" / name, expected)
    return load_caption(ROOT / "configs/decoder.yaml", root / "gpt2", root / dataset / "decoder.ckpt", device)[:2]


def load_bridge(root, dataset, device):
    import torch
    pin = read(ROOT / "configs/assets.json")[dataset]
    checkpoint = verified(root / dataset / pin["bridge_name"], pin["original"]["bridge"])
    folder = root / "upstream/denoising_diffusion_pytorch"
    source = verified(folder / "denoising_diffusion_pytorch_1d.py", pin["bridge_source"])
    # The original file imports only the version constant from its package.
    package = types.ModuleType("denoising_diffusion_pytorch")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    version = types.ModuleType("denoising_diffusion_pytorch.version")
    version.__version__ = "2.0.17"
    sys.modules[version.__name__] = version
    name = "denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    denoiser = module.Unet1D(dim=512, init_dim=32, dim_mults=(1, 2, 4, 8), channels=1)
    model = module.GaussianDiffusion1D_norm(model=denoiser, seq_length=512, timesteps=1000,
                                           objective="pred_x0", sampling_timesteps=5)
    # The archived trainer payload also serializes optimizer metadata. Only model is used.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_blip(root, device):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    p = read(ROOT / "configs/blip.json")
    folder = root / "blip"
    for f, h in {**p["small_files"], p["weights"]["name"]: p["weights"]["sha256"]}.items():
        verified(folder / f, h)
    processor = BlipProcessor.from_pretrained(str(folder), local_files_only=True, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(str(folder), local_files_only=True,
                      weights_only=True, attn_implementation="eager").float().to(device)
    return model.eval().requires_grad_(False), processor

