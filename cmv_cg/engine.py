"""Independent view decoding, native token aggregation, and shared selection."""
import gzip
import hashlib
import json
import math
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from . import models
from .aggregation import token_candidates
from .blip import blip_search
from .decoder import _deduplicate, beam_candidates, score_pool
from .io import ROOT, read, verified
from .likelihood import likelihoods
from .views import ROLES, transformed


class Captioner:
    def __init__(self, root, source, backbone, device, seed=1234, cached=False):
        self.root, self.source, self.backbone = root, source, backbone
        self.device, self.seed = torch.device(device), seed
        self.config = read(ROOT / "configs" / f"{source}.json")
        self.pin = read(ROOT / "configs/assets.json")[source]
        self.clip, self.pre = models.load_clip(root, self.device)
        self.bank = None
        self.alignments = None
        if backbone == "blip":
            if cached:
                raise ValueError("Cached bridge endpoints are specific to Diffusion Bridge.")
            self.model, self.processor = models.load_blip(root, self.device)
        else:
            self.model, self.tokenizer = models.load_decoder(root, source, self.device)
            if cached:
                filename = f"features/{source}-test.npz"
                pack = read(ROOT / "configs/asset-packs.json")[f"cmvcg-{source}-test-features.zip"]
                with np.load(verified(root / filename, pack["files"][filename]), allow_pickle=False) as a:
                    self.bank = {k: a[k] for k in a.files}
                self.by_id = {int(i): n for n, i in enumerate(self.bank["image_ids"])}
                if "raw" not in self.bank:
                    with gzip.open(ROOT / "data/coco-test-alignment.json.gz", "rt", encoding="utf-8") as f:
                        self.alignments = json.load(f)
            else:
                self.bridge = models.load_bridge(root, source, self.device)
                filename = f"{source}/image_mean.npy"
                pack = read(ROOT / "configs/asset-packs.json")[f"cmvcg-{source}-captioner.zip"]
                mean = np.load(verified(root / filename, pack["files"][filename]), allow_pickle=False)
                self.mean = torch.as_tensor(mean, device=self.device).float().reshape(1, 512)

    def encode(self, image_id, image, split, strength, mode, generation_roles):
        if self.bank is not None:
            i = self.by_id[image_id]
            raw = torch.from_numpy(self.bank["raw"][i]).to(self.device) if "raw" in self.bank else None
            bank = torch.from_numpy(self.bank["bridged"][i]).to(self.device)
            return raw, {role:bank[j] for j,role in enumerate(ROLES)}
        angles = [v * strength for v in self.config["rotations"]]
        images = [transformed(image, r, angles, self.config["zoom_ratio"]) for r in ROLES]
        if mode == "historical-batch" and self.backbone == "bridge":
            if strength != 1:
                raise ValueError("Historical image batching is defined for the nominal rotation strength.")
            extras = ([5,8,10,-5,-8,-10] if self.source == "flickr" else [5,-5,-10])
            batch = [images[0]] + [image.rotate(a, Image.Resampling.BICUBIC, expand=False,
                            fillcolor=(0,0,0)) for a in extras] + images[3:]
            inputs = torch.stack([self.pre(v) for v in batch]).to(self.device)
            if self.source == "coco":
                encoded = torch.cat([self.clip.encode_image(inputs[:1]), self.clip.encode_image(inputs[1:])])
            else:
                encoded = self.clip.encode_image(inputs)
            indices = [0, extras.index(angles[0])+1, extras.index(angles[1])+1, len(batch)-2, len(batch)-1]
            raw = encoded.float().half().float()[indices]
            x = F.normalize(F.normalize(raw, dim=-1)-self.mean,dim=-1)*5
            feat = self.bridge.ddim_sample_with_img(x.unsqueeze(1),inference_step=self.config["ddim_start"])
            bank = F.normalize(feat.squeeze(1),dim=-1)
            return raw, dict(zip(ROLES, bank))
        raw, features = [], {}
        for role, view in zip(ROLES, images):
            raw.append(self.clip.encode_image(self.pre(view).unsqueeze(0).to(self.device)).float().half().float()[0])
            if role not in generation_roles:
                continue
            if self.backbone == "blip":
                pixels = self.processor(images=view,return_tensors="pt")["pixel_values"].to(self.device)
                features[role] = self.model.vision_model(pixel_values=pixels,return_dict=True).last_hidden_state
            else:
                v = int.from_bytes(hashlib.sha256(f"ablation-v1:{self.seed}:{self.source}:{split}:{image_id}:{role}".encode()).digest()[:4],"little")
                torch.manual_seed(v)
                x = F.normalize(F.normalize(raw[-1][None],dim=-1)-self.mean,dim=-1)*5
                feat = self.bridge.ddim_sample_with_img(x.unsqueeze(1),inference_step=self.config["ddim_start"])
                features[role] = F.normalize(feat.squeeze(1),dim=-1)[0]
        return torch.stack(raw), features

    def beam(self, feature, role, beam):
        if self.backbone == "bridge":
            return beam_candidates(self.model,self.tokenizer,feature,role,beam_size=beam)
        items = blip_search(self.model,feature,beam)
        return [dict(c,caption=self.processor.tokenizer.decode(c["tokens"],skip_special_tokens=True).strip(),
                     source=role) for c in items]

    @torch.inference_mode()
    def predict(self, image_id, image, split, method, beam, weights, likelihood="source", drop=None,
                strength=1, mode="per-view", return_pool=False):
        roles = list(ROLES) if method in ("multi","token") else ["original"]
        if drop:
            roles.remove(drop)
        required = set(roles)
        if likelihood == "original":
            required.add("original")
        raw, features = self.encode(image_id,image,split,strength,mode,required)
        if method == "token":
            if self.backbone != "bridge":
                raise ValueError("The token-aggregation comparison uses Diffusion Bridge.")
            records = token_candidates(self.model,self.tokenizer,torch.stack([features[r] for r in roles]),beam)
            if likelihood == "source":
                return {"caption":records[0]["caption"], **({"records":records} if return_pool else {})}
        else:
            records = [c for r in roles for c in self.beam(features[r],r,beam)]
        if method == "greedy":
            return {"caption":records[0]["caption"], **({"records":records} if return_pool else {})}
        pool = _deduplicate(records)
        if likelihood == "original":
            if self.backbone != "bridge":
                raise ValueError("Original-image teacher forcing is a Diffusion Bridge control.")
            vals = likelihoods(self.model,self.tokenizer,features["original"].reshape(1,-1),pool)
            pool = [dict(c,source_mean_logprob=c["mean_logprob"],mean_logprob=p["mean_logprob"]) for c,p in zip(pool,vals)]
        vis = None
        if raw is None:
            lookup = self.alignments[str(image_id)]
            if all(c["caption"] in lookup for c in pool):
                vis = [lookup[c["caption"]] for c in pool]
            elif image is not None:
                views = [transformed(image,r,self.config["rotations"]) for r in ROLES]
                raw = torch.stack([self.clip.encode_image(self.pre(v)[None].to(self.device)).float().half().float()[0] for v in views])
            else:
                raise ValueError("The cached COCO visual scores cover archived captions. Supply --images for new captions.")
        scored, best, _ = score_pool(pool,self.clip,raw,weights,visual_scores=vis)
        return {"caption":scored[best]["caption"], **({"records":records,"pool":scored,"selected":best} if return_pool else {})}

