# CMV-CG

**Controlled Multi-View Candidate Generation for Training-Free Image Captioning**  
Jingwen Huang

A frozen captioner runs beam search independently on the original image and four controlled views. After deduplication, a reference-free selector chooses one complete caption. This repository includes Diffusion Bridge and supervised BLIP inference, the image splits, experiment settings, archived candidates and selected captions, and code for the comparisons in the paper.

## What can be reproduced

| Starting point | Available runs |
| --- | --- |
| This checkout, Python only | Replay 36,000 archived selections across two backbones and both directions of dataset transfer. Verify the three motivation-figure captions and recover the shared weights from 263 validation configurations. |
| Released test features, captioner weights and CLIP | Decode the 5,000 MSCOCO and 1,000 Flickr30k Karpathy test images from their five stored bridge endpoints. Compute evaluation metrics with the public references. |
| Flickr30k weights and original images | Start before the image encoder, construct the five views, run the original diffusion bridge and decode. The source-trained Flickr model can also be applied to MSCOCO. |
| Public BLIP weights and original images | Run single-view and multi-view BLIP on either dataset. |
| New images with the MSCOCO Diffusion Bridge | Requires the original COCO 3M-step checkpoint, `model-300.pt`. Its SHA-256 is published in `configs/assets.json`. The release includes the five test features for every in-domain image. Original COCO-source transfer and nocaps selections/predictions can be evaluated from the archive. |

## Quick check

From the repository root, the first two commands use only the Python standard library.

```bash
python scripts/replay.py
python scripts/verify_results.py
```

The full check compares the reselected caption with an independent prediction file for each archived arm. The result should end in **36,000 exact selections**. To inspect the experiment values directly, open [metrics-x100.csv](results/metrics-x100.csv) and [the table index](docs/experiments.md).

## Installation for inference

The recorded GPU runs used Linux and Python 3.12.3. Create an environment and install the original CUDA build of PyTorch, followed by the dependencies. The code also supports a CPU for a small decoding example.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "setuptools<81"
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For CPU, use `https://download.pytorch.org/whl/cpu` as the PyTorch index. On Windows use `.venv\Scripts\Activate.ps1` in PowerShell. Run commands from this checkout. PTB tokenization, METEOR and SPICE also require Java. The original COCO-caption evaluation stack expects Java 8; [evaluation and data preparation](docs/data-and-models.md) covers its additional SPICE resources.

## Set up weights and features

Download the five ZIP files from [GitHub Releases](https://github.com/hjw-gzu/CMV-CG/releases). The filenames and SHA-256 hashes are in [the asset manifest](configs/asset-packs.json). Place the ZIP files in one directory, for example `release-assets/`. Source ZIP downloads from GitHub and individual model downloads are separate. Install the local files and fetch upstream CLIP and the verified CVPR bridge implementation:

```bash
python scripts/install_assets.py --archives release-assets --download --assets assets
```

An offline installation accepts existing copies of the OpenAI CLIP archive and the CVPR bridge source. The installer checks them against the same hashes used for the online installation.

```bash
python scripts/install_assets.py --archives release-assets --assets assets --clip-file /path/to/ViT-B-32.pt --bridge-source /path/to/denoising_diffusion_pytorch_1d.py
```

For Flickr alone add `--dataset flickr`. Add `--blip` to download the pinned Salesforce checkpoint and processor files. For existing BLIP files use `--blip-dir /path/to/blip-image-captioning-base`. The installer checks SHA-256 before accepting every asset. To supply the original COCO bridge later:

```bash
python scripts/install_assets.py --assets assets --coco-bridge /path/to/model-300.pt
```

Both bridges were trained for **3,000,000 steps**. COCO saved every 10,000 steps (milestone 300), Flickr every 50,000 (milestone 60). Use the filenames `coco/model-300.pt` and `flickr/model-60.pt`. Dataset-specific checkpoint hashes, schedules, captioner hashes and the exact view transforms are given in [data and models](docs/data-and-models.md).

## Decode one image

```bash
python -m cmv_cg --source flickr --dataset flickr --protocol cached --image-id 13990 --assets assets --device cpu --save-pools --output runs/flickr-example
python scripts/replay.py --pools runs/flickr-example/pools.jsonl --predictions runs/flickr-example/predictions.json
```

On the verified CPU run, this produces **“A dog is jumping over a hurdle in a dog show.”** This prediction belongs to the cached test protocol. The motivation figure uses its own beam-allocation comparison and its own selector, recorded in [the example](examples/motivation.json).

Run the COCO single-image example with `--source coco --dataset coco --protocol cached --image-id 0`. The tested CPU output is **“A man riding a motorcycle down a dirt road.”**

## Run an entire dataset

Each command writes to a new output directory. Remove `--image-id` to process every image in the split. `--save-pools` also records beams, source views and scores. For COCO, supplying the photos to the cached run enables visual rescoring if a software or precision difference generates a caption beyond the archived score list.

```bash
python scripts/prepare_data.py --dataset coco --split test --annotations data/local/dataset_coco.json --output runs/labels-coco
python -m cmv_cg --source coco --dataset coco --protocol cached --images data/local/val2014 --assets assets --output runs/coco-main
python scripts/evaluate.py --labels runs/labels-coco/labels.json --predictions runs/coco-main/predictions.json --output runs/coco-main/metrics.json
```

For the Flickr pixel-to-caption comparison:

```bash
python -m cmv_cg --source flickr --dataset flickr --protocol per-view --method multi --images data/local/flickr30k-images --output runs/flickr-five
python -m cmv_cg --source flickr --dataset flickr --protocol per-view --method single --beam 20 --images data/local/flickr30k-images --output runs/flickr-single
```

[Experiment recipes](docs/experiments.md) cover the main paper, external evaluation, baselines, bootstrap intervals, efficiency, and supplementary analyses. [Verification](docs/verification.md) records the numerical checks and hardware used to test this release.

## Citation and credits

Use the paper title and author above to cite the work. Machine-readable citation metadata is in [CITATION.cff](CITATION.cff). CMV-CG builds on [Diffusion Bridge](https://github.com/mongeoroo/diffusion-bridge), [OpenAI CLIP](https://github.com/openai/CLIP), [BLIP](https://github.com/salesforce/BLIP), and [COCO caption evaluation](https://github.com/tylin/coco-caption). See [the license](LICENSE) and [third-party notices](THIRD_PARTY.md) for the scope of the release.

