# Data and frozen models

## Datasets and identifiers

| Dataset | Download and split | Validation / test used here |
| --- | --- | --- |
| MSCOCO | Download the original 2014 validation photos from the [COCO project](https://cocodataset.org/#download). Obtain `dataset_coco.json` from the [Karpathy caption splits](https://cs.stanford.edu/people/karpathy/deepimagesent/). | 5,000 / 5,000 |
| Flickr30k | Obtain the images from the [Flickr30k project](https://shannon.cs.illinois.edu/DenotationGraph/) and `dataset_flickr30k.json` from the [Karpathy caption splits](https://cs.stanford.edu/people/karpathy/deepimagesent/). | 1,014 / 1,000 |
| nocaps | Use the public validation annotations and the matching Open Images photographs from the [official download page](https://nocaps.org/download). The validation JSON contains the image filenames. | 4,500 validation |

Karpathy `imgid` is the prediction/evaluation ID in this project. In MSCOCO it differs from the numeric suffix in the original filename. Use the supplied manifests, which pair each filename with an ID and an image SHA-256. The 5,000 MSCOCO test files contain 25,010 reference captions. Flickr30k test has 5,000 and nocaps validation has 45,000. Human captions enter evaluation and the post-hoc oracle. Generation and test selection use only the image, candidate captions and model outputs.

```bash
python scripts/prepare_data.py --dataset flickr --split test --annotations data/local/dataset_flickr30k.json --output runs/labels-flickr
python scripts/prepare_data.py --dataset nocaps --split val --annotations data/local/nocaps_val_4500_captions.json --output runs/labels-nocaps
```

For Flickr provide the directory containing the JPEGs. For MSCOCO either pass the JPEG directory or a parent containing `val2014/`. For nocaps put the filenames from the manifest directly under the image directory.

## Pretrained assets

| Model | File / model identity | Settings |
| --- | --- | --- |
| CLIP | OpenAI ViT-B/32 JIT archive, `40d36571…` | Frozen image encoder for Diffusion Bridge and frozen image/text encoders for selection. |
| COCO bridge | `model-300.pt`, `caf23089…` | 1,000-step diffusion schedule, five DDIM steps starting at 600, training milestone 300 × 10,000. Released in-domain inputs are the cached endpoints. |
| Flickr30k bridge | `model-60.pt`, `89dc8163…` | Five DDIM steps starting at 400, training milestone 60 × 50,000. The checkpoint records step 3,000,000. |
| COCO / Flickr30k captioners | Author-reproduced `meteor=0.3630.ckpt` / `meteor=0.3274.ckpt` | GPT-2 with learned MLP prefix, ten prefix tokens, CLIP width 512. Each tensor-only export contains the full trained language model and mapper. |
| BLIP | [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base/tree/82a37760796d32b1411fe092ab5d4e227313294b) | Paired-data supervised captioner. BOS-only input, stop on generated SEP (102), eager attention, search in float32. |

The bridge uses the `model` weights in the original training checkpoint. CLIP image features are normalized, centered on the **source dataset's** stored image mean, normalized again and scaled by five. The bridge outputs are normalized before prefix decoding. A frozen captioner requires no CMV-CG training step or optimizer state. The original [CVPR training code](https://github.com/mongeoroo/diffusion-bridge) describes training the upstream models.

Take the original view, rotate +5° and −10° for the COCO source or +5° and −5° for the Flickr source, center-crop to 80% then resize, and shrink to 80% on a black canvas. Transforms preserve the original canvas dimensions and use bicubic sampling. In `historical-batch` the encoder processes the historical transform superset; the five selected views enter the bridge together. Later token and BLIP control runs use independent per-view image encoding. Keep the specified protocol when comparing rounded table values.

## Official evaluation

For Linux, install the dependencies in the root README, Java 8, and the Stanford CoreNLP 3.6.0 resources required by SPICE. The [Python 3 COCO-caption implementation](https://github.com/salaniz/pycocoevalcap) documents the Java dependency and resource installation. The SPICE wrapper needs both `stanford-corenlp-3.6.0.jar` and `stanford-corenlp-3.6.0-models.jar` in its `spice/lib/` directory. When running offline, install the official resources in advance. An existing compatible checkout can be used with `--eval-root /path/to/coco-caption`.

```bash
python scripts/evaluate.py --labels runs/labels-flickr/labels.json --predictions results/predictions/per-view-flickr-db-ours.json.gz --output runs/flickr-official.json
python scripts/cider.py --labels runs/labels-flickr/labels.json --predictions results/predictions/per-view-flickr-db-ours.json.gz
```

The full evaluator outputs the eight native COCOEvalCap scores. Multiply by 100 for the paper tables. The focused CIDEr and oracle command already reports the paper scale. All scores are calculated over the complete target split unless you explicitly prepare matching references for a subset.


For modern Java or Windows, add `--portable` to `scripts/evaluate.py`. It uses the same metric implementations with a Java-compatible SPICE output writer and verifies that all eight metrics completed. Set TEMP/TMP (or TMPDIR on Linux) to a volume with free space for the evaluation. The scorer directory must contain the installed Java resources before the run.

## A new photograph

For a Flickr-source pixel run, put the image in a directory and give it an integer ID in a small JSON manifest. For example, save `[{"image_id": 1, "file_name": "photo.jpg"}]` as `my-image.json`. The ID is used by the deterministic sampling seed. Each row may include an expected SHA-256. References are required only when computing evaluation metrics.

```bash
python -m cmv_cg --source flickr --dataset flickr --protocol per-view --manifest my-image.json --images ./photos --output runs/new-photo
```
