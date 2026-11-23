# Reproducing the experiments

All table files below preserve the panel labels and the manuscript's rounding. They are reporting targets. Independent inference, candidate replay, reference evaluation and statistical analysis have separate entry points.

| Paper item | Inputs and run |
| --- | --- |
| [Table 1](../results/paper-tables/table-1.csv) | Reproduced greedy vs CMV-CG. Use `--protocol cached --method greedy` and `--protocol cached --method multi --beam 4` for each source on its own test split. Evaluate the prediction files against the full corresponding references. The in-domain five-view CIDEr/SPICE targets are **99.07/19.21 on MSCOCO and 61.13/16.30 on Flickr30k**. |
| [Table 2](../results/paper-tables/table-2.csv) | Reverse source and target for dataset transfer. The source controls the bridge, rotations and historical weights. Full archived predictions and candidate components for both directions are under `results/predictions/transfer-*` and `results/pools/*_to_*`. For nocaps, use the COCO source, validation manifest and the archive of 4,500 predictions for each arm. |
| [Table 3](../results/paper-tables/table-3.csv) | Later Diffusion Bridge controls use `--protocol per-view`. Run native token aggregation at beams 1, 8 and 32, single-view beam 20, and independent five-view beam 4. Their results are `per-view-*-db-*`. For panel (e) rescore the three pools with original-image teacher forcing (`--probability original`). Archived predictions have prefix `teacher-forced-`. |
| [Table 4](../results/paper-tables/table-4.csv) | Pool counts, reference-based CIDEr oracle and selected captions from the preceding DB and BLIP comparisons. Full and count-matched statistics use distinct estimands. Pair records by image ID. |
| [Table S1](../results/paper-tables/table-S1.csv) | Use `--backbone blip --protocol per-view` with `--method single --beam 20` and `--method multi --beam 4`. Research targets and full prediction files use `per-view-*-blip-*`. Timings are hardware-dependent. |
| [Table S2](../results/paper-tables/table-S2.csv) and Fig. 3 | Compare single-view beams 4, 10 and 20 vs five-view beam 4; compare the same candidate pools under nine fixed selectors. Full unrounded scores are in `results/ablation/*-selector-grid.csv`. The test grid preserves each arm's original protocol. |
| [Table S3](../results/paper-tables/table-S3.csv), Fig. S2 | Omit exactly one named generation role with `--drop-view`; visual scoring still sees all five views. Set `--angle-scale` to 0.5, 1 or 2 for auxiliary rotation strength. Source scores are in `results/ablation/` and `results/figures/`. |
| [Table S4](../results/paper-tables/table-S4.csv), Figs. S3–S4 | Change selector components or the generation-probability source. The one-weight perturbation figure uses **validation only**. The input-rotation figure uses `--input-rotation` from −20 to +20 degrees, then applies the method's own five transforms. The curves use a fixed selector or remove its original-image term. |
| [Table S5](../results/paper-tables/table-S5.csv), Figs. S5–S7 | Stage timing, near-budget single-view widths 60 on COCO and 80 on Flickr, and paired per-image CIDEr. Timing data retain the original hardware and measured seconds. The near-budget image-paired measurements and metric differences are in `results/figures/paired_latency.csv` and `quality_values.csv`. |

**Which results to compare.** Table 1 uses the historical five post-bridge embeddings. Tables 3–4 use later independently encoded five-view comparisons. The later Diffusion Bridge five-view CIDEr values are **99.108578** and **61.236617**. Comparing these to Table 1's 99.07 and 61.13 would mix experiments.

## Generation and scoring

The four weights are ordered as original-image CLIP alignment, mean view alignment, generation probability, and caption consensus. Generation probability is the geometric mean token probability of the retained generating beam. A duplicate caption is identified by lowercased stripped text; the highest generating mean log probability is retained in stable order. Each score is min–max normalized over the image's deduplicated pool. Ranges at most 1e−12 contribute zero. Ties go to the earliest candidate.

| Use | Weights |
| --- | --- |
| Both in-domain splits and nocaps | 0.35, 0.25, 0.20, 0.20 |
| MSCOCO source evaluated on Flickr30k | 0.45, 0.10, 0.40, 0.05 |
| Flickr30k source evaluated on MSCOCO | 0.35, 0.20, 0.15, 0.30 |
| Fig. 1(a), Fig. 1(b,c) | Probability alone for (a); 0.50 original alignment plus 0.50 probability for (b) and (c). |

```bash
python scripts/select_weights.py
python -m cmv_cg --dataset flickr --source flickr --protocol per-view --method token --beam 32 --images data/local/flickr30k-images --output runs/flickr-token32
python -m cmv_cg --dataset flickr --source flickr --protocol per-view --method token --beam 32 --probability original --images data/local/flickr30k-images --output runs/flickr-token32-common
python -m cmv_cg --dataset coco --source flickr --protocol historical-batch --method multi --images data/local/val2014 --output runs/flickr-to-coco
python scripts/replay.py --pools results/pools/coco_to_flickr.jsonl.gz
python scripts/evaluate.py --labels runs/labels-nocaps/labels.json --predictions results/predictions/external-nocaps-db-multiview.json.gz --output runs/nocaps-official.json
```

For pixel inference on nocaps, run `--source coco --dataset nocaps --split val --manifest data/splits/nocaps-val.json` with the original COCO bridge. The target-set results can be evaluated from archived predictions without this checkpoint. The transfer records use source-derived CLIP view banks. Pixel regeneration depends on matching image hashes, original encoding batches and the software/device. The published predictions are provided as an independent comparison for both directions.

## Pool analysis and uncertainty

```bash
python scripts/cider.py --labels runs/labels-flickr/labels.json --pools results/pools/flickr-db.jsonl.gz
python scripts/match_counts.py --dataset flickr --backbone db --pools results/pools/flickr-db.jsonl.gz --assets assets --output runs/flickr-matched.jsonl
python scripts/bootstrap.py results/statistics/flickr-matched-counts.npz
python scripts/bootstrap.py results/statistics/coco-shared-selector.npz
```

Each image gets ten matched subsets at the smaller of its two unique candidate counts. Seeds 1001 through 1010 fix the hash priorities. The code preserves the first three raw components and recomputes consensus and normalization within each subset. CIDEr oracle scoring fixes document frequencies on the entire reference split. Bootstrap resampling operates on paired images after averaging the ten repetitions within each image, uses seed 1234 and 10,000 resamples, and applies the family of 16 for Tables 3(e) and 4(c). The two other NPZ files run the same checks for the opposite dataset. Published adjusted intervals and the actual per-image differences are included. For Table 3(d), run the same bootstrap script on the corresponding `*-token-controls.npz`. These original comparisons use seed 20261115 and a separate family of 16. The `*-token-contrasts.json` files retain the published intervals. For Supplementary Table S1, use `*-blip-controls.npz`. Its simultaneous comparison covers two endpoints on two datasets (family of four).

## Timing and figures

```bash
python scripts/benchmark.py --dataset flickr --images data/local/flickr30k-images --assets assets --output runs/flickr-timing.json
python scripts/benchmark.py --dataset flickr --backbone blip --images data/local/flickr30k-images --assets assets --output runs/flickr-timing-blip.json
python scripts/plot_results.py --output runs/figures
python scripts/export_views.py --dataset flickr --image data/local/flickr30k-images/2867026654.jpg --output runs/module-views
```

The token-aggregation/BLIP timing benchmark uses the recorded 100 test images, alternating method order across three repetitions, two warm-up images and CUDA synchronization. It includes file reading, integrity checking, transforms, encoding, bridge, decoding and selection. Use the raw-image protocol and the same device class for timing comparisons. The separate all-image main efficiency and near-budget comparisons time each configuration once after three warm-up images. GPU timings from the paper are retained in `results/timing`; cached-feature generation is unsuitable for measuring the original end-to-end latency.

The plotting script writes data plots from the recorded numbers; publication artwork can differ in typography. The Flickr30k photos are `3474406285.jpg` in Fig. 1 and `2867026654.jpg` in Fig. 2. The eight Fig. 4 examples have their source image IDs, filenames, hashes, single-view captions, five-view captions and human reference in [qualitative.json](../examples/qualitative.json).

## External comparisons

Table 1 distinguishes literature, study, and mixed-source rows. In the mixed ViECap rows, B@1 and ROUGE-L were measured by the study; the other four metrics are from the publication. The [table snapshot](../results/paper-tables/table-1.csv) retains the original source column and the reporting precision. Full prediction files for the study's COCO CLIPRe, DeCap, ESPER and MeaCap-ToT comparisons and Flickr30k CLIPRe comparison are in `results/baselines/`. Use the same evaluation script as for CMV-CG. The toolchain does not train or rerun third-party published systems whose metrics were quoted from the literature.

