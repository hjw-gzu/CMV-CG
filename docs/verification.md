# Release verification

The checks below distinguish decoding, replaying selections, and scoring predictions against human references. The release was assembled on a Windows CPU machine. The original GPU experiments and timing values are archived separately.

| Check | Coverage and outcome |
| --- | --- |
| Model exports | All 153 tensor entries in each released GPT-2 captioner matched its original Lightning checkpoint exactly, including the learned prefix mapper. The Flickr30k diffusion checkpoint records training step 3,000,000. The original files and the export files have separate SHA-256 digests. |
| Archived selections | Reselected the captions in all six pool files and compared them by image ID with separately archived predictions. All 36,000 selections matched, including both directions of dataset transfer. |
| Test image identities | Verified SHA-256 for the 5,000 MSCOCO test JPEGs. The split manifests bind each Karpathy ID to its filename. The two photo examples in the README and figure records use their dataset IDs. |
| Main-text examples | Recovered each of the three captions in Fig. 1 from the recorded candidates and the panel's stated selection weights. The qualitative examples include captions and source-image checksums. |
| Validation | The two 263-entry weight grids select the shared weights (0.35, 0.25, 0.20, 0.20), ranked 25th on COCO and 31st on Flickr30k. |
| Official caption scores | From the public test references and the archived later five-view predictions, full-split CIDEr × 100 was 99.108578 on MSCOCO (5,000 images) and 61.236617 on Flickr30k (1,000). All eight full-split scores on each dataset matched the experimental record. |
| Uncertainty estimates | Repeated the paired bootstrap on the four released matrices for the common-probability and count-matched comparisons. All 16 point differences and adjusted intervals matched within 1e-10. Four more matrices recovered the 16 original-selector comparisons in Table 3(d) and the four BLIP comparisons in Table S1, using each family’s own correction. |
| Baseline controls | Recomputed corpus CIDEr for the available reproduced COCO CLIPRe, DeCap, ESPER and MeaCap-ToT predictions, and Flickr30k CLIPRe. All five agree at the precision used in Table 1. |

## Inference checks

| Backbone and input | Image | Output in the release check |
| --- | --- | --- |
| COCO Diffusion Bridge, five cached endpoints | Karpathy ID 0 | A man riding a motorcycle down a dirt road. |
| Flickr30k Diffusion Bridge, five cached endpoints | Karpathy ID 13990 | A dog is jumping over a hurdle in a dog show. |
| Flickr30k Diffusion Bridge, five views computed from the JPEG | Karpathy ID 13990 | A dog is jumping over a hurdle in an agility competition. |
| Salesforce BLIP, five views computed from the JPEG | Karpathy ID 13990 | a brown dog jumping over a hurdle in a field |

The example captions are tied to the listed protocol. Fig. 1(c) uses a separate selector. The primary Table 1 cached comparisons and the later per-view comparisons also have separate reporting targets. The archived metrics file has 32 rows with 30 original prediction lists. The two cached five-view Table 1 rows include the original metrics; use full-split inference to generate their prediction lists.

The CPU checks included PyTorch 2.5.1 and 2.8.0. A clean source extraction with PyTorch 2.8.0 and Transformers 5.8.0 reproduced both cached example captions. The original token-aggregation and BLIP experiments used Python 3.12.3, PyTorch 2.8.0 and Transformers 5.8.0. Timing values in the paper were measured on the original GPUs. This release check covered example decoding and full-split replay/scoring; it did not repeat every full GPU decoding or latency run.

## Official scores on both test splits

Values are on the paper’s ×100 scale for the later Diffusion Bridge five-view arm.

| Dataset | B@1 | B@2 | B@3 | B@4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MSCOCO | 71.976635 | 54.406081 | 39.769644 | 28.754807 | 25.758346 | 52.784930 | 99.108578 | 19.209899 |
| Flickr30k | 68.224969 | 49.547126 | 34.950104 | 23.994203 | 22.578564 | 48.729481 | 61.236617 | 16.317529 |

The original COCO diffusion checkpoint has a recorded SHA-256 but is unavailable in this release. The released five-view endpoints cover every image on the in-domain COCO test split. The COCO-source transfer and nocaps prediction files allow independent reference-based evaluation. New COCO-source images and raw-pixel repeats of those runs require the original checkpoint.

