# Third-party software, models and datasets

The MIT license in this repository covers the authors' CMV-CG code. Model and dataset providers retain their respective rights. Release ZIPs identify the dataset-specific captioners, features, source model files and upstream tokenizer files; the SHA-256 values are recorded in `configs/`. Photos and full collections of human reference captions are obtained from their providers.

- [Diffusion Bridge, CVPR 2025](https://github.com/mongeoroo/diffusion-bridge) supplies the diffusion model implementation and original captioning architecture. The installer retrieves the upstream one-dimensional model implementation and checks its file digest.
- [OpenAI CLIP](https://github.com/openai/CLIP) supplies ViT-B/32 and the tokenizer used to compute similarity scores. [GPT-2](https://huggingface.co/openai-community/gpt2) supplies the tokenizer and configuration used by the reproduced captioners. The [upstream software and model notice](licenses/GPT-2-LICENSE.txt) accompanies these assets.
- [Salesforce BLIP](https://github.com/salesforce/BLIP) and its [model card](https://huggingface.co/Salesforce/blip-image-captioning-base) provide the supervised comparison model. The installer downloads the named release from its original source.
- [COCO caption evaluation](https://github.com/tylin/coco-caption) and its [Python 3 implementation](https://github.com/salaniz/pycocoevalcap) implement PTB tokenization, BLEU, METEOR, ROUGE-L, CIDEr-D and SPICE; associated Java resources are separately distributed by their providers.
- Dataset attribution, terms and access instructions are available from [MSCOCO](https://cocodataset.org/#termsofuse), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/), [Karpathy splits](https://cs.stanford.edu/people/karpathy/deepimagesent/), and [nocaps](https://nocaps.org/download).

