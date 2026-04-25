# Third-party licences

Aiki-GeNano source code is MIT-licensed (see `LICENSE`). The Zenodo deposit (model checkpoints + data) is CC-BY-NC-4.0. Many of the foundation models, predictors, and libraries the pipeline depends on retain their own upstream licences. Each of those upstream terms continues to apply when you run, redistribute, or build on the corresponding component.

Users deploying Aiki-GeNano, its outputs, or derived sequences in their own workflows are responsible for complying with the respective upstream terms; follow the links below for each component.

## Foundation models

| Component | Licence | Where it enters the pipeline |
|---|---|---|
| **ProtGPT2** ([nferruz/ProtGPT2](https://huggingface.co/nferruz/ProtGPT2)) | Apache 2.0 | Base autoregressive model fine-tuned in the SFT stage. Downloaded at runtime to `/models`. |
| **ESM-2 (650M)** ([facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)) | MIT | Embeddings backbone for TEMPRO and NetSolP predictors. Downloaded at runtime. |
| **Sapiens** ([Lab608/Sapiens](https://github.com/Merck/Sapiens)) | MIT | Antibody humanness scoring (used by `predict --with-properties`). |

## Predictors and analysis tools

| Component | Licence | Where it enters the pipeline |
|---|---|---|
| **TEMPRO** ([Alvarez 2024](https://github.com/JaredAlvarez/TEMPRO)) | MIT | Predicted melting temperature; the trained Keras model is downloaded separately by the user. |
| **NetSolP-1.0** ([Thumuluri 2022](https://services.healthtech.dtu.dk/services/NetSolP-1.0/)) | Academic-only (DTU Health Tech) | Predicted solubility / usability scores. **Verify the academic-use clause matches your intended use** before running `predict --with-properties` in a commercial setting. |
| **Aggrescan3D** ([Conchillo-Solé 2007](https://academic.oup.com/bioinformatics/article/23/9/1106/272400)) | CC-BY 4.0 (scale only) | Per-residue aggregation propensity scale used in the local profile. |
| **BLAST+** ([NCBI](https://blast.ncbi.nlm.nih.gov/)) | Public domain (US Government work) | Optional novelty audit (`analysis/blast_novelty.py`). |
| **ANARCI / anarcii** ([Dunbar 2016](https://github.com/oxpig/ANARCI)) | BSD-3-Clause | CDR boundary annotation. |

## Training framework

| Component | Licence | Where it enters the pipeline |
|---|---|---|
| **transformers** (Hugging Face) | Apache 2.0 | Tokeniser, model API. |
| **peft** (Hugging Face) | Apache 2.0 | LoRA adapters for DPO and GDPO. |
| **datasets** (Hugging Face) | Apache 2.0 | Data loading. |
| **accelerate** (Hugging Face) | Apache 2.0 | Multi-device training. |
| **trl** (Hugging Face) — DPO | Apache 2.0 | DPO trainer. |
| **TRL-GDPO fork** ([NVlabs/GDPO](https://github.com/NVlabs/GDPO)) | Apache 2.0 | GDPO trainer (NVIDIA's GRPO + per-reward decoupling extension). |
| **bitsandbytes** | MIT | 4-bit / 8-bit quantisation. |
| **PyTorch** | BSD-3 | Tensor backend. |

## Datasets

| Asset | Licence | Notes |
|---|---|---|
| Aikium 65-target nanobody screen (full 1.35 M binders + 522 800 DPO pairs) | Proprietary | Not redistributed. Generated on Aikium's mRNA-display platform. |
| 10-target representative subset (10 000 binders + 21 998 DPO pairs) | CC-BY-NC-4.0 | Released via Zenodo under the same DOI as the model weights. |
| Generated sequences for the paper's 4 models × 3 seeds × 3 temperatures | CC-BY-NC-4.0 | Restricted to the same 10 disclosed targets in the Zenodo deposit. |

## Patent disclosure

Aikium Inc. has filed patent applications covering aspects of the work described in the paper. The MIT licence on this code does not grant any patent licence beyond what is implied by MIT itself; commercial users should consult `partnerships@aikium.com`.
