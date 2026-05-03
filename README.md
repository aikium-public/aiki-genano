# Aiki-GeNano

Code for **"Aiki-GeNano: Multi-Stage Preference Optimization for Generative Design of Developable Nanobodies"** (Meda et al., mAbs (submitted, 2026)).

A staged language-model alignment pipeline (SFT → DPO → GDPO) that takes a target peptide epitope and generates 126-AA Nb-v1 nanobody candidates with developability built into the training objective rather than applied as a post-hoc filter.

| | |
|---|---|
| Paper (preprint) | [bioRxiv 2026.04.28.721526](https://www.biorxiv.org/content/10.64898/2026.04.28.721526v1) (submitted to mAbs) |
| Figure data | Zenodo: <https://doi.org/10.5281/zenodo.19757842> (CC-BY-NC-4.0) |
| Trained checkpoints | available from `partnerships@aikium.com` (NDA) |
| Docker image | `ghcr.io/aikium-public/aiki-genano:1.0.0` |
| Live demo | [aiki-genano on Modal](https://aikium--aiki-genano-fastapi-app.modal.run) |
| Licence | MIT (code) / CC-BY-NC-4.0 (figure data) |

## Quickstart

### 1. Pull the image

```bash
docker pull ghcr.io/aikium-public/aiki-genano:1.0.0
git clone https://github.com/aikium-public/aiki-genano.git
cd aiki-genano
```

The trained model checkpoints (SFT, DPO, GDPO_DPO, GDPO_SFT — ~3.4 GB combined) are not redistributed publicly; request access from `partnerships@aikium.com` and place them under `./checkpoints/{SFT,DPO,GDPO_DPO,GDPO_SFT}/` for the steps below. To play with the model without setting up checkpoints locally, use the [Modal demo](https://aikium--aiki-genano-fastapi-app.modal.run).

### 2. Generate 50 candidate nanobodies for an epitope

```bash
docker run --rm --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:1.0.0 \
    generate \
        --epitope HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG \
        --n_candidates 50 \
        --temperature 0.7 \
        --model GDPO_DPO \
        --output /app/output/preds.csv
```

`preds.csv` columns: `epitope, generated_sequence, gen_length, is_valid_126, model, seed, temperature, reward_{fr2_aggregation, hydrophobic_patch, liability, expression, vhh_hallmark, scaffold_integrity}`.

### 3. (Optional) Add the full property profile

```bash
docker run --rm --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:1.0.0 \
    predict --sequences /app/output/preds.csv --with-properties \
            --output /app/output/preds_profiled.csv
```
## Pre-trained model checkpoints

The four trained checkpoints (SFT merged + DPO / GDPO\_DPO / GDPO\_SFT LoRA adapters) are proprietary to Aikium Inc. and not redistributed publicly. They are available under non-disclosure agreement via [`partnerships@aikium.com`](mailto:partnerships@aikium.com) for academic evaluation, custom design campaigns, or licensed deployments. To try the model immediately without local checkpoints, use the [Modal demo](https://aikium--aiki-genano-fastapi-app.modal.run).

### 4. Without GPU (laptop demo)

```bash
docker run --rm \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:1.0.0 \
    generate --epitope HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG \
             --n_candidates 5 --model SFT --device cpu \
             --output /app/output/preds.csv
```

PyTorch falls back to CPU; expect ~5 minutes per 5 candidates instead of ~5 seconds on GPU.

## Reproducing paper numbers

The numerical data underlying every figure and table in the paper is deposited at Zenodo ([10.5281/zenodo.19757842](https://doi.org/10.5281/zenodo.19757842), CC-BY-NC-4.0):

- **`figure_data/`** — per-figure aggregates and per-position derivatives covering all 64 evaluated targets, plus the per-tool head-to-head benchmark table behind Fig 5.
- **`full_property_tables/`** — per-sequence computed properties (TEMPRO Tm, NetSolP solubility, Sapiens humanness, six GDPO reward scores, motif counts, biophysical descriptors) for the 10 representative GPCR targets disclosed in the paper. Amino-acid sequences are stripped.
- **`PROVENANCE.md`** — figure-by-figure mapping to the exact CSV(s).

The deposit is sufficient to verify every numerical claim in the paper. The 1.35 M nanobody–epitope screening corpus, the 522,800 DPO preference pairs, the generated-sequence amino-acid strings, and the trained checkpoints are not redistributed; request via [`partnerships@aikium.com`](mailto:partnerships@aikium.com) under NDA.

```bash
# Re-score a sequence list under the property pipeline used in the paper
docker run --rm --gpus all \
    -v "$(pwd)/sequences:/app/data" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:1.0.0 \
    predict --sequences /app/data/my_sequences.csv \
            --output /app/output/profiled.csv
```

## Project layout

```
aiki-genano/
├── aiki_genano/
│   ├── cli/               generate / predict / train / smoke sub-commands
│   ├── training/          SFT (`sft.py`), DPO (`dpo.py`), GDPO (`gdpo.py`) entrypoints
│   ├── rewards/           the six paper reward functions + Nb-v1 scaffold helpers
│   ├── evaluation/        TEMPRO / NetSolP / Sapiens predictors + `profile.py`
│   └── analysis/          BLAST novelty, dataset figures, the figure-rebuild notebook
├── configs/               Hydra configs for SFT (`sft_10k.yaml`, `sft_continue_20k.yaml`),
│                          DPO (`dpo_developability.yaml`), GDPO (`gdpo/*.yaml`)
├── docker/                Single Dockerfile + mount conventions
├── scripts/               Checkpoint downloader, smoke tests, GHCR push wrapper
├── benchmarks/            Head-to-head vs IgLM, nanoBERT, NanoAbLLaMA, PepMLM, IgGM,
│                          ProteinDPO, ProtGPT2 (paper §"Comparison to existing tools")
├── modal_app.py           Modal hosting (single ASGI app, four routes)
└── pyproject.toml         pip install -e .
```

## Sequence-format note

Every input epitope and every output sequence is uppercase amino-acid letters. Generated nanobodies are 126 residues = 121-AA core VHH + 5-AA C-terminal `GGGGS` linker (the synthetic Nb-v1 backbone described in Contreras et al. 2023). When passing generated sequences to predictors that expect a raw VHH domain (TEMPRO, NetSolP, Sapiens), call `aiki_genano.rewards.nanobody_scaffold.normalize_for_prediction(seq).core` first — `predict` does this for you.

## FAQ

**Q. Can I generate nanobodies for any target?**
Yes — pass a target sequence (linear peptide, intrinsically-disordered region, or whole soluble domain) of 4–244 amino acids as `--epitope`. The model has seen 65 training targets spanning short peptide windows (e.g. multi-pass receptor N-termini) and whole soluble extracellular domains of single-pass membrane proteins, so quality is best on inputs resembling those. Inputs dominated by transmembrane segments or with strong conformational dependence are not covered by the training distribution.

**Q. How big is the image?**
About 5.6 GB compressed / 16 GB on disk. CUDA 12.1 + PyTorch 2.2 + transformers 4.57 + the NVIDIA TRL-GDPO fork + property predictors. Foundation-model weights download to `/models` on first run, not bundled.

**Q. The image runs on CPU but very slowly. Is that intended?**
Yes. PyTorch picks the device at runtime; the CUDA libs in the image become dead weight without a GPU but do not block execution. CPU is for laptop demos; a real evaluation needs an A10 / A100-class GPU.

**Q. The Modal demo is slow on the first request.**
The GPU function scales to zero when idle; cold start is 30–60 seconds (model load). Subsequent requests within a few minutes are warm (~5 seconds per 10 candidates).

**Q. Can I use the model commercially?**
Code is MIT — yes for the code. Model checkpoints and data are CC-BY-NC-4.0 — non-commercial only without explicit permission. Email `partnerships@aikium.com` for commercial enquiries. See `THIRD_PARTY_LICENSES.md` for the per-component licence breakdown (some predictors are academic-only).

**Q. Why did the GDPO trainer need a forked TRL?**
NVIDIA's GDPO extension to GRPO normalises each reward independently before aggregation, which keeps individual reward signals from getting drowned out when you combine objectives with very different variances (the "reward-signal collapse" problem). At the time of training there was no PyPI release; the `docker/Dockerfile` clones the fork and applies three small patches (cross-version compat for `transformers.utils.is_rich_available`, an older-model fallback for `logits_to_keep`, and an optional `vllm` import).

**Q. How do I extend with a new reward function?**
Add a function with signature `f(completions: list[str], **kwargs) -> list[float]` to `aiki_genano/rewards/rewards.py`, register it in the GDPO config under `gdpo.script_args.reward_weights`, and re-train. The validity gate (`_is_valid_nbv1`) ensures invalid sequences receive zero across every reward channel.

**Q. Where do I report a bug?**
[GitHub Issues](https://github.com/aikium-public/aiki-genano/issues). For security-relevant issues, email `partnerships@aikium.com` instead.

## Citation

```bibtex
@article{Meda2026Aiki-GeNano,
  title   = {Aiki-GeNano: Multi-Stage Preference Optimization for Generative Design of Developable Nanobodies},
  author  = {Meda, Radheesh Sharma and Doshi, Jigar and Iyer, Eswar and
             Shastry, Shankar and Mysore, Venkatesh},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.04.28.721526},
  url     = {https://www.biorxiv.org/content/10.64898/2026.04.28.721526v1},
  note    = {Preprint; submitted to mAbs}
}

@dataset{Aiki-GeNano-Zenodo-2026,
  title     = {Aiki-GeNano figure data and computed property profiles},
  author    = {Meda, Radheesh Sharma and Doshi, Jigar and Iyer, Eswar and
               Shastry, Shankar and Mysore, Venkatesh},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.19757842},
  url       = {https://doi.org/10.5281/zenodo.19757842}
}
```

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Model checkpoints and data: CC-BY-NC-4.0 (see the `LICENSE.txt` inside the Zenodo deposit). Per-upstream-component terms: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

Each upstream model and dataset retains its own licence. Users deploying Aiki-GeNano, its outputs, or derived sequences in their own workflows are responsible for complying with the respective upstream terms.

## Acknowledgements

We thank the entire Synthetic Biology and Protein Sciences teams at Aikium Inc. for their contributions to data generation. The paper's nanobody-specific developability framing draws on the Therapeutic Nanobody Profiler (Gordon, Gervasio, Souders, Deane 2026, Oxford OPIG); the head-to-head benchmark contrasts against five contemporary VHH generators (nanoBERT, IgLM, NanoAbLLaMA, ProteinDPO, IgGM). A portion of this work was enabled by the Google for AI Startups program and by infrastructure support from Modal for model hosting.
