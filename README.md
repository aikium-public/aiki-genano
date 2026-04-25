# Aiki-GeNano

Code for **"Preference-optimized generation of developable nanobodies across 65 epitope targets"** (Meda et al., mAbs 2026).

A staged language-model alignment pipeline (SFT → DPO → GDPO) that takes a target peptide epitope and generates 126-AA Nb-v1 nanobody candidates with developability built into the training objective rather than applied as a post-hoc filter.

| | |
|---|---|
| Paper (preprint) | bioRxiv (DOI assigned on screening) |
| Data + weights | Zenodo deposit (forthcoming) |
| Docker image | `ghcr.io/aikium-public/aiki-genano:1.0.0` |
| Live demo | [aiki-genano on Modal](https://aikium-public--aiki-genano-fastapi-app.modal.run) |
| Licence | MIT (code) / CC-BY-NC-4.0 (weights + data) |

## Quickstart

### 1. Pull the image and the checkpoints

```bash
docker pull ghcr.io/aikium-public/aiki-genano:1.0.0
git clone https://github.com/aikium-public/aiki-genano.git
cd aiki-genano
bash scripts/download_checkpoints.sh        # ~3.4 GB from Zenodo, sha256-verified
```

### 2. Generate 50 candidate nanobodies for an epitope

```bash
docker run --rm --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:1.0.0 \
    generate \
        --epitope MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL \
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

Adds TEMPRO predicted Tm, NetSolP solubility, Sapiens humanness, and ~50 motif/biophysical/CDR/Aggrescan columns. First run downloads ~2 GB of foundation-model weights into `/models`; subsequent runs are warm.

### 4. Without GPU (laptop demo)

```bash
docker run --rm \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:1.0.0 \
    generate --epitope MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL \
             --n_candidates 5 --model SFT --device cpu \
             --output /app/output/preds.csv
```

PyTorch falls back to CPU; expect ~5 minutes per 5 candidates instead of ~5 seconds on GPU.

## Reproducing paper numbers

The Zenodo deposit ships:
- **Training subset**: 10,000 binder sequences across 10 representative target epitopes plus the matching 21,998 DPO preference pairs.
- **Generated sequences + computed properties**: every sequence from the four paper models (SFT, DPO, GDPO_DPO, GDPO_SFT) at every reported seed (42, 123, 456) and temperature (0.7, 0.9, 1.2), restricted to the 10 disclosed targets.
- **Model weights**: SFT full merged checkpoint plus DPO / GDPO_DPO / GDPO_SFT LoRA adapters.
- **PROVENANCE.md**: each figure / table maps to the exact CSV(s) and the analysis-notebook cell that produced it.

```bash
# Recompute one figure end-to-end from the Zenodo bundle
docker run --rm --gpus all \
    -v "$(pwd)/zenodo_deposit:/app/data" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:1.0.0 \
    predict \
        --sequences /app/data/generated_sequences/GDPO_DPO/properties/GDPO_DPO_seed42_temp0.7_profiled.csv \
        --output /app/output/recomputed.csv
```

Numbers will reflect the 10-target subset; the paper headline numbers are aggregated over all 65 targets and use the proprietary screen.

## Re-training on the released subset

```bash
docker run --rm --gpus all \
    -v "$(pwd)/zenodo_deposit:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:1.0.0 \
    train --stage sft --config sft_10k
```

SFT runs ~few GPU-hours per 10,000 steps on an A100 40 GB. DPO is similar. GDPO is ~30 minutes for the 2,000-step run reported in the paper. Subset training is for verification — paper headline numbers require the proprietary 65-target dataset.

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
Yes — pass any short peptide as `--epitope`. The model has only seen Aikium's 65 training targets, so quality is best on epitopes resembling those (linear peptide, IDR, or extended disordered region). Quality on globular folded targets is untested.

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
  title   = {Preference-optimized generation of developable nanobodies across 65 epitope targets},
  author  = {Meda, Radheesh Sharma and Doshi, Jigar and Iyer, Eswar and
             Shastry, Shankar and Mysore, Venkatesh},
  journal = {mAbs},
  year    = {2026},
  note    = {Preprint forthcoming on bioRxiv}
}
```

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Model checkpoints and data: CC-BY-NC-4.0 (see the `LICENSE.txt` inside the Zenodo deposit). Per-upstream-component terms: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

Each upstream model and dataset retains its own licence. Users deploying Aiki-GeNano, its outputs, or derived sequences in their own workflows are responsible for complying with the respective upstream terms.

## Acknowledgements

We thank the entire Synthetic Biology and Protein Sciences teams at Aikium Inc. for their contributions to data generation. A portion of this work was enabled by the Google for AI Startups program and by infrastructure support from Modal for model hosting.
